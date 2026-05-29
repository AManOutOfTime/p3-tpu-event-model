#include "schedule/llama_schedule.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>

namespace sim {
namespace {

struct Builder {
    std::vector<Instruction> out;
    InstructionId next = 0;

    InstructionId add(std::string op, std::string unit, std::string label,
                      ParamMap params = {}, std::vector<InstructionId> deps = {}) {
        Instruction inst;
        inst.id = next++;
        inst.op = std::move(op);
        inst.unit = std::move(unit);
        inst.label = std::move(label);
        inst.params = std::move(params);
        inst.depends_on = std::move(deps);
        out.push_back(std::move(inst));
        return out.back().id;
    }
};

uint32_t ceil_div(uint32_t x, uint32_t y) {
    return y ? (x + y - 1) / y : 0;
}

uint32_t tile_extent(uint32_t total, uint32_t tile, uint32_t tile_idx) {
    const uint32_t start = tile_idx * tile;
    return std::min(tile, total - start);
}

uint32_t effective_max_seq_len(const LlamaScheduleConfig& cfg) {
    if (cfg.max_seq_len) return cfg.max_seq_len;
    return std::max(cfg.seq_len, cfg.prompt_len + cfg.generation_steps);
}

uint32_t effective_cache_block_tokens(const LlamaScheduleConfig& cfg) {
    return cfg.kv_cache_block_tokens ? cfg.kv_cache_block_tokens : cfg.tile_cols;
}

std::string range_suffix(uint32_t start, uint32_t rows) {
    return ".range" + std::to_string(start) + "_" + std::to_string(start + rows);
}

uint64_t kv_cache_required_bytes(const LlamaScheduleConfig& cfg, uint32_t tokens) {
    return static_cast<uint64_t>(cfg.num_layers)
         * cfg.num_kv_heads * tokens * cfg.head_dim
         * cfg.dtype_bytes * 2;
}

uint64_t kv_cache_page_bytes(const LlamaScheduleConfig& cfg) {
    return static_cast<uint64_t>(cfg.num_layers)
         * cfg.num_kv_heads * effective_cache_block_tokens(cfg)
         * cfg.head_dim * cfg.dtype_bytes * 2;
}

bool cache_name_is_hbm(const std::string& cache) {
    return cache.rfind("HBM.", 0) == 0;
}

bool cache_page_is_hbm(const LlamaScheduleConfig& cfg, uint32_t start,
                       uint32_t live_tokens) {
    if (cfg.kv_cache_location == "hbm") return true;
    if (cfg.kv_cache_eviction_policy != "spill_to_hbm") return false;
    if (!cfg.sram_kv_capacity_kb) return false;

    const uint64_t page_bytes = kv_cache_page_bytes(cfg);
    if (!page_bytes) return false;
    const uint64_t capacity = static_cast<uint64_t>(cfg.sram_kv_capacity_kb) * 1024;
    const uint32_t hot_pages = static_cast<uint32_t>(capacity / page_bytes);
    if (!hot_pages) return true;

    const uint32_t block_tokens = effective_cache_block_tokens(cfg);
    const uint32_t page = start / block_tokens;
    const uint32_t live_pages = ceil_div(std::max<uint32_t>(live_tokens, 1), block_tokens);
    const uint32_t first_hot_page = live_pages > hot_pages ? live_pages - hot_pages : 0;
    return page < first_hot_page;
}

std::string cache_range_name(const LlamaScheduleConfig& cfg, const std::string& layer,
                             uint32_t kvh, const std::string& kind,
                             uint32_t start, uint32_t rows, uint32_t live_tokens) {
    const uint32_t block_tokens = effective_cache_block_tokens(cfg);
    const uint32_t page = start / block_tokens;
    const uint32_t block = page;
    const std::string prefix = cache_page_is_hbm(cfg, start, live_tokens)
        ? "HBM.kv_cache"
        : "SRAM.kv_cache";
    return prefix + "." + layer + ".kv" + std::to_string(kvh)
         + "." + kind + ".page" + std::to_string(page)
         + ".block" + std::to_string(block)
         + range_suffix(start, rows);
}

LlamaScheduleConfig normalize_cfg(LlamaScheduleConfig cfg) {
    if (!cfg.num_q_heads || !cfg.num_kv_heads)
        throw std::runtime_error("LlamaScheduleConfig: heads must be non-zero");
    if (!cfg.num_layers)
        throw std::runtime_error("LlamaScheduleConfig: num_layers must be non-zero");
    if (!cfg.hidden_dim || !cfg.intermediate_dim || !cfg.tile_rows || !cfg.tile_cols
        || !cfg.linear_tile_rows || !cfg.linear_tile_cols)
        throw std::runtime_error("LlamaScheduleConfig: dimensions and tile sizes must be non-zero");
    if (!cfg.dtype_bytes)
        throw std::runtime_error("LlamaScheduleConfig: dtype_bytes must be non-zero");
    if (!cfg.kv_cache_block_tokens)
        cfg.kv_cache_block_tokens = cfg.tile_cols;
    if (!cfg.kv_cache_block_tokens)
        throw std::runtime_error("LlamaScheduleConfig: kv_cache_block_tokens must be non-zero");
    if (cfg.kv_prefetch != "none" && cfg.kv_prefetch != "double_buffer")
        throw std::runtime_error("LlamaScheduleConfig: kv_prefetch must be none or double_buffer");
    if (!cfg.kv_stage_buffers)
        throw std::runtime_error("LlamaScheduleConfig: kv_stage_buffers must be non-zero");
    if (cfg.kv_prefetch == "double_buffer" && cfg.kv_stage_buffers < 2)
        throw std::runtime_error("LlamaScheduleConfig: double_buffer prefetch requires at least two KV stage buffers");
    if (cfg.kv_cache_eviction_policy != "fail" && cfg.kv_cache_eviction_policy != "spill_to_hbm")
        throw std::runtime_error("LlamaScheduleConfig: kv_cache_eviction_policy must be fail or spill_to_hbm");
    if (cfg.schedule_granularity != "detailed" && cfg.schedule_granularity != "coarse")
        throw std::runtime_error("LlamaScheduleConfig: schedule_granularity must be detailed or coarse");
    if (cfg.hidden_dim % cfg.num_q_heads != 0)
        throw std::runtime_error("LlamaScheduleConfig: hidden_dim must be divisible by num_q_heads");
    const uint32_t derived_head_dim = cfg.hidden_dim / cfg.num_q_heads;
    if (!cfg.head_dim) {
        cfg.head_dim = derived_head_dim;
    } else if (cfg.head_dim != derived_head_dim) {
        throw std::runtime_error("LlamaScheduleConfig: head_dim must equal hidden_dim/num_q_heads");
    }
    if (cfg.num_q_heads % cfg.num_kv_heads != 0)
        throw std::runtime_error("LlamaScheduleConfig: num_kv_heads must divide num_q_heads");
    const uint32_t group = cfg.num_q_heads / cfg.num_kv_heads;
    if (!cfg.gqa_group_size) {
        cfg.gqa_group_size = group;
    } else if (cfg.gqa_group_size != group) {
        throw std::runtime_error("LlamaScheduleConfig: gqa_group_size must equal num_q_heads/num_kv_heads");
    }
    if (cfg.kv_cache_location != "sram" && cfg.kv_cache_location != "hbm")
        throw std::runtime_error("LlamaScheduleConfig: kv_cache_location must be sram or hbm");
    const bool uses_cache = cfg.kv_cache_enabled || cfg.mode == "prefill_decode";
    if (uses_cache && cfg.kv_cache_location == "sram" && cfg.sram_kv_capacity_kb
        && cfg.kv_cache_eviction_policy == "fail") {
        const uint64_t required = kv_cache_required_bytes(cfg, effective_max_seq_len(cfg));
        const uint64_t capacity = static_cast<uint64_t>(cfg.sram_kv_capacity_kb) * 1024;
        if (required > capacity) {
            throw std::runtime_error("LlamaScheduleConfig: SRAM KV cache capacity exceeded");
        }
    }
    return cfg;
}

InstructionId cache_move(Builder& b, const LlamaScheduleConfig& cfg, bool read,
                         const std::string& tensor, const std::string& cache,
                         uint32_t rows, std::vector<InstructionId> deps) {
    const bool hbm = cache_name_is_hbm(cache);
    ParamMap p;
    p["source"] = read ? cache : tensor;
    p["destination"] = read ? tensor : cache;
    p["rows"] = static_cast<int64_t>(rows);
    p["cols"] = static_cast<int64_t>(cfg.head_dim);
    p["cache_location"] = hbm ? std::string("hbm") : std::string("sram");
    return b.add(hbm ? (read ? "dma_load" : "dma_store") : "sram_copy",
                 hbm ? "dma" : "access_core",
                 (read ? "KV cache read " : "KV cache write ") + cache,
                 std::move(p), std::move(deps));
}

InstructionId on_chip_move(Builder& b, const std::string& source,
                           const std::string& destination,
                           uint32_t rows, uint32_t cols,
                           const std::string& label,
                           std::vector<InstructionId> deps) {
    ParamMap p;
    p["source"] = source;
    p["destination"] = destination;
    p["rows"] = static_cast<int64_t>(rows);
    p["cols"] = static_cast<int64_t>(cols);
    return b.add("sram_copy", "access_core", label, std::move(p), std::move(deps));
}

InstructionId append_coarse_staged_gemm(Builder& b, const std::string& label,
                                        const std::string& tag,
                                        const std::string& source_a,
                                        const std::string& hbm_weight,
                                        const std::string& destination,
                                        uint32_t m, uint32_t k, uint32_t n,
                                        const std::vector<InstructionId>& deps) {
    const std::string weight_buf = "shared_ibuf." + tag + ".W";
    const std::string a_operand = "systolic_array." + tag + ".A_operand";
    const std::string b_operand = "systolic_array." + tag + ".B_operand";

    ParamMap load_w;
    load_w["source"] = hbm_weight;
    load_w["destination"] = weight_buf;
    load_w["rows"] = static_cast<int64_t>(k);
    load_w["cols"] = static_cast<int64_t>(n);
    InstructionId weight_load = b.add("dma_load", "dma", label + " weight load",
                                      load_w, deps);

    ParamMap stage_a;
    stage_a["source"] = source_a;
    stage_a["destination"] = a_operand;
    stage_a["rows"] = static_cast<int64_t>(m);
    stage_a["cols"] = static_cast<int64_t>(k);
    InstructionId a_stage = b.add("dma_stage", "dma", label + " activation stage",
                                  stage_a, deps);

    ParamMap stage_b;
    stage_b["source"] = weight_buf;
    stage_b["destination"] = b_operand;
    stage_b["rows"] = static_cast<int64_t>(k);
    stage_b["cols"] = static_cast<int64_t>(n);
    InstructionId b_stage = b.add("dma_stage", "dma", label + " weight stage",
                                  stage_b, {weight_load});

    ParamMap gemm;
    gemm["source_a"] = a_operand;
    gemm["source_b"] = b_operand;
    gemm["destination"] = destination;
    gemm["M"] = static_cast<int64_t>(m);
    gemm["K"] = static_cast<int64_t>(k);
    gemm["N"] = static_cast<int64_t>(n);
    return b.add("gemm", "systolic", label, gemm, {a_stage, b_stage});
}

InstructionId append_detailed_tiled_gemm(Builder& b, const LlamaScheduleConfig& cfg,
                                         const std::string& label,
                                         const std::string& tag,
                                         const std::string& source_a,
                                         const std::string& hbm_weight,
                                         const std::string& destination,
                                         uint32_t m, uint32_t k, uint32_t n,
                                         const std::vector<InstructionId>& deps) {
    const uint32_t row_tiles = ceil_div(m, cfg.linear_tile_rows);
    const uint32_t col_tiles = ceil_div(n, cfg.linear_tile_cols);
    std::vector<InstructionId> tile_outputs;
    InstructionId prev_gemm = 0;
    bool has_prev_gemm = false;

    for (uint32_t rt = 0; rt < row_tiles; rt++) {
        const uint32_t row_start = rt * cfg.linear_tile_rows;
        const uint32_t rows = tile_extent(m, cfg.linear_tile_rows, rt);
        const std::string a_tile = "shared_ibuf." + tag + ".A.r" + std::to_string(rt);

        InstructionId a_copy = on_chip_move(
            b, source_a, a_tile, rows, k,
            label + " activation tile r" + std::to_string(rt) + " load", deps);

        ParamMap stage_a;
        stage_a["source"] = a_tile;
        stage_a["destination"] = "systolic_array." + tag + ".A.r" + std::to_string(rt);
        stage_a["rows"] = static_cast<int64_t>(rows);
        stage_a["cols"] = static_cast<int64_t>(k);
        std::vector<InstructionId> stage_a_deps = {a_copy};
        if (has_prev_gemm) stage_a_deps.push_back(prev_gemm);
        InstructionId a_stage = b.add("dma_stage", "dma",
                                      label + " activation tile r" + std::to_string(rt) + " stage",
                                      stage_a, stage_a_deps);

        for (uint32_t ct = 0; ct < col_tiles; ct++) {
            const uint32_t col_start = ct * cfg.linear_tile_cols;
            const uint32_t cols = tile_extent(n, cfg.linear_tile_cols, ct);
            const std::string tile_tag = tag + ".r" + std::to_string(rt)
                                       + ".c" + std::to_string(ct);
            const std::string w_tile = "shared_ibuf." + tile_tag + ".W";
            const std::string b_operand = "systolic_array." + tile_tag + ".B";
            const std::string c_tile = "shared_obuf." + tile_tag + ".C";

            ParamMap load_w;
            load_w["source"] = hbm_weight + "[0:" + std::to_string(k)
                              + "," + std::to_string(col_start)
                              + ":" + std::to_string(col_start + cols) + "]";
            load_w["destination"] = w_tile;
            load_w["rows"] = static_cast<int64_t>(k);
            load_w["cols"] = static_cast<int64_t>(cols);
            InstructionId weight_load = b.add("dma_load", "dma",
                                              label + " weight tile c" + std::to_string(ct) + " load",
                                              load_w, deps);

            ParamMap stage_b;
            stage_b["source"] = w_tile;
            stage_b["destination"] = b_operand;
            stage_b["rows"] = static_cast<int64_t>(k);
            stage_b["cols"] = static_cast<int64_t>(cols);
            std::vector<InstructionId> stage_b_deps = {weight_load};
            if (has_prev_gemm) stage_b_deps.push_back(prev_gemm);
            InstructionId b_stage = b.add("dma_stage", "dma",
                                          label + " weight tile c" + std::to_string(ct) + " stage",
                                          stage_b, stage_b_deps);

            ParamMap gemm;
            gemm["source_a"] = "systolic_array." + tag + ".A.r" + std::to_string(rt);
            gemm["source_b"] = b_operand;
            gemm["destination"] = c_tile;
            gemm["M"] = static_cast<int64_t>(rows);
            gemm["K"] = static_cast<int64_t>(k);
            gemm["N"] = static_cast<int64_t>(cols);
            std::vector<InstructionId> gemm_deps = {a_stage, b_stage};
            if (has_prev_gemm) gemm_deps.push_back(prev_gemm);
            InstructionId gemm_id = b.add("gemm", "systolic",
                                          label + " tile r" + std::to_string(rt)
                                          + " c" + std::to_string(ct),
                                          gemm, gemm_deps);

            ParamMap place;
            place["source"] = c_tile;
            place["destination"] = destination + ".range"
                                 + std::to_string(row_start) + "_"
                                 + std::to_string(row_start + rows)
                                 + "." + std::to_string(col_start) + "_"
                                 + std::to_string(col_start + cols);
            place["rows"] = static_cast<int64_t>(rows);
            place["cols"] = static_cast<int64_t>(cols);
            InstructionId place_id = b.add("sram_copy", "access_core",
                                           label + " place tile r" + std::to_string(rt)
                                           + " c" + std::to_string(ct),
                                           place, {gemm_id});
            tile_outputs.push_back(place_id);
            prev_gemm = gemm_id;
            has_prev_gemm = true;
        }
    }

    ParamMap assemble;
    assemble["source"] = destination + ".tiles";
    assemble["destination"] = destination;
    assemble["rows"] = static_cast<int64_t>(m);
    assemble["cols"] = static_cast<int64_t>(n);
    assemble["tile_rows"] = static_cast<int64_t>(row_tiles);
    assemble["tile_cols"] = static_cast<int64_t>(col_tiles);
    return b.add("sram_copy", "access_core", label + " assemble output",
                 assemble, tile_outputs);
}

InstructionId append_staged_gemm(Builder& b, const LlamaScheduleConfig& cfg,
                                 const std::string& label,
                                 const std::string& tag,
                                 const std::string& source_a,
                                 const std::string& hbm_weight,
                                 const std::string& destination,
                                 uint32_t m, uint32_t k, uint32_t n,
                                 const std::vector<InstructionId>& deps) {
    if (cfg.schedule_granularity == "coarse") {
        return append_coarse_staged_gemm(b, label, tag, source_a, hbm_weight,
                                         destination, m, k, n, deps);
    }
    return append_detailed_tiled_gemm(b, cfg, label, tag, source_a, hbm_weight,
                                     destination, m, k, n, deps);
}

InstructionId append_vector_phase(Builder& b, const std::string& op,
                                  const std::string& unit,
                                  const std::string& label,
                                  const std::string& source,
                                  const std::string& destination,
                                  uint32_t rows, uint32_t cols,
                                  std::vector<InstructionId> deps) {
    ParamMap p;
    p["source"] = source;
    p["destination"] = destination;
    p["rows"] = static_cast<int64_t>(rows);
    p["cols"] = static_cast<int64_t>(cols);
    return b.add(op, unit, label, p, std::move(deps));
}

InstructionId append_detailed_rmsnorm(Builder& b, const LlamaScheduleConfig& cfg,
                                      const std::string& label,
                                      const std::string& tag,
                                      const std::string& source,
                                      const std::string& destination,
                                      uint32_t rows, uint32_t cols,
                                      const std::vector<InstructionId>& deps) {
    std::vector<InstructionId> row_outputs;
    const uint32_t row_tiles = ceil_div(rows, cfg.tile_rows);
    for (uint32_t rt = 0; rt < row_tiles; rt++) {
        const uint32_t r = tile_extent(rows, cfg.tile_rows, rt);
        const std::string t = tag + ".rms.r" + std::to_string(rt);
        InstructionId load = on_chip_move(b, source, "shared_ibuf." + t + ".x",
                                          r, cols, label + " load tile r" + std::to_string(rt),
                                          deps);
        InstructionId square = append_vector_phase(
            b, "square", "vector_core", label + " square tile r" + std::to_string(rt),
            "shared_ibuf." + t + ".x", "vector_scratch." + t + ".x2", r, cols, {load});

        ParamMap reduce;
        reduce["source"] = "vector_scratch." + t + ".x2";
        reduce["destination"] = "vector_scratch." + t + ".mean_square";
        reduce["rows"] = static_cast<int64_t>(r);
        reduce["cols"] = static_cast<int64_t>(cols);
        InstructionId row_reduce = b.add("row_reduce_sum", "vector_core",
                                         label + " row reduce tile r" + std::to_string(rt),
                                         reduce, {square});

        ParamMap eps;
        eps["source"] = "vector_scratch." + t + ".mean_square";
        eps["destination"] = "vector_scratch." + t + ".mean_square_eps";
        eps["length"] = static_cast<int64_t>(r);
        InstructionId add_eps = b.add("add_epsilon", "vector_core",
                                      label + " add epsilon tile r" + std::to_string(rt),
                                      eps, {row_reduce});

        ParamMap rsqrt;
        rsqrt["source"] = "vector_scratch." + t + ".mean_square_eps";
        rsqrt["destination"] = "vector_scratch." + t + ".inv_rms";
        rsqrt["length"] = static_cast<int64_t>(r);
        InstructionId inv = b.add("rsqrt", "vector_core",
                                  label + " rsqrt tile r" + std::to_string(rt),
                                  rsqrt, {add_eps});

        ParamMap weight;
        weight["source"] = "HBM." + tag + ".rms_weight";
        weight["destination"] = "shared_ibuf." + t + ".weight";
        weight["length"] = static_cast<int64_t>(cols);
        InstructionId weight_load = b.add("dma_load", "dma",
                                          label + " RMS weight load tile r" + std::to_string(rt),
                                          weight, deps);

        ParamMap scale;
        scale["source"] = "shared_ibuf." + t + ".x";
        scale["source_scale"] = "vector_scratch." + t + ".inv_rms";
        scale["source_b"] = "shared_ibuf." + t + ".weight";
        scale["destination"] = destination + ".r" + std::to_string(rt);
        scale["rows"] = static_cast<int64_t>(r);
        scale["cols"] = static_cast<int64_t>(cols);
        InstructionId norm = b.add("norm_scale", "vector_core",
                                   label + " scale/write tile r" + std::to_string(rt),
                                   scale, {inv, weight_load});
        row_outputs.push_back(norm);
    }

    ParamMap assemble;
    assemble["source"] = destination + ".tiles";
    assemble["destination"] = destination;
    assemble["rows"] = static_cast<int64_t>(rows);
    assemble["cols"] = static_cast<int64_t>(cols);
    return b.add("sram_copy", "access_core", label + " assemble output",
                 assemble, row_outputs);
}

InstructionId append_rmsnorm(Builder& b, const LlamaScheduleConfig& cfg,
                             const std::string& label,
                             const std::string& tag,
                             const std::string& source,
                             const std::string& destination,
                             uint32_t rows, uint32_t cols,
                             const std::vector<InstructionId>& deps) {
    if (cfg.schedule_granularity == "coarse") {
        ParamMap rn;
        rn["source"] = source;
        rn["destination"] = destination;
        rn["rows"] = static_cast<int64_t>(rows);
        rn["cols"] = static_cast<int64_t>(cols);
        return b.add("rmsnorm", "vector_core", label, rn, deps);
    }
    return append_detailed_rmsnorm(b, cfg, label, tag, source, destination, rows, cols, deps);
}

InstructionId append_detailed_rope(Builder& b,
                                   const std::string& label,
                                   const std::string& tag,
                                   const std::string& source,
                                   const std::string& destination,
                                   uint32_t rows, uint32_t cols,
                                   uint32_t row_start,
                                   const std::vector<InstructionId>& deps) {
    ParamMap table;
    table["source"] = "HBM.rope_table.pos" + std::to_string(row_start);
    table["destination"] = "shared_ibuf." + tag + ".rope_table";
    table["rows"] = static_cast<int64_t>(rows);
    table["cols"] = static_cast<int64_t>(cols);
    InstructionId load_table = b.add("dma_load", "dma", label + " sin/cos table load", table, deps);

    InstructionId load_x = on_chip_move(b, source, "shared_ibuf." + tag + ".rope_x",
                                        rows, cols, label + " input tile load", deps);

    ParamMap split;
    split["source"] = "shared_ibuf." + tag + ".rope_x";
    split["destination"] = "vector_scratch." + tag + ".rope_pairs";
    split["rows"] = static_cast<int64_t>(rows);
    split["cols"] = static_cast<int64_t>(cols);
    InstructionId pair = b.add("rope_pair_split", "vector_core",
                               label + " pair split", split, {load_x});

    ParamMap rotate;
    rotate["source"] = "vector_scratch." + tag + ".rope_pairs";
    rotate["source_b"] = "shared_ibuf." + tag + ".rope_table";
    rotate["destination"] = "vector_scratch." + tag + ".rotated";
    rotate["rows"] = static_cast<int64_t>(rows);
    rotate["cols"] = static_cast<int64_t>(cols);
    rotate["row_start"] = static_cast<int64_t>(row_start);
    InstructionId rot = b.add("rope_rotate", "vector_core",
                              label + " rotate/mul-add", rotate, {pair, load_table});

    return append_vector_phase(b, "rope_store", "vector_core", label + " write rotated tile",
                               "vector_scratch." + tag + ".rotated", destination,
                               rows, cols, {rot});
}

InstructionId append_rope(Builder& b, const LlamaScheduleConfig& cfg,
                          const std::string& label,
                          const std::string& tag,
                          const std::string& source,
                          const std::string& destination,
                          uint32_t rows, uint32_t cols, uint32_t row_start,
                          const std::vector<InstructionId>& deps) {
    if (cfg.schedule_granularity == "coarse") {
        ParamMap rope;
        rope["source"] = source;
        rope["destination"] = destination;
        rope["rows"] = static_cast<int64_t>(rows);
        rope["cols"] = static_cast<int64_t>(cols);
        rope["row_start"] = static_cast<int64_t>(row_start);
        return b.add("rope", "vector_core", label, rope, deps);
    }
    return append_detailed_rope(b, label, tag, source, destination, rows, cols, row_start, deps);
}

InstructionId append_swiglu(Builder& b, const LlamaScheduleConfig& cfg,
                            const std::string& label,
                            const std::string& tag,
                            const std::string& gate,
                            const std::string& up,
                            const std::string& destination,
                            uint32_t rows, uint32_t cols,
                            const std::vector<InstructionId>& deps) {
    if (cfg.schedule_granularity == "coarse") {
        ParamMap act;
        act["source_a"] = gate;
        act["source_b"] = up;
        act["destination"] = destination;
        act["rows"] = static_cast<int64_t>(rows);
        act["cols"] = static_cast<int64_t>(cols);
        return b.add("silu_mul", "vector_core", label, act, deps);
    }
    ParamMap silu;
    silu["source"] = gate;
    silu["destination"] = "vector_scratch." + tag + ".silu_gate";
    silu["rows"] = static_cast<int64_t>(rows);
    silu["cols"] = static_cast<int64_t>(cols);
    InstructionId silu_id = b.add("silu", "vector_core", label + " SiLU gate", silu, deps);

    ParamMap mul;
    mul["source_a"] = "vector_scratch." + tag + ".silu_gate";
    mul["source_b"] = up;
    mul["destination"] = destination;
    mul["rows"] = static_cast<int64_t>(rows);
    mul["cols"] = static_cast<int64_t>(cols);
    return b.add("elementwise_mul", "vector_core", label + " multiply up", mul, {silu_id});
}

InstructionId append_logits_softmax(Builder& b, const LlamaScheduleConfig& cfg,
                                    const std::string& tag,
                                    const std::string& logits,
                                    const std::string& probs,
                                    uint32_t vocab_size,
                                    const std::vector<InstructionId>& deps) {
    if (cfg.schedule_granularity == "coarse") {
        ParamMap sm;
        sm["source"] = logits;
        sm["destination"] = probs;
        sm["rows"] = static_cast<int64_t>(1);
        sm["cols"] = static_cast<int64_t>(vocab_size);
        return b.add("softmax", "vector_core", "logits softmax", sm, deps);
    }

    ParamMap rowmax;
    rowmax["source"] = logits;
    rowmax["destination"] = "vector_scratch." + tag + ".logits_max";
    rowmax["rows"] = static_cast<int64_t>(1);
    rowmax["cols"] = static_cast<int64_t>(vocab_size);
    InstructionId max_id = b.add("rowmax", "vector_core", "logits softmax rowmax", rowmax, deps);

    ParamMap exp;
    exp["source_matrix"] = logits;
    exp["source_shift"] = "vector_scratch." + tag + ".logits_max";
    exp["destination"] = "shared_ibuf." + tag + ".logits_exp";
    exp["rows"] = static_cast<int64_t>(1);
    exp["cols"] = static_cast<int64_t>(vocab_size);
    InstructionId exp_id = b.add("exp_shift", "vector_core", "logits softmax exp", exp, {max_id});

    ParamMap sum;
    sum["source"] = "shared_ibuf." + tag + ".logits_exp";
    sum["destination"] = "vector_scratch." + tag + ".logits_sum";
    sum["rows"] = static_cast<int64_t>(1);
    sum["cols"] = static_cast<int64_t>(vocab_size);
    InstructionId sum_id = b.add("row_reduce_sum", "vector_core", "logits softmax rowsum", sum, {exp_id});

    ParamMap norm;
    norm["source_matrix"] = "shared_ibuf." + tag + ".logits_exp";
    norm["source_denom"] = "vector_scratch." + tag + ".logits_sum";
    norm["destination"] = probs;
    norm["rows"] = static_cast<int64_t>(1);
    norm["cols"] = static_cast<int64_t>(vocab_size);
    return b.add("normalize", "vector_core", "logits softmax normalize", norm, {sum_id});
}

InstructionId append_sample(Builder& b, const LlamaScheduleConfig& cfg,
                            const std::string& source,
                            const std::string& destination,
                            uint32_t vocab_size,
                            const std::vector<InstructionId>& deps) {
    ParamMap sample;
    sample["source"] = source;
    sample["destination"] = destination;
    sample["rows"] = static_cast<int64_t>(1);
    sample["cols"] = static_cast<int64_t>(vocab_size);
    if (cfg.schedule_granularity == "coarse") {
        return b.add("sample_token", "vector_core", "sample next token from logits",
                     sample, deps);
    }
    return b.add("sample_top1", "vector_core", "sample next token from logits",
                 sample, deps);
}

struct AttentionIds {
    InstructionId output = 0;
    std::vector<InstructionId> terminal_heads;
};

struct KvTileLoad {
    uint32_t start = 0;
    uint32_t rows = 0;
    uint32_t slot = 0;
    std::string tag;
    std::string k;
    std::string k_t;
    std::string v;
    InstructionId read_k = 0;
    InstructionId read_v = 0;
    InstructionId transpose_k = 0;
};

KvTileLoad append_kv_tile_load(Builder& b, const LlamaScheduleConfig& cfg,
                               const std::string& layer, const std::string& step,
                               uint32_t kvh, uint32_t kt, uint32_t kv_len,
                               const std::vector<InstructionId>& deps) {
    KvTileLoad load;
    load.start = kt * cfg.tile_cols;
    load.rows = tile_extent(kv_len, cfg.tile_cols, kt);
    load.slot = cfg.kv_prefetch == "double_buffer"
        ? kt % cfg.kv_stage_buffers
        : 0;
    load.tag = layer + "." + step + ".kv" + std::to_string(kvh)
             + ".tile" + std::to_string(kt);

    const std::string slot_tag = "shared_ibuf.kv_stage.kv" + std::to_string(kvh)
                               + ".slot" + std::to_string(load.slot);
    load.k = slot_tag + ".K";
    load.k_t = slot_tag + ".K_T";
    load.v = slot_tag + ".V";

    if (cfg.kv_cache_enabled) {
        const std::string k_cache = cache_range_name(cfg, layer, kvh, "K",
                                                     load.start, load.rows, kv_len);
        const std::string v_cache = cache_range_name(cfg, layer, kvh, "V",
                                                     load.start, load.rows, kv_len);
        load.read_k = cache_move(b, cfg, true, load.k, k_cache, load.rows, deps);
        load.read_v = cache_move(b, cfg, true, load.v, v_cache, load.rows, deps);
    } else {
        load.read_k = on_chip_move(b, "shared_obuf." + layer + "." + step + ".K_rope",
                                   load.k, load.rows, cfg.head_dim,
                                   "on-chip K tile read " + load.tag, deps);
        load.read_v = on_chip_move(b, "shared_obuf." + layer + "." + step + ".V",
                                   load.v, load.rows, cfg.head_dim,
                                   "on-chip V tile read " + load.tag, deps);
    }

    ParamMap tr;
    tr["source"] = load.k;
    tr["destination"] = load.k_t;
    tr["input_rows"] = static_cast<int64_t>(load.rows);
    tr["input_cols"] = static_cast<int64_t>(cfg.head_dim);
    tr["stage_slot"] = static_cast<int64_t>(load.slot);
    load.transpose_k = b.add("transpose", "access_core",
                             "transpose " + load.tag + " K in slot"
                             + std::to_string(load.slot),
                             tr, {load.read_k});
    return load;
}

AttentionIds append_attention(Builder& b, const LlamaScheduleConfig& cfg,
                              uint32_t layer, uint32_t q_len, uint32_t kv_len,
                              uint32_t decode_step, const std::string& input,
                              const std::vector<InstructionId>& deps) {
    const uint32_t group = cfg.num_q_heads / cfg.num_kv_heads;
    const std::string l = "L" + std::to_string(layer);
    const std::string step = "S" + std::to_string(decode_step);

    InstructionId q_proj = append_staged_gemm(
        b, cfg, l + " Q projection", l + "." + step + ".Wq", input, "HBM." + l + ".Wq",
        "shared_obuf." + l + "." + step + ".Q",
        q_len, cfg.hidden_dim, cfg.num_q_heads * cfg.head_dim, deps);

    InstructionId k_proj = append_staged_gemm(
        b, cfg, l + " K projection", l + "." + step + ".Wk", input, "HBM." + l + ".Wk",
        "shared_obuf." + l + "." + step + ".K",
        q_len, cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim, deps);

    InstructionId v_proj = append_staged_gemm(
        b, cfg, l + " V projection", l + "." + step + ".Wv", input, "HBM." + l + ".Wv",
        "shared_obuf." + l + "." + step + ".V",
        q_len, cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim, deps);

    InstructionId q_rope = append_rope(
        b, cfg, l + " RoPE Q", l + "." + step + ".Q",
        "shared_obuf." + l + "." + step + ".Q",
        "shared_obuf." + l + "." + step + ".Q_rope",
        q_len, cfg.head_dim, decode_step, {q_proj});

    InstructionId k_rope = append_rope(
        b, cfg, l + " RoPE K", l + "." + step + ".K",
        "shared_obuf." + l + "." + step + ".K",
        "shared_obuf." + l + "." + step + ".K_rope",
        q_len, cfg.head_dim, decode_step, {k_proj});

    const uint32_t current_start = (q_len == 1) ? decode_step : 0;
    std::vector<InstructionId> cache_write_deps = {k_rope, v_proj};
    if (cfg.kv_cache_enabled) {
        for (uint32_t kvh = 0; kvh < cfg.num_kv_heads; kvh++) {
            for (uint32_t wt = 0; wt < ceil_div(q_len, cfg.tile_cols); wt++) {
                const uint32_t rows = tile_extent(q_len, cfg.tile_cols, wt);
                const uint32_t start = current_start + wt * cfg.tile_cols;
                const std::string k_src = "shared_obuf." + l + "." + step
                                        + ".K_rope.kv" + std::to_string(kvh)
                                        + range_suffix(start, rows);
                const std::string v_src = "shared_obuf." + l + "." + step
                                        + ".V.kv" + std::to_string(kvh)
                                        + range_suffix(start, rows);
                cache_write_deps.push_back(cache_move(
                    b, cfg, false, k_src,
                    cache_range_name(cfg, l, kvh, "K", start, rows, kv_len), rows, {k_rope}));
                cache_write_deps.push_back(cache_move(
                    b, cfg, false, v_src,
                    cache_range_name(cfg, l, kvh, "V", start, rows, kv_len), rows, {v_proj}));
            }
        }
    }

    std::vector<InstructionId> head_outputs;
    const uint32_t q_tiles = ceil_div(q_len, cfg.tile_rows);
    const uint32_t kv_tiles = ceil_div(kv_len, cfg.tile_cols);

    for (uint32_t kvh = 0; kvh < cfg.num_kv_heads; kvh++) {
        std::vector<InstructionId> slot_free(cfg.kv_stage_buffers);
        for (uint32_t kt = 0; kt < kv_tiles; kt++) {
            std::vector<InstructionId> read_deps = cache_write_deps;
            const uint32_t slot = cfg.kv_prefetch == "double_buffer"
                ? kt % cfg.kv_stage_buffers
                : 0;
            if (slot_free[slot]) read_deps.push_back(slot_free[slot]);
            KvTileLoad kv_load = append_kv_tile_load(
                b, cfg, l, step, kvh, kt, kv_len, read_deps);
            std::vector<InstructionId> slot_consumers;

            for (uint32_t local = 0; local < group; local++) {
                const uint32_t qh = kvh * group + local;
                for (uint32_t qt = 0; qt < q_tiles; qt++) {
                    const uint32_t q_rows = tile_extent(q_len, cfg.tile_rows, qt);
                    const uint32_t q_start = (cfg.mode == "decode") ? decode_step : qt * cfg.tile_rows;
                    const std::string tag = kv_load.tag + ".qh" + std::to_string(qh)
                                          + ".qt" + std::to_string(qt);

                    ParamMap init_o;
                    init_o["destination"] = "shared_obuf." + tag + ".O_acc";
                    init_o["rows"] = static_cast<int64_t>(q_rows);
                    init_o["cols"] = static_cast<int64_t>(cfg.head_dim);
                    init_o["init_value"] = static_cast<int64_t>(0);
                    InstructionId init_o_id = b.add("init_fill", "access_core", "init O " + tag,
                                                    init_o, {q_rope});

                    ParamMap init_m;
                    init_m["destination"] = "shared_obuf." + tag + ".m";
                    init_m["length"] = static_cast<int64_t>(q_rows);
                    init_m["init_value"] = std::string("-inf");
                    InstructionId init_m_id = b.add("init_fill", "access_core", "init m " + tag,
                                                    init_m, {init_o_id});

                    ParamMap init_l;
                    init_l["destination"] = "shared_obuf." + tag + ".l";
                    init_l["length"] = static_cast<int64_t>(q_rows);
                    init_l["init_value"] = static_cast<int64_t>(0);
                    InstructionId init_l_id = b.add("init_fill", "access_core", "init l " + tag,
                                                    init_l, {init_m_id});

                    ParamMap stq;
                    stq["source"] = "shared_obuf." + l + "." + step + ".Q_rope";
                    stq["destination"] = "systolic_array.Q_operand";
                    stq["rows"] = static_cast<int64_t>(q_rows);
                    stq["cols"] = static_cast<int64_t>(cfg.head_dim);
                    InstructionId stage_q = b.add("dma_stage", "dma", "stage Q " + tag,
                                                  stq, {init_l_id});

                    ParamMap qk;
                    qk["source_a"] = "systolic_array.Q_operand";
                    qk["source_b"] = kv_load.k_t;
                    qk["destination"] = "shared_obuf." + tag + ".S";
                    qk["M"] = static_cast<int64_t>(q_rows);
                    qk["K"] = static_cast<int64_t>(cfg.head_dim);
                    qk["N"] = static_cast<int64_t>(kv_load.rows);
                    InstructionId mat_qk = b.add("gemm", "systolic", "QK " + tag,
                                                 qk, {stage_q, kv_load.transpose_k});

                    ParamMap scale;
                    scale["source"] = "shared_obuf." + tag + ".S";
                    scale["destination"] = "shared_obuf." + tag + ".S";
                    scale["rows"] = static_cast<int64_t>(q_rows);
                    scale["cols"] = static_cast<int64_t>(kv_load.rows);
                    InstructionId scale_id = b.add("scale", "vector_core", "scale " + tag,
                                                   scale, {mat_qk});

                    ParamMap mask = scale;
                    mask["row_start"] = static_cast<int64_t>(q_start);
                    mask["col_start"] = static_cast<int64_t>(kv_load.start);
                    InstructionId mask_id = b.add("causal_mask", "vector_core",
                                                  "causal mask " + tag, mask, {scale_id});

                    ParamMap rowmax;
                    rowmax["source"] = "shared_obuf." + tag + ".S";
                    rowmax["destination"] = "vector_scratch." + tag + ".rowmax";
                    rowmax["rows"] = static_cast<int64_t>(q_rows);
                    rowmax["cols"] = static_cast<int64_t>(kv_load.rows);
                    InstructionId rowmax_id = b.add("rowmax", "vector_core", "rowmax " + tag,
                                                    rowmax, {mask_id});

                    ParamMap upd_m;
                    upd_m["source_m_old"] = "shared_obuf." + tag + ".m";
                    upd_m["source_rowmax"] = "vector_scratch." + tag + ".rowmax";
                    upd_m["destination_m"] = "shared_obuf." + tag + ".m";
                    upd_m["destination_correction"] = "shared_obuf." + tag + ".correction";
                    upd_m["length"] = static_cast<int64_t>(q_rows);
                    InstructionId upd_m_id = b.add("update_rowmax", "vector_core",
                                                   "update m " + tag, upd_m, {rowmax_id, init_m_id});

                    ParamMap exp;
                    exp["source_matrix"] = "shared_obuf." + tag + ".S";
                    exp["source_shift"] = "shared_obuf." + tag + ".m";
                    exp["destination"] = "shared_ibuf." + tag + ".P";
                    exp["rows"] = static_cast<int64_t>(q_rows);
                    exp["cols"] = static_cast<int64_t>(kv_load.rows);
                    InstructionId exp_id = b.add("exp_shift", "vector_core", "softmax exp " + tag,
                                                 exp, {upd_m_id});

                    ParamMap rowsum;
                    rowsum["source_p"] = "shared_ibuf." + tag + ".P";
                    rowsum["source_correction"] = "shared_obuf." + tag + ".correction";
                    rowsum["source_l_old"] = "shared_obuf." + tag + ".l";
                    rowsum["destination"] = "shared_obuf." + tag + ".l";
                    rowsum["rows"] = static_cast<int64_t>(q_rows);
                    rowsum["cols"] = static_cast<int64_t>(kv_load.rows);
                    InstructionId rowsum_id = b.add("update_rowsum", "vector_core",
                                                    "update l " + tag, rowsum, {exp_id, upd_m_id, init_l_id});

                    ParamMap rescale;
                    rescale["source"] = "shared_obuf." + tag + ".O_acc";
                    rescale["source_scale"] = "shared_obuf." + tag + ".correction";
                    rescale["destination"] = "shared_obuf." + tag + ".O_acc";
                    rescale["rows"] = static_cast<int64_t>(q_rows);
                    rescale["cols"] = static_cast<int64_t>(cfg.head_dim);
                    InstructionId rescale_id = b.add("scale", "vector_core", "rescale O " + tag,
                                                     rescale, {upd_m_id, init_o_id});

                    ParamMap stp;
                    stp["source"] = "shared_ibuf." + tag + ".P";
                    stp["destination"] = "systolic_array.P_operand";
                    stp["rows"] = static_cast<int64_t>(q_rows);
                    stp["cols"] = static_cast<int64_t>(kv_load.rows);
                    InstructionId stage_p = b.add("dma_stage", "dma", "stage P " + tag,
                                                  stp, {exp_id, kv_load.read_v});

                    ParamMap pv;
                    pv["source_a"] = "systolic_array.P_operand";
                    pv["source_b"] = kv_load.v;
                    pv["destination"] = "shared_obuf." + tag + ".Temp";
                    pv["M"] = static_cast<int64_t>(q_rows);
                    pv["K"] = static_cast<int64_t>(kv_load.rows);
                    pv["N"] = static_cast<int64_t>(cfg.head_dim);
                    InstructionId mat_pv = b.add("gemm", "systolic", "PV " + tag,
                                                 pv, {stage_p, kv_load.read_v});

                    ParamMap acc;
                    acc["source_a"] = "shared_obuf." + tag + ".O_acc";
                    acc["source_b"] = "shared_obuf." + tag + ".Temp";
                    acc["destination"] = "shared_obuf." + tag + ".O_acc";
                    acc["rows"] = static_cast<int64_t>(q_rows);
                    acc["cols"] = static_cast<int64_t>(cfg.head_dim);
                    InstructionId acc_id = b.add("accumulate", "vector_core", "accumulate O " + tag,
                                                 acc, {rescale_id, mat_pv});

                    ParamMap norm;
                    norm["source_matrix"] = "shared_obuf." + tag + ".O_acc";
                    norm["source_denom"] = "shared_obuf." + tag + ".l";
                    norm["destination"] = "shared_obuf." + tag + ".O";
                    norm["rows"] = static_cast<int64_t>(q_rows);
                    norm["cols"] = static_cast<int64_t>(cfg.head_dim);
                    InstructionId out = b.add("normalize", "vector_core", "normalize " + tag,
                                              norm, {acc_id, rowsum_id});
                    head_outputs.push_back(out);
                    slot_consumers.push_back(out);
                }
            }
            slot_free[kv_load.slot] = b.add("kv_stage_release", "access_core",
                                            "release KV stage slot"
                                            + std::to_string(kv_load.slot)
                                            + " after " + kv_load.tag,
                                            {{"slot", static_cast<int64_t>(kv_load.slot)},
                                             {"rows", static_cast<int64_t>(kv_load.rows)},
                                             {"cols", static_cast<int64_t>(cfg.head_dim)}},
                                            slot_consumers);
        }
    }

    ParamMap merge;
    merge["source"] = "shared_obuf." + l + "." + step + ".attention_tile_outputs";
    merge["destination"] = "shared_obuf." + l + "." + step + ".attention_heads";
    merge["rows"] = static_cast<int64_t>(q_len);
    merge["cols"] = static_cast<int64_t>(cfg.num_q_heads * cfg.head_dim);
    merge["q_tiles"] = static_cast<int64_t>(q_tiles);
    merge["kv_tiles"] = static_cast<int64_t>(kv_tiles);
    merge["num_q_heads"] = static_cast<int64_t>(cfg.num_q_heads);
    merge["head_dim"] = static_cast<int64_t>(cfg.head_dim);
    merge["input_elements"] = static_cast<int64_t>(
        static_cast<uint64_t>(q_len) * cfg.num_q_heads * cfg.head_dim * kv_tiles);
    merge["output_elements"] = static_cast<int64_t>(
        static_cast<uint64_t>(q_len) * cfg.num_q_heads * cfg.head_dim);
    InstructionId merge_id = b.add("attention_merge", "vector_core",
                                   l + " merge attention tiles", merge, head_outputs);

    InstructionId out = append_staged_gemm(
        b, cfg, l + " output projection", l + "." + step + ".Wo",
        "shared_obuf." + l + "." + step + ".attention_heads", "HBM." + l + ".Wo",
        "shared_obuf." + l + "." + step + ".attention_out",
        q_len, cfg.num_q_heads * cfg.head_dim, cfg.hidden_dim, {merge_id});
    return AttentionIds{out, head_outputs};
}

InstructionId append_embedding(Builder& b, const LlamaScheduleConfig& cfg,
                               const std::string& source_tokens,
                               const std::string& destination,
                               uint32_t token_count,
                               const std::vector<InstructionId>& deps) {
    ParamMap emb;
    emb["source"] = source_tokens + " via HBM.token_embeddings";
    emb["destination"] = destination;
    emb["rows"] = static_cast<int64_t>(token_count);
    emb["cols"] = static_cast<int64_t>(cfg.hidden_dim);
    return b.add("embedding_lookup", "dma", "token embedding lookup",
                 emb, deps);
}

InstructionId append_output_head(Builder& b, const LlamaScheduleConfig& cfg,
                                 const std::string& hidden,
                                 uint32_t step,
                                 uint32_t token_count,
                                 const std::vector<InstructionId>& deps) {
    const std::string tag = "S" + std::to_string(step);
    std::vector<InstructionId> cur = deps;
    std::string head_input = hidden;
    if (token_count > 1) {
        ParamMap last;
        last["source"] = hidden;
        last["destination"] = "shared_obuf." + tag + ".last_hidden";
        last["rows"] = static_cast<int64_t>(1);
        last["cols"] = static_cast<int64_t>(cfg.hidden_dim);
        last["row_index"] = static_cast<int64_t>(token_count - 1);
        cur = {b.add(cfg.schedule_granularity == "coarse" ? "select_last_token" : "gather_select",
                     "access_core", "select final position hidden state", last, cur)};
        head_input = "shared_obuf." + tag + ".last_hidden";
    }

    InstructionId final_norm = append_rmsnorm(
        b, cfg, "final RMSNorm before logits", tag + ".final_norm",
        head_input, "shared_obuf." + tag + ".final_norm",
        1, cfg.hidden_dim, cur);

    InstructionId logits = append_staged_gemm(
        b, cfg, "LM head linear logits projection", tag + ".lm_head",
        "shared_obuf." + tag + ".final_norm", "HBM.lm_head",
        "shared_obuf." + tag + ".logits",
        1, cfg.hidden_dim, cfg.vocab_size, {final_norm});

    InstructionId softmax = append_logits_softmax(
        b, cfg, tag, "shared_obuf." + tag + ".logits",
        "shared_obuf." + tag + ".probs", cfg.vocab_size, {logits});

    return append_sample(b, cfg, "shared_obuf." + tag + ".probs",
                         "shared_obuf." + tag + ".sampled_token",
                         cfg.vocab_size, {softmax});
}

InstructionId append_token_feedback(Builder& b, uint32_t position,
                                    const std::string& sampled_token,
                                    const std::vector<InstructionId>& deps) {
    ParamMap feedback;
    feedback["source"] = sampled_token;
    feedback["destination"] = "HBM.decode_token.pos" + std::to_string(position);
    feedback["rows"] = static_cast<int64_t>(1);
    feedback["cols"] = static_cast<int64_t>(1);
    return b.add("token_feedback", "vector_core",
                 "feed sampled token back to decode input", feedback, deps);
}

InstructionId append_layer(Builder& b, const LlamaScheduleConfig& cfg,
                           uint32_t layer, uint32_t q_len, uint32_t kv_len,
                           uint32_t decode_step, const std::string& input,
                           const std::vector<InstructionId>& deps) {
    const std::string l = "L" + std::to_string(layer);
    InstructionId norm1 = append_rmsnorm(
        b, cfg, l + " attention RMSNorm", l + ".attn_norm",
        input, "shared_obuf." + l + ".attn_norm",
        q_len, cfg.hidden_dim, deps);

    AttentionIds attn = append_attention(b, cfg, layer, q_len, kv_len, decode_step,
                                         "shared_obuf." + l + ".attn_norm", {norm1});

    ParamMap add1;
    add1["source_a"] = input;
    add1["source_b"] = "shared_obuf." + l + ".S" + std::to_string(decode_step) + ".attention_out";
    add1["destination"] = "shared_obuf." + l + ".attn_residual";
    add1["rows"] = static_cast<int64_t>(q_len);
    add1["cols"] = static_cast<int64_t>(cfg.hidden_dim);
    InstructionId attn_res = b.add("residual_add", "vector_core", l + " attention residual",
                                   add1, {attn.output});

    InstructionId norm2 = append_rmsnorm(
        b, cfg, l + " MLP RMSNorm", l + ".mlp_norm",
        "shared_obuf." + l + ".attn_residual",
        "shared_obuf." + l + ".mlp_norm",
        q_len, cfg.hidden_dim, {attn_res});

    InstructionId gate_id = append_staged_gemm(
        b, cfg, l + " MLP gate", l + ".W_gate",
        "shared_obuf." + l + ".mlp_norm", "HBM." + l + ".W_gate",
        "shared_obuf." + l + ".gate",
        q_len, cfg.hidden_dim, cfg.intermediate_dim, {norm2});

    InstructionId up_id = append_staged_gemm(
        b, cfg, l + " MLP up", l + ".W_up",
        "shared_obuf." + l + ".mlp_norm", "HBM." + l + ".W_up",
        "shared_obuf." + l + ".up",
        q_len, cfg.hidden_dim, cfg.intermediate_dim, {norm2});

    InstructionId act_id = append_swiglu(
        b, cfg, l + " SwiGLU", l + ".swiglu",
        "shared_obuf." + l + ".gate", "shared_obuf." + l + ".up",
        "shared_obuf." + l + ".ff",
        q_len, cfg.intermediate_dim, {gate_id, up_id});

    InstructionId down_id = append_staged_gemm(
        b, cfg, l + " MLP down", l + ".W_down",
        "shared_obuf." + l + ".ff", "HBM." + l + ".W_down",
        "shared_obuf." + l + ".mlp_out",
        q_len, cfg.intermediate_dim, cfg.hidden_dim, {act_id});

    ParamMap add2;
    add2["source_a"] = "shared_obuf." + l + ".attn_residual";
    add2["source_b"] = "shared_obuf." + l + ".mlp_out";
    add2["destination"] = "shared_obuf." + l + ".layer_out";
    add2["rows"] = static_cast<int64_t>(q_len);
    add2["cols"] = static_cast<int64_t>(cfg.hidden_dim);
    return b.add("residual_add", "vector_core", l + " MLP residual", add2, {down_id});
}

InstructionId append_layer_stack(Builder& b, const LlamaScheduleConfig& cfg,
                                 uint32_t q_len, uint32_t kv_len,
                                 uint32_t decode_step,
                                 const std::string& input,
                                 const std::vector<InstructionId>& deps) {
    std::string cur_input = input;
    std::vector<InstructionId> cur_deps = deps;
    InstructionId prev = 0;
    for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
        prev = append_layer(b, cfg, layer, q_len, kv_len, decode_step,
                            cur_input, cur_deps);
        cur_input = "shared_obuf.L" + std::to_string(layer) + ".layer_out";
        cur_deps = {prev};
    }
    return prev;
}

Schedule finish(Builder& b) {
    Schedule s;
    s.instructions = std::move(b.out);
    s.validate();
    return s;
}

}  // namespace

Schedule build_attention_schedule(const LlamaScheduleConfig& input_cfg) {
    const LlamaScheduleConfig cfg = normalize_cfg(input_cfg);
    Builder b;
    const uint32_t q_len = cfg.mode == "decode" ? 1
                         : (cfg.mode == "prefill" ? cfg.prompt_len : cfg.seq_len);
    const uint32_t kv_len = cfg.mode == "decode" ? cfg.prompt_len + 1 : q_len;
    append_attention(b, cfg, 0, q_len, kv_len, cfg.prompt_len, "HBM.input", {});
    return finish(b);
}

Schedule build_transformer_layer_schedule(const LlamaScheduleConfig& input_cfg) {
    const LlamaScheduleConfig cfg = normalize_cfg(input_cfg);
    Builder b;
    const uint32_t q_len = cfg.mode == "decode" ? 1
                         : (cfg.mode == "prefill" ? cfg.prompt_len : cfg.seq_len);
    const uint32_t kv_len = cfg.mode == "decode" ? cfg.prompt_len + 1 : q_len;
    const uint32_t step = cfg.mode == "decode" ? cfg.prompt_len : 0;
    InstructionId emb = append_embedding(b, cfg, "HBM.input_tokens",
                                         "shared_obuf.S" + std::to_string(step) + ".embeddings",
                                         q_len, {});
    InstructionId layers = append_layer_stack(
        b, cfg, q_len, kv_len, step,
        "shared_obuf.S" + std::to_string(step) + ".embeddings", {emb});
    append_output_head(b, cfg,
                       "shared_obuf.L" + std::to_string(cfg.num_layers - 1) + ".layer_out",
                       step, q_len, {layers});
    return finish(b);
}

Schedule build_prefill_decode_schedule(const LlamaScheduleConfig& input_cfg) {
    const LlamaScheduleConfig cfg = normalize_cfg(input_cfg);
    Builder b;
    LlamaScheduleConfig prefill = cfg;
    prefill.mode = "prefill";
    prefill.kv_cache_enabled = true;
    InstructionId prompt_emb = append_embedding(b, prefill, "HBM.prompt_tokens",
                                                "shared_obuf.S0.embeddings",
                                                cfg.prompt_len, {});
    InstructionId layers = append_layer_stack(b, prefill, cfg.prompt_len, cfg.prompt_len, 0,
                                              "shared_obuf.S0.embeddings", {prompt_emb});
    InstructionId prev_sample = append_output_head(
        b, cfg,
        "shared_obuf.L" + std::to_string(cfg.num_layers - 1) + ".layer_out",
        cfg.prompt_len - 1, cfg.prompt_len, {layers});

    LlamaScheduleConfig decode = cfg;
    decode.mode = "decode";
    decode.kv_cache_enabled = true;
    for (uint32_t step = 0; step < cfg.generation_steps; step++) {
        const uint32_t pos = cfg.prompt_len + step;
        InstructionId feedback = append_token_feedback(
            b, pos,
            "shared_obuf.S" + std::to_string(pos - 1) + ".sampled_token",
            {prev_sample});
        InstructionId decode_emb = append_embedding(
            b, decode, "HBM.decode_token.pos" + std::to_string(pos),
            "shared_obuf.S" + std::to_string(pos) + ".embeddings",
            1, {feedback});
        InstructionId decode_layers = append_layer_stack(
            b, decode, 1, cfg.prompt_len + step + 1, pos,
            "shared_obuf.S" + std::to_string(pos) + ".embeddings", {decode_emb});
        prev_sample = append_output_head(
            b, cfg,
            "shared_obuf.L" + std::to_string(cfg.num_layers - 1) + ".layer_out",
            pos, 1, {decode_layers});
    }
    return finish(b);
}

Schedule build_llama_schedule(const LlamaScheduleConfig& cfg) {
    if (cfg.mode == "attention") return build_attention_schedule(cfg);
    if (cfg.mode == "layer" || cfg.mode == "prefill" || cfg.mode == "decode")
        return build_transformer_layer_schedule(cfg);
    if (cfg.mode == "prefill_decode") return build_prefill_decode_schedule(cfg);
    throw std::runtime_error("LlamaScheduleConfig: unsupported mode '" + cfg.mode + "'");
}

namespace {

template <typename T>
void read_scalar(const YAML::Node& n, const char* key, T& dst) {
    if (n[key]) dst = n[key].as<T>();
}

LlamaScheduleConfig llama_config_from_node(const YAML::Node& root) {
    YAML::Node n = root["llama"] ? root["llama"] : root;
    LlamaScheduleConfig cfg;
    read_scalar(n, "mode", cfg.mode);
    read_scalar(n, "schedule_granularity", cfg.schedule_granularity);
    read_scalar(n, "seq_len", cfg.seq_len);
    read_scalar(n, "prompt_len", cfg.prompt_len);
    read_scalar(n, "generation_steps", cfg.generation_steps);
    read_scalar(n, "num_layers", cfg.num_layers);
    read_scalar(n, "num_q_heads", cfg.num_q_heads);
    read_scalar(n, "num_kv_heads", cfg.num_kv_heads);
    read_scalar(n, "gqa_group_size", cfg.gqa_group_size);
    read_scalar(n, "head_dim", cfg.head_dim);
    read_scalar(n, "hidden_dim", cfg.hidden_dim);
    read_scalar(n, "intermediate_dim", cfg.intermediate_dim);
    read_scalar(n, "vocab_size", cfg.vocab_size);
    read_scalar(n, "tile_rows", cfg.tile_rows);
    read_scalar(n, "tile_cols", cfg.tile_cols);
    read_scalar(n, "linear_tile_rows", cfg.linear_tile_rows);
    read_scalar(n, "linear_tile_cols", cfg.linear_tile_cols);
    read_scalar(n, "max_seq_len", cfg.max_seq_len);
    read_scalar(n, "dtype_bytes", cfg.dtype_bytes);
    read_scalar(n, "sram_kv_capacity_kb", cfg.sram_kv_capacity_kb);
    read_scalar(n, "kv_cache_block_tokens", cfg.kv_cache_block_tokens);
    read_scalar(n, "kv_stage_buffers", cfg.kv_stage_buffers);
    read_scalar(n, "kv_cache_enabled", cfg.kv_cache_enabled);
    read_scalar(n, "kv_cache_location", cfg.kv_cache_location);
    read_scalar(n, "kv_prefetch", cfg.kv_prefetch);
    read_scalar(n, "kv_cache_eviction_policy", cfg.kv_cache_eviction_policy);
    return normalize_cfg(cfg);
}

}  // namespace

LlamaScheduleConfig llama_config_from_yaml_file(const std::string& path) {
    return llama_config_from_node(YAML::LoadFile(path));
}

LlamaScheduleConfig llama_config_from_yaml_string(const std::string& yaml) {
    return llama_config_from_node(YAML::Load(yaml));
}

}  // namespace sim
