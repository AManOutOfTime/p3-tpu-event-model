#include "schedule/llama_schedule.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>
#include <variant>

namespace sim {
namespace {

struct Builder {
    std::vector<Instruction> out;
    InstructionId next = 0;
    // When set, trace-only metadata is dropped to slash host RAM on huge
    // schedules: the human-readable label and the symbolic buffer-name string
    // params (source/destination/source_a/...). The timing model reads only
    // numeric params (M/K/N/rows/cols/length/...) — the LLaMA builder always
    // stores those as int64, so dropping string params is timing-neutral. Only
    // enabled when tracing is off (the strings exist purely for the trace log
    // and the currently-dead numerical data path).
    bool minimal = false;

    InstructionId add(std::string op, std::string unit, std::string label,
                      ParamMap params = {}, std::vector<InstructionId> deps = {}) {
        Instruction inst;
        inst.id = next++;
        inst.op = std::move(op);
        inst.unit = std::move(unit);
        if (minimal) {
            // Keep numeric params plus init_value (a fill semantic that can be
            // stored as the string "-inf"); drop buffer-name strings + label.
            for (auto& kv : params)
                if (!kv.second.is_string() || kv.first == "init_value")
                    inst.params[kv.first] = std::move(kv.second);
            // Each operator[] call above can double the vector capacity,
            // leaving up to 50% unused. Shrink to exactly the number of
            // params kept so the wasted capacity doesn't persist for the
            // lifetime of the simulation.
            inst.params.shrink();
        } else {
            inst.label = std::move(label);
            inst.params = std::move(params);
            // In non-minimal mode params arrive via initializer_list (exact
            // capacity), but shrink defensively in case any caller grew via
            // operator[].
            inst.params.shrink();
        }
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
    if (cfg.schedule_granularity != "detailed")
        throw std::runtime_error("LlamaScheduleConfig: only detailed schedule generation is supported");
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

InstructionId append_rope(Builder& b,
                          const std::string& label,
                          const std::string& tag,
                          const std::string& source,
                          const std::string& destination,
                          uint32_t rows, uint32_t cols, uint32_t row_start,
                          const std::vector<InstructionId>& deps) {
    return append_detailed_rope(b, label, tag, source, destination, rows, cols, row_start, deps);
}

InstructionId append_detailed_mlp_kernel(Builder& b, const LlamaScheduleConfig& cfg,
                                         const std::string& layer,
                                         const std::string& source,
                                         const std::string& destination,
                                         uint32_t rows,
                                         const std::vector<InstructionId>& deps) {
    const uint32_t row_tiles = ceil_div(rows, cfg.linear_tile_rows);
    const uint32_t inter_tiles = ceil_div(cfg.intermediate_dim, cfg.linear_tile_cols);
    const uint32_t out_tiles = ceil_div(cfg.hidden_dim, cfg.linear_tile_cols);
    std::vector<InstructionId> output_tiles;

    for (uint32_t rt = 0; rt < row_tiles; rt++) {
        const uint32_t row_start = rt * cfg.linear_tile_rows;
        const uint32_t row_count = tile_extent(rows, cfg.linear_tile_rows, rt);
        const std::string row_tag = layer + ".mlp.r" + std::to_string(rt);
        const std::string a_tile = "shared_ibuf." + row_tag + ".A";

        InstructionId a_load = on_chip_move(
            b, source, a_tile, row_count, cfg.hidden_dim,
            layer + " MLP activation tile r" + std::to_string(rt) + " load", deps);

        std::vector<InstructionId> acc_state(out_tiles);
        for (uint32_t ot = 0; ot < out_tiles; ot++) {
            const uint32_t out_cols = tile_extent(cfg.hidden_dim, cfg.linear_tile_cols, ot);
            const std::string out_tag = row_tag + ".out.c" + std::to_string(ot);

            ParamMap init;
            init["destination"] = "shared_obuf." + out_tag + ".O_acc";
            init["rows"] = static_cast<int64_t>(row_count);
            init["cols"] = static_cast<int64_t>(out_cols);
            init["init_value"] = static_cast<int64_t>(0);
            acc_state[ot] = b.add("init_fill", "access_core",
                                  layer + " MLP down init O tile r" + std::to_string(rt)
                                  + " c" + std::to_string(ot),
                                  init, {a_load});
        }

        for (uint32_t it = 0; it < inter_tiles; it++) {
            const uint32_t inter_start = it * cfg.linear_tile_cols;
            const uint32_t inter_cols = tile_extent(cfg.intermediate_dim,
                                                   cfg.linear_tile_cols, it);
            const std::string tile_tag = row_tag + ".i" + std::to_string(it);

            ParamMap stage_gate_a;
            stage_gate_a["source"] = a_tile;
            stage_gate_a["destination"] = "systolic_array." + tile_tag + ".gate.A";
            stage_gate_a["rows"] = static_cast<int64_t>(row_count);
            stage_gate_a["cols"] = static_cast<int64_t>(cfg.hidden_dim);
            InstructionId gate_a = b.add("dma_stage", "dma",
                                         layer + " MLP gate activation tile r"
                                         + std::to_string(rt) + " i" + std::to_string(it)
                                         + " stage",
                                         stage_gate_a, {a_load});

            ParamMap gate_w_load;
            gate_w_load["source"] = "HBM." + layer + ".W_gate[0:"
                                  + std::to_string(cfg.hidden_dim) + ","
                                  + std::to_string(inter_start) + ":"
                                  + std::to_string(inter_start + inter_cols) + "]";
            gate_w_load["destination"] = "shared_ibuf." + tile_tag + ".W_gate";
            gate_w_load["rows"] = static_cast<int64_t>(cfg.hidden_dim);
            gate_w_load["cols"] = static_cast<int64_t>(inter_cols);
            InstructionId gate_w = b.add("dma_load", "dma",
                                         layer + " MLP gate weight tile i"
                                         + std::to_string(it) + " load",
                                         gate_w_load, deps);

            ParamMap stage_gate_w;
            stage_gate_w["source"] = "shared_ibuf." + tile_tag + ".W_gate";
            stage_gate_w["destination"] = "systolic_array." + tile_tag + ".gate.B";
            stage_gate_w["rows"] = static_cast<int64_t>(cfg.hidden_dim);
            stage_gate_w["cols"] = static_cast<int64_t>(inter_cols);
            InstructionId gate_b = b.add("dma_stage", "dma",
                                         layer + " MLP gate weight tile i"
                                         + std::to_string(it) + " stage",
                                         stage_gate_w, {gate_w});

            ParamMap gate_gemm;
            gate_gemm["source_a"] = "systolic_array." + tile_tag + ".gate.A";
            gate_gemm["source_b"] = "systolic_array." + tile_tag + ".gate.B";
            gate_gemm["destination"] = "shared_obuf." + tile_tag + ".gate";
            gate_gemm["M"] = static_cast<int64_t>(row_count);
            gate_gemm["K"] = static_cast<int64_t>(cfg.hidden_dim);
            gate_gemm["N"] = static_cast<int64_t>(inter_cols);
            InstructionId gate = b.add("gemm", "systolic",
                                       layer + " MLP gate tile r" + std::to_string(rt)
                                       + " i" + std::to_string(it),
                                       gate_gemm, {gate_a, gate_b});

            ParamMap stage_up_a = stage_gate_a;
            stage_up_a["destination"] = "systolic_array." + tile_tag + ".up.A";
            InstructionId up_a = b.add("dma_stage", "dma",
                                       layer + " MLP up activation tile r"
                                       + std::to_string(rt) + " i" + std::to_string(it)
                                       + " stage",
                                       stage_up_a, {a_load});

            ParamMap up_w_load;
            up_w_load["source"] = "HBM." + layer + ".W_up[0:"
                                + std::to_string(cfg.hidden_dim) + ","
                                + std::to_string(inter_start) + ":"
                                + std::to_string(inter_start + inter_cols) + "]";
            up_w_load["destination"] = "shared_ibuf." + tile_tag + ".W_up";
            up_w_load["rows"] = static_cast<int64_t>(cfg.hidden_dim);
            up_w_load["cols"] = static_cast<int64_t>(inter_cols);
            InstructionId up_w = b.add("dma_load", "dma",
                                       layer + " MLP up weight tile i"
                                       + std::to_string(it) + " load",
                                       up_w_load, deps);

            ParamMap stage_up_w;
            stage_up_w["source"] = "shared_ibuf." + tile_tag + ".W_up";
            stage_up_w["destination"] = "systolic_array." + tile_tag + ".up.B";
            stage_up_w["rows"] = static_cast<int64_t>(cfg.hidden_dim);
            stage_up_w["cols"] = static_cast<int64_t>(inter_cols);
            InstructionId up_b = b.add("dma_stage", "dma",
                                       layer + " MLP up weight tile i"
                                       + std::to_string(it) + " stage",
                                       stage_up_w, {up_w});

            ParamMap up_gemm;
            up_gemm["source_a"] = "systolic_array." + tile_tag + ".up.A";
            up_gemm["source_b"] = "systolic_array." + tile_tag + ".up.B";
            up_gemm["destination"] = "shared_obuf." + tile_tag + ".up";
            up_gemm["M"] = static_cast<int64_t>(row_count);
            up_gemm["K"] = static_cast<int64_t>(cfg.hidden_dim);
            up_gemm["N"] = static_cast<int64_t>(inter_cols);
            InstructionId up = b.add("gemm", "systolic",
                                     layer + " MLP up tile r" + std::to_string(rt)
                                     + " i" + std::to_string(it),
                                     up_gemm, {up_a, up_b});

            ParamMap silu;
            silu["source"] = "shared_obuf." + tile_tag + ".gate";
            silu["destination"] = "vector_scratch." + tile_tag + ".silu_gate";
            silu["rows"] = static_cast<int64_t>(row_count);
            silu["cols"] = static_cast<int64_t>(inter_cols);
            InstructionId silu_id = b.add("silu", "vector_core",
                                          layer + " MLP SwiGLU SiLU tile r"
                                          + std::to_string(rt) + " i" + std::to_string(it),
                                          silu, {gate});

            ParamMap mul;
            mul["source_a"] = "vector_scratch." + tile_tag + ".silu_gate";
            mul["source_b"] = "shared_obuf." + tile_tag + ".up";
            mul["destination"] = "shared_ibuf." + tile_tag + ".ff";
            mul["rows"] = static_cast<int64_t>(row_count);
            mul["cols"] = static_cast<int64_t>(inter_cols);
            InstructionId ff = b.add("elementwise_mul", "vector_core",
                                     layer + " MLP SwiGLU multiply tile r"
                                     + std::to_string(rt) + " i" + std::to_string(it),
                                     mul, {silu_id, up});

            for (uint32_t ot = 0; ot < out_tiles; ot++) {
                const uint32_t out_col_start = ot * cfg.linear_tile_cols;
                const uint32_t out_cols = tile_extent(cfg.hidden_dim, cfg.linear_tile_cols, ot);
                const std::string down_tag = tile_tag + ".out.c" + std::to_string(ot);
                const std::string out_tag = row_tag + ".out.c" + std::to_string(ot);

                ParamMap stage_ff;
                stage_ff["source"] = "shared_ibuf." + tile_tag + ".ff";
                stage_ff["destination"] = "systolic_array." + down_tag + ".ff.A";
                stage_ff["rows"] = static_cast<int64_t>(row_count);
                stage_ff["cols"] = static_cast<int64_t>(inter_cols);
                InstructionId ff_stage = b.add("dma_stage", "dma",
                                               layer + " MLP down FF tile r"
                                               + std::to_string(rt) + " i" + std::to_string(it)
                                               + " stage",
                                               stage_ff, {ff});

                ParamMap down_w_load;
                down_w_load["source"] = "HBM." + layer + ".W_down["
                                      + std::to_string(inter_start) + ":"
                                      + std::to_string(inter_start + inter_cols) + ","
                                      + std::to_string(out_col_start) + ":"
                                      + std::to_string(out_col_start + out_cols) + "]";
                down_w_load["destination"] = "shared_ibuf." + down_tag + ".W_down";
                down_w_load["rows"] = static_cast<int64_t>(inter_cols);
                down_w_load["cols"] = static_cast<int64_t>(out_cols);
                InstructionId down_w = b.add("dma_load", "dma",
                                             layer + " MLP down weight tile i"
                                             + std::to_string(it) + " c"
                                             + std::to_string(ot) + " load",
                                             down_w_load, deps);

                ParamMap stage_down_w;
                stage_down_w["source"] = "shared_ibuf." + down_tag + ".W_down";
                stage_down_w["destination"] = "systolic_array." + down_tag + ".down.B";
                stage_down_w["rows"] = static_cast<int64_t>(inter_cols);
                stage_down_w["cols"] = static_cast<int64_t>(out_cols);
                InstructionId down_b = b.add("dma_stage", "dma",
                                             layer + " MLP down weight tile i"
                                             + std::to_string(it) + " c"
                                             + std::to_string(ot) + " stage",
                                             stage_down_w, {down_w});

                ParamMap down_gemm;
                down_gemm["source_a"] = "systolic_array." + down_tag + ".ff.A";
                down_gemm["source_b"] = "systolic_array." + down_tag + ".down.B";
                down_gemm["destination"] = "shared_obuf." + down_tag + ".partial";
                down_gemm["M"] = static_cast<int64_t>(row_count);
                down_gemm["K"] = static_cast<int64_t>(inter_cols);
                down_gemm["N"] = static_cast<int64_t>(out_cols);
                InstructionId down = b.add("gemm", "systolic",
                                           layer + " MLP down tile r" + std::to_string(rt)
                                           + " i" + std::to_string(it)
                                           + " c" + std::to_string(ot),
                                           down_gemm, {ff_stage, down_b});

                ParamMap acc;
                acc["source_a"] = "shared_obuf." + out_tag + ".O_acc";
                acc["source_b"] = "shared_obuf." + down_tag + ".partial";
                acc["destination"] = "shared_obuf." + out_tag + ".O_acc";
                acc["rows"] = static_cast<int64_t>(row_count);
                acc["cols"] = static_cast<int64_t>(out_cols);
                acc_state[ot] = b.add("accumulate", "vector_core",
                                      layer + " MLP down accumulate tile r"
                                      + std::to_string(rt) + " i" + std::to_string(it)
                                      + " c" + std::to_string(ot),
                                      acc, {acc_state[ot], down});
            }
        }

        for (uint32_t ot = 0; ot < out_tiles; ot++) {
            const uint32_t out_col_start = ot * cfg.linear_tile_cols;
            const uint32_t out_cols = tile_extent(cfg.hidden_dim, cfg.linear_tile_cols, ot);
            const std::string out_tag = row_tag + ".out.c" + std::to_string(ot);

            ParamMap place;
            place["source"] = "shared_obuf." + out_tag + ".O_acc";
            place["destination"] = destination + ".range"
                                 + std::to_string(row_start) + "_"
                                 + std::to_string(row_start + row_count)
                                 + "." + std::to_string(out_col_start) + "_"
                                 + std::to_string(out_col_start + out_cols);
            place["rows"] = static_cast<int64_t>(row_count);
            place["cols"] = static_cast<int64_t>(out_cols);
            InstructionId place_id = b.add("sram_copy", "access_core",
                                           layer + " MLP down place tile r"
                                           + std::to_string(rt) + " c"
                                           + std::to_string(ot),
                                           place, {acc_state[ot]});
            output_tiles.push_back(place_id);
        }
    }

    ParamMap assemble;
    assemble["source"] = destination + ".tiles";
    assemble["destination"] = destination;
    assemble["rows"] = static_cast<int64_t>(rows);
    assemble["cols"] = static_cast<int64_t>(cfg.hidden_dim);
    assemble["tile_rows"] = static_cast<int64_t>(row_tiles);
    assemble["tile_cols"] = static_cast<int64_t>(out_tiles);
    return b.add("sram_copy", "access_core", layer + " MLP down assemble output",
                 assemble, output_tiles);
}

InstructionId append_logits_softmax(Builder& b, const std::string& tag,
                                    const std::string& logits,
                                    const std::string& probs,
                                    uint32_t vocab_size,
                                    const std::vector<InstructionId>& deps) {
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

InstructionId append_sample(Builder& b, const std::string& source,
                            const std::string& destination,
                            uint32_t vocab_size,
                            const std::vector<InstructionId>& deps) {
    ParamMap sample;
    sample["source"] = source;
    sample["destination"] = destination;
    sample["rows"] = static_cast<int64_t>(1);
    sample["cols"] = static_cast<int64_t>(vocab_size);
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

struct Fa2TileState {
    uint32_t qh = 0;
    uint32_t qt = 0;
    uint32_t rows = 0;
    uint32_t q_start = 0;
    std::string tag;
    InstructionId last_m = 0;
    InstructionId last_l = 0;
    InstructionId last_o = 0;
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
        b, l + " RoPE Q", l + "." + step + ".Q",
        "shared_obuf." + l + "." + step + ".Q",
        "shared_obuf." + l + "." + step + ".Q_rope",
        q_len, cfg.head_dim, decode_step, {q_proj});

    InstructionId k_rope = append_rope(
        b, l + " RoPE K", l + "." + step + ".K",
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
        std::vector<Fa2TileState> states;
        states.reserve(static_cast<size_t>(group) * q_tiles);
        for (uint32_t local = 0; local < group; local++) {
            const uint32_t qh = kvh * group + local;
            for (uint32_t qt = 0; qt < q_tiles; qt++) {
                const uint32_t q_rows = tile_extent(q_len, cfg.tile_rows, qt);
                const uint32_t q_start = (cfg.mode == "decode") ? decode_step : qt * cfg.tile_rows;
                const std::string tag = l + "." + step + ".kv" + std::to_string(kvh)
                                      + ".qh" + std::to_string(qh)
                                      + ".qt" + std::to_string(qt);

                ParamMap init_o;
                init_o["destination"] = "shared_obuf." + tag + ".O_acc";
                init_o["rows"] = static_cast<int64_t>(q_rows);
                init_o["cols"] = static_cast<int64_t>(cfg.head_dim);
                init_o["init_value"] = static_cast<int64_t>(0);
                InstructionId init_o_id = b.add("init_fill", "access_core",
                                                "init O " + tag, init_o, {q_rope});

                ParamMap init_m;
                init_m["destination"] = "shared_obuf." + tag + ".m";
                init_m["length"] = static_cast<int64_t>(q_rows);
                init_m["init_value"] = std::string("-inf");
                InstructionId init_m_id = b.add("init_fill", "access_core",
                                                "init m " + tag, init_m, {init_o_id});

                ParamMap init_l;
                init_l["destination"] = "shared_obuf." + tag + ".l";
                init_l["length"] = static_cast<int64_t>(q_rows);
                init_l["init_value"] = static_cast<int64_t>(0);
                InstructionId init_l_id = b.add("init_fill", "access_core",
                                                "init l " + tag, init_l, {init_m_id});

                states.push_back(Fa2TileState{qh, qt, q_rows, q_start, tag,
                                              init_m_id, init_l_id, init_o_id});
            }
        }

        std::vector<InstructionId> slot_free(cfg.kv_stage_buffers);
        for (uint32_t kt = 0; kt < kv_tiles; kt++) {
            // P1.3: FA2 causal block-skip. KV tile kt covers absolute key
            // positions [kv_first, kv_last]. It is fully above the diagonal for
            // a Q tile when kv_first > q_last (every key is in the future).
            const uint32_t kv_first = kt * cfg.tile_cols;
            const uint32_t kv_rows  = tile_extent(kv_len, cfg.tile_cols, kt);
            const uint32_t kv_last  = kv_first + (kv_rows ? kv_rows - 1 : 0);
            if (cfg.causal_block_skip) {
                bool any_needed = false;
                for (const auto& state : states) {
                    const uint32_t q_last = state.q_start + state.rows - 1;
                    if (kv_first <= q_last) { any_needed = true; break; }
                }
                if (!any_needed) continue;   // whole KV tile is in the future
            }

            std::vector<InstructionId> read_deps = cache_write_deps;
            const uint32_t slot = cfg.kv_prefetch == "double_buffer"
                ? kt % cfg.kv_stage_buffers
                : 0;
            if (slot_free[slot]) read_deps.push_back(slot_free[slot]);
            KvTileLoad kv_load = append_kv_tile_load(
                b, cfg, l, step, kvh, kt, kv_len, read_deps);
            std::vector<InstructionId> slot_consumers;

            for (auto& state : states) {
                // P1.3: per (q-tile, kv-tile) causal skip + conditional mask.
                const uint32_t q_first = state.q_start;
                const uint32_t q_last  = state.q_start + state.rows - 1;
                if (cfg.causal_block_skip && kv_first > q_last)
                    continue;                         // this block is all future
                // Masking is only needed where the tile straddles the diagonal
                // (some keys ahead of some queries). Fully-below-diagonal tiles
                // (kv_last <= q_first) need no mask.
                const bool need_mask = !cfg.causal_block_skip || (kv_last > q_first);

                const std::string tile_tag = kv_load.tag + ".qh"
                                           + std::to_string(state.qh)
                                           + ".qt" + std::to_string(state.qt);
                ParamMap stq;
                stq["source"] = "shared_obuf." + l + "." + step + ".Q_rope";
                stq["destination"] = "systolic_array." + tile_tag + ".Q_operand";
                stq["rows"] = static_cast<int64_t>(state.rows);
                stq["cols"] = static_cast<int64_t>(cfg.head_dim);
                InstructionId stage_q = b.add("dma_stage", "dma",
                                              "stage Q " + tile_tag,
                                              stq, {state.last_l});

                ParamMap qk;
                qk["source_a"] = "systolic_array." + tile_tag + ".Q_operand";
                qk["source_b"] = kv_load.k_t;
                qk["destination"] = "shared_obuf." + tile_tag + ".S";
                qk["M"] = static_cast<int64_t>(state.rows);
                qk["K"] = static_cast<int64_t>(cfg.head_dim);
                qk["N"] = static_cast<int64_t>(kv_load.rows);
                InstructionId mat_qk = b.add("gemm", "systolic", "QK " + tile_tag,
                                             qk, {stage_q, kv_load.transpose_k});

                ParamMap scale;
                scale["source"] = "shared_obuf." + tile_tag + ".S";
                scale["destination"] = "shared_obuf." + tile_tag + ".S";
                scale["rows"] = static_cast<int64_t>(state.rows);
                scale["cols"] = static_cast<int64_t>(kv_load.rows);
                InstructionId scale_id = b.add("scale", "vector_core", "scale " + tile_tag,
                                               scale, {mat_qk});

                InstructionId pre_rowmax = scale_id;
                if (need_mask) {
                    ParamMap mask = scale;
                    mask["row_start"] = static_cast<int64_t>(state.q_start);
                    mask["col_start"] = static_cast<int64_t>(kv_load.start);
                    pre_rowmax = b.add("causal_mask", "vector_core",
                                       "causal mask " + tile_tag, mask, {scale_id});
                }

                ParamMap rowmax;
                rowmax["source"] = "shared_obuf." + tile_tag + ".S";
                rowmax["destination"] = "vector_scratch." + tile_tag + ".rowmax";
                rowmax["rows"] = static_cast<int64_t>(state.rows);
                rowmax["cols"] = static_cast<int64_t>(kv_load.rows);
                InstructionId rowmax_id = b.add("rowmax", "vector_core", "rowmax " + tile_tag,
                                                rowmax, {pre_rowmax});

                ParamMap upd_m;
                upd_m["source_m_old"] = "shared_obuf." + state.tag + ".m";
                upd_m["source_rowmax"] = "vector_scratch." + tile_tag + ".rowmax";
                upd_m["destination_m"] = "shared_obuf." + state.tag + ".m";
                upd_m["destination_correction"] = "shared_obuf." + state.tag + ".correction";
                upd_m["length"] = static_cast<int64_t>(state.rows);
                InstructionId upd_m_id = b.add("update_rowmax", "vector_core",
                                               "update m " + tile_tag,
                                               upd_m, {rowmax_id, state.last_m});

                ParamMap exp;
                exp["source_matrix"] = "shared_obuf." + tile_tag + ".S";
                exp["source_shift"] = "shared_obuf." + state.tag + ".m";
                exp["destination"] = "shared_ibuf." + tile_tag + ".P";
                exp["rows"] = static_cast<int64_t>(state.rows);
                exp["cols"] = static_cast<int64_t>(kv_load.rows);
                InstructionId exp_id = b.add("exp_shift", "vector_core", "softmax exp " + tile_tag,
                                             exp, {upd_m_id});

                ParamMap rowsum;
                rowsum["source_p"] = "shared_ibuf." + tile_tag + ".P";
                rowsum["source_correction"] = "shared_obuf." + state.tag + ".correction";
                rowsum["source_l_old"] = "shared_obuf." + state.tag + ".l";
                rowsum["destination"] = "shared_obuf." + state.tag + ".l";
                rowsum["rows"] = static_cast<int64_t>(state.rows);
                rowsum["cols"] = static_cast<int64_t>(kv_load.rows);
                InstructionId rowsum_id = b.add("update_rowsum", "vector_core",
                                                "update l " + tile_tag,
                                                rowsum, {exp_id, upd_m_id, state.last_l});

                ParamMap rescale;
                rescale["source"] = "shared_obuf." + state.tag + ".O_acc";
                rescale["source_scale"] = "shared_obuf." + state.tag + ".correction";
                rescale["destination"] = "shared_obuf." + state.tag + ".O_acc";
                rescale["rows"] = static_cast<int64_t>(state.rows);
                rescale["cols"] = static_cast<int64_t>(cfg.head_dim);
                InstructionId rescale_id = b.add("scale", "vector_core",
                                                 "rescale O " + tile_tag,
                                                 rescale, {upd_m_id, state.last_o});

                ParamMap stp;
                stp["source"] = "shared_ibuf." + tile_tag + ".P";
                stp["destination"] = "systolic_array." + tile_tag + ".P_operand";
                stp["rows"] = static_cast<int64_t>(state.rows);
                stp["cols"] = static_cast<int64_t>(kv_load.rows);
                InstructionId stage_p = b.add("dma_stage", "dma", "stage P " + tile_tag,
                                              stp, {exp_id, kv_load.read_v});

                ParamMap pv;
                pv["source_a"] = "systolic_array." + tile_tag + ".P_operand";
                pv["source_b"] = kv_load.v;
                pv["destination"] = "shared_obuf." + tile_tag + ".Temp";
                pv["M"] = static_cast<int64_t>(state.rows);
                pv["K"] = static_cast<int64_t>(kv_load.rows);
                pv["N"] = static_cast<int64_t>(cfg.head_dim);
                InstructionId mat_pv = b.add("gemm", "systolic", "PV " + tile_tag,
                                             pv, {stage_p, kv_load.read_v});

                ParamMap acc;
                acc["source_a"] = "shared_obuf." + state.tag + ".O_acc";
                acc["source_b"] = "shared_obuf." + tile_tag + ".Temp";
                acc["destination"] = "shared_obuf." + state.tag + ".O_acc";
                acc["rows"] = static_cast<int64_t>(state.rows);
                acc["cols"] = static_cast<int64_t>(cfg.head_dim);
                InstructionId acc_id = b.add("accumulate", "vector_core", "accumulate O " + tile_tag,
                                             acc, {rescale_id, mat_pv});

                state.last_m = upd_m_id;
                state.last_l = rowsum_id;
                state.last_o = acc_id;
                slot_consumers.push_back(acc_id);
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

        for (auto& state : states) {
            ParamMap norm;
            norm["source_matrix"] = "shared_obuf." + state.tag + ".O_acc";
            norm["source_denom"] = "shared_obuf." + state.tag + ".l";
            norm["destination"] = "shared_obuf." + state.tag + ".O";
            norm["rows"] = static_cast<int64_t>(state.rows);
            norm["cols"] = static_cast<int64_t>(cfg.head_dim);
            InstructionId out = b.add("normalize", "vector_core", "normalize " + state.tag,
                                      norm, {state.last_o, state.last_l});

            ParamMap lse;
            lse["source_m"] = "shared_obuf." + state.tag + ".m";
            lse["source_l"] = "shared_obuf." + state.tag + ".l";
            lse["destination"] = "shared_obuf." + state.tag + ".L";
            lse["length"] = static_cast<int64_t>(state.rows);
            InstructionId logsumexp = b.add("logsumexp", "vector_core",
                                            "logsumexp " + state.tag,
                                            lse, {state.last_m, state.last_l, out});
            head_outputs.push_back(logsumexp);
        }
    }

    ParamMap merge;
    merge["source"] = "shared_obuf." + l + "." + step + ".attention_head_tiles";
    merge["destination"] = "shared_obuf." + l + "." + step + ".attention_heads";
    merge["rows"] = static_cast<int64_t>(q_len);
    merge["cols"] = static_cast<int64_t>(cfg.num_q_heads * cfg.head_dim);
    merge["q_tiles"] = static_cast<int64_t>(q_tiles);
    merge["kv_tiles"] = static_cast<int64_t>(kv_tiles);
    merge["num_q_heads"] = static_cast<int64_t>(cfg.num_q_heads);
    merge["head_dim"] = static_cast<int64_t>(cfg.head_dim);
    merge["input_elements"] = static_cast<int64_t>(
        static_cast<uint64_t>(q_len) * cfg.num_q_heads * cfg.head_dim);
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
        cur = {b.add("gather_select", "access_core",
                     "select final position hidden state", last, cur)};
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
        b, tag, "shared_obuf." + tag + ".logits",
        "shared_obuf." + tag + ".probs", cfg.vocab_size, {logits});

    return append_sample(b, "shared_obuf." + tag + ".probs",
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

    InstructionId down_id = append_detailed_mlp_kernel(
        b, cfg, l,
        "shared_obuf." + l + ".mlp_norm",
        "shared_obuf." + l + ".mlp_out",
        q_len, {norm2});

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
    // Programmatic schedules are built by tested code that assigns sequential
    // IDs and wires deps correctly by construction — duplicate IDs and unknown
    // deps are not possible here. Skipping validate() avoids rebuilding three
    // large hash maps / flat-vectors over the full instruction set a second
    // time (the scheduler constructor builds the equivalent structures anyway).
    // validate() is still called on YAML-loaded schedules via from_node().
    return s;
}

}  // namespace

Schedule build_attention_schedule(const LlamaScheduleConfig& input_cfg, bool minimal) {
    const LlamaScheduleConfig cfg = normalize_cfg(input_cfg);
    Builder b;
    b.minimal = minimal;
    b.out.reserve(25000UL * cfg.num_kv_heads + 10000);
    const uint32_t q_len = cfg.mode == "decode" ? 1
                         : (cfg.mode == "prefill" ? cfg.prompt_len : cfg.seq_len);
    const uint32_t kv_len = cfg.mode == "decode" ? cfg.prompt_len + 1 : q_len;
    append_attention(b, cfg, 0, q_len, kv_len, cfg.prompt_len, "HBM.input", {});
    return finish(b);
}

// ---------------------------------------------------------------------------
// Estimate how many instructions the schedule will contain.
//
// Pre-reserving Builder::out to (approximately) the right size is the key
// RAM optimisation: without a reservation the vector doubles ~24 times on a
// full 8B run, leaving ~2 GB of freed old buffers scattered through the heap
// (glibc never releases these pages for sub-mmap-threshold allocations because
// they are interleaved with live param data above them).  A single upfront
// reservation means ONE allocation, zero doublings, zero freed holes.
//
// The estimate intentionally runs ~10% hot so we never under-reserve and fall
// back to doubling.  The small over-reserve sits at the end of the sched_
// block in the Scheduler and is freed at program exit.
// ---------------------------------------------------------------------------
static size_t estimate_instruction_count(const LlamaScheduleConfig& cfg) {
    // Empirical per-layer instruction counts (LLaMA-3-8B, tile 256×256):
    //   val_layer  (1 layer,  mode=layer)        → 366 654  ≈ 367K / layer
    //   val_full8b (32 layers, mode=layer)        → 11 600 000 ≈ 363K / layer
    //   llama_prefill_decode (32 layers, 1 step)  → 11 168 084
    //     ≈ 340K / prefill-layer + 8.5K / decode-layer
    //
    // Use 10 % margin to guarantee no doubling even for slightly larger models.
    const size_t kLayer   = 400000UL;   // 363K actual × 1.10
    const size_t kDecode  =  10000UL;   // 8.5K actual × 1.18
    const size_t kFixed   =  300000UL;  // embedding + output head + misc

    if (cfg.mode == "layer" || cfg.mode == "prefill")
        return kLayer * cfg.num_layers + kFixed;
    if (cfg.mode == "decode")
        return kDecode * cfg.num_layers + kFixed;
    if (cfg.mode == "prefill_decode")
        return kLayer  * cfg.num_layers
             + kDecode * cfg.num_layers * static_cast<size_t>(cfg.generation_steps)
             + kFixed;
    // attention / unknown: generous fallback
    return kLayer * cfg.num_layers + kFixed;
}

Schedule build_transformer_layer_schedule(const LlamaScheduleConfig& input_cfg, bool minimal) {
    const LlamaScheduleConfig cfg = normalize_cfg(input_cfg);
    Builder b;
    b.minimal = minimal;
    b.out.reserve(estimate_instruction_count(cfg));
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

Schedule build_prefill_decode_schedule(const LlamaScheduleConfig& input_cfg, bool minimal) {
    const LlamaScheduleConfig cfg = normalize_cfg(input_cfg);
    Builder b;
    b.minimal = minimal;
    b.out.reserve(estimate_instruction_count(cfg));
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

Schedule build_llama_schedule(const LlamaScheduleConfig& cfg, bool minimal) {
    if (cfg.mode == "attention") return build_attention_schedule(cfg, minimal);
    if (cfg.mode == "layer" || cfg.mode == "prefill" || cfg.mode == "decode")
        return build_transformer_layer_schedule(cfg, minimal);
    if (cfg.mode == "prefill_decode") return build_prefill_decode_schedule(cfg, minimal);
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
    read_scalar(n, "causal_block_skip", cfg.causal_block_skip);
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