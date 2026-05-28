#include <doctest/doctest.h>
#include <algorithm>
#include <sstream>
#include <string>
#include "config/arch_config.h"
#include "core/event_engine.h"
#include "core/tensor_store.h"
#include "schedule/llama_schedule.h"
#include "schedule/op_handlers.h"
#include "schedule/schedule.h"
#include "schedule/scheduler.h"
#include "units/access_unit.h"
#include "units/dma_unit.h"
#include "units/systolic_unit.h"
#include "units/vector_unit.h"

using namespace sim;

namespace {

bool has_op(const Schedule& s, const std::string& op) {
    return std::any_of(s.instructions.begin(), s.instructions.end(),
        [&](const Instruction& inst) { return inst.op == op; });
}

int count_op(const Schedule& s, const std::string& op) {
    return static_cast<int>(std::count_if(s.instructions.begin(), s.instructions.end(),
        [&](const Instruction& inst) { return inst.op == op; }));
}

int count_label_op(const Schedule& s, const std::string& text, const std::string& op) {
    return static_cast<int>(std::count_if(s.instructions.begin(), s.instructions.end(),
        [&](const Instruction& inst) {
            return inst.op == op && inst.label.find(text) != std::string::npos;
        }));
}

const Instruction* find_label(const Schedule& s, const std::string& text) {
    auto it = std::find_if(s.instructions.begin(), s.instructions.end(),
        [&](const Instruction& inst) { return inst.label.find(text) != std::string::npos; });
    return it == s.instructions.end() ? nullptr : &*it;
}

const Instruction* find_label_op(const Schedule& s, const std::string& text,
                                 const std::string& op) {
    auto it = std::find_if(s.instructions.begin(), s.instructions.end(),
        [&](const Instruction& inst) {
            return inst.op == op && inst.label.find(text) != std::string::npos;
        });
    return it == s.instructions.end() ? nullptr : &*it;
}

bool has_cache_label(const Schedule& s, const std::string& op,
                     const std::string& text) {
    return std::any_of(s.instructions.begin(), s.instructions.end(),
        [&](const Instruction& inst) {
            return inst.op == op && inst.label.find(text) != std::string::npos;
        });
}

void register_real_units(EventEngine& engine, TensorStore& ts, const ArchConfig& arch) {
    engine.register_unit(std::make_unique<SystolicUnit>("systolic", arch.systolic, nullptr, &ts));
    engine.register_unit(std::make_unique<DmaUnit>("dma_0", arch, &ts));
    engine.register_unit(std::make_unique<VectorUnit>("vector_core_0", arch.vector_core, nullptr, &ts));
    engine.register_unit(std::make_unique<VectorUnit>("vector_core_1", arch.vector_core, nullptr, &ts));
    engine.register_unit(std::make_unique<AccessUnit>("access_core_0", arch.access_core, nullptr, &ts));
}

void wire(EventEngine& engine, Scheduler& sched, TensorStore& ts) {
    for (UnitId uid = 0; uid < static_cast<UnitId>(engine.num_units()); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<SystolicUnit*>(u)) { x->set_scheduler(&sched); x->set_tensor_store(&ts); }
        if (auto* x = dynamic_cast<DmaUnit*>(u))      { x->set_scheduler(&sched); x->set_tensor_store(&ts); }
        if (auto* x = dynamic_cast<VectorUnit*>(u))   { x->set_scheduler(&sched); x->set_tensor_store(&ts); }
        if (auto* x = dynamic_cast<AccessUnit*>(u))   { x->set_scheduler(&sched); x->set_tensor_store(&ts); }
    }
}

}  // namespace

TEST_CASE("typed FA2 schedule runs through reusable builtin op handlers as timing events") {
    ArchConfig arch;
    arch.systolic.rows = 16;
    arch.systolic.cols = 16;
    arch.systolic.d_head = 16;
    arch.vector_core.simd_width = 8;
    arch.access_core.bandwidth = 16;
    arch.sram.banking_factor = 16;

    TensorStore ts;
    Schedule schedule = Schedule::from_yaml_file(
        std::string(SIM_PROJECT_ROOT) + "/schedules/fa2_single_tile.yaml");
    EventEngine engine(arch.clock_ghz);
    register_real_units(engine, ts, arch);
    OpRegistry reg;
    register_builtin_ops(reg, arch);
    Scheduler sched(engine, reg, schedule);
    wire(engine, sched, ts);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(final_cycle > 0);
}

TEST_CASE("attention schedule includes GQA grouped cache reads and causal mask") {
    LlamaScheduleConfig cfg;
    cfg.mode = "attention";
    cfg.seq_len = 4;
    cfg.num_q_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 8;
    cfg.hidden_dim = 32;
    cfg.tile_rows = 2;
    cfg.tile_cols = 2;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "hbm";

    Schedule s = build_attention_schedule(cfg);
    REQUIRE(!s.instructions.empty());
    REQUIRE(has_op(s, "causal_mask"));
    REQUIRE(has_op(s, "rope_rotate"));
    REQUIRE(has_op(s, "dma_load"));
    REQUIRE(has_op(s, "dma_store"));
    REQUIRE(count_op(s, "transpose") == 4); // num_kv_heads * kv_tiles
    REQUIRE(count_op(s, "normalize") == 16); // kv_heads * group * q_tiles * kv_tiles
}

TEST_CASE("SRAM KV cache uses access-core copies instead of HBM DMA movement") {
    LlamaScheduleConfig cfg;
    cfg.seq_len = 2;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 8;
    cfg.hidden_dim = 16;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "sram";

    Schedule s = build_attention_schedule(cfg);
    REQUIRE(has_op(s, "sram_copy"));
    REQUIRE(has_cache_label(s, "sram_copy", "KV cache read"));
    REQUIRE(has_cache_label(s, "sram_copy", "KV cache write"));
    REQUIRE(!has_cache_label(s, "dma_load", "KV cache read"));
    REQUIRE(!has_cache_label(s, "dma_store", "KV cache write"));
}

TEST_CASE("generated LLaMA GEMMs stage HBM weights through on-chip buffers") {
    LlamaScheduleConfig cfg;
    cfg.mode = "layer";
    cfg.seq_len = 2;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 8;
    cfg.hidden_dim = 16;
    cfg.intermediate_dim = 32;
    cfg.vocab_size = 64;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;

    Schedule s = build_transformer_layer_schedule(cfg);
    for (const Instruction& inst : s.instructions) {
        if (inst.op != "gemm") continue;
        CHECK(pget_str(inst.params, "source_a").find("HBM.") != 0);
        CHECK(pget_str(inst.params, "source_b").find("HBM.") != 0);
    }

    const Instruction* q = find_label_op(s, "L0 Q projection", "gemm");
    REQUIRE(q != nullptr);
    REQUIRE(pget_str(q->params, "source_a").find("systolic_array.L0.S0.Wq.A") == 0);
    REQUIRE(pget_str(q->params, "source_b").find("systolic_array.L0.S0.Wq.r") == 0);

    const Instruction* q_weight_stage = find_label_op(s, "L0 Q projection weight tile", "dma_stage");
    REQUIRE(q_weight_stage != nullptr);
    CHECK(std::find(q->depends_on.begin(), q->depends_on.end(), q_weight_stage->id) != q->depends_on.end());
}

TEST_CASE("LLaMA projection dimensions are explicit in staged GEMMs") {
    LlamaScheduleConfig cfg;
    cfg.mode = "layer";
    cfg.seq_len = 2;
    cfg.num_q_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 8;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.vocab_size = 128;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;

    Schedule s = build_transformer_layer_schedule(cfg);

    const Instruction* q = find_label_op(s, "L0 Q projection", "gemm");
    const Instruction* k = find_label_op(s, "L0 K projection", "gemm");
    const Instruction* v = find_label_op(s, "L0 V projection", "gemm");
    const Instruction* o = find_label_op(s, "L0 output projection", "gemm");
    const Instruction* gate = find_label_op(s, "L0 MLP gate", "gemm");
    const Instruction* up = find_label_op(s, "L0 MLP up", "gemm");
    const Instruction* down = find_label_op(s, "L0 MLP down", "gemm");
    const Instruction* lm = find_label_op(s, "LM head linear logits projection", "gemm");
    REQUIRE(q != nullptr);
    REQUIRE(k != nullptr);
    REQUIRE(v != nullptr);
    REQUIRE(o != nullptr);
    REQUIRE(gate != nullptr);
    REQUIRE(up != nullptr);
    REQUIRE(down != nullptr);
    REQUIRE(lm != nullptr);

    CHECK(pget_int(q->params, "M") == 2);
    CHECK(pget_int(q->params, "K") == 32);
    CHECK(pget_int(q->params, "N") == 32);
    CHECK(pget_int(k->params, "N") == 16);
    CHECK(pget_int(v->params, "N") == 16);
    CHECK(pget_int(o->params, "K") == 32);
    CHECK(pget_int(o->params, "N") == 32);
    CHECK(pget_int(gate->params, "N") == 64);
    CHECK(pget_int(up->params, "N") == 64);
    CHECK(pget_int(down->params, "K") == 64);
    CHECK(pget_int(down->params, "N") == 32);
    CHECK(pget_int(lm->params, "M") == 1);
    CHECK(pget_int(lm->params, "K") == 32);
    CHECK(pget_int(lm->params, "N") == 128);
}

TEST_CASE("detailed LLaMA schedule decomposes large linear projections into MXU tiles") {
    LlamaScheduleConfig cfg;
    cfg.mode = "layer";
    cfg.seq_len = 2;
    cfg.num_q_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.vocab_size = 128;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.linear_tile_rows = 1;
    cfg.linear_tile_cols = 16;

    Schedule s = build_transformer_layer_schedule(cfg);

    CHECK(count_label_op(s, "L0 Q projection tile r", "gemm") == 4); // 2 row tiles * 2 col tiles
    CHECK(count_label_op(s, "L0 Q projection weight tile", "dma_load") == 4);
    CHECK(count_label_op(s, "L0 Q projection place tile", "sram_copy") == 4);
    REQUIRE(find_label_op(s, "L0 Q projection assemble output", "sram_copy") != nullptr);

    const Instruction* q_tile = find_label_op(s, "L0 Q projection tile r0 c0", "gemm");
    REQUIRE(q_tile != nullptr);
    CHECK(pget_int(q_tile->params, "M") == 1);
    CHECK(pget_int(q_tile->params, "K") == 32);
    CHECK(pget_int(q_tile->params, "N") == 16);
}

TEST_CASE("detailed LLaMA schedule expands norm, RoPE, SwiGLU, logits, and sample phases") {
    LlamaScheduleConfig cfg;
    cfg.mode = "layer";
    cfg.seq_len = 2;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.hidden_dim = 16;
    cfg.intermediate_dim = 32;
    cfg.vocab_size = 64;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.linear_tile_rows = 1;
    cfg.linear_tile_cols = 16;

    Schedule s = build_transformer_layer_schedule(cfg);

    CHECK(!has_op(s, "rmsnorm"));
    CHECK(has_op(s, "square"));
    CHECK(has_op(s, "row_reduce_sum"));
    CHECK(has_op(s, "rsqrt"));
    CHECK(has_op(s, "norm_scale"));

    CHECK(!has_op(s, "rope"));
    CHECK(has_op(s, "rope_pair_split"));
    CHECK(has_op(s, "rope_rotate"));
    CHECK(has_op(s, "rope_store"));

    CHECK(!has_op(s, "silu_mul"));
    CHECK(has_op(s, "silu"));
    CHECK(has_op(s, "elementwise_mul"));

    CHECK(!has_op(s, "softmax"));
    CHECK(has_op(s, "exp_shift"));
    CHECK(has_op(s, "normalize"));
    CHECK(!has_op(s, "sample_token"));
    CHECK(has_op(s, "sample_top1"));
    CHECK(has_op(s, "gather_select"));
}

TEST_CASE("LLaMA schedule derives head_dim and GQA group size from model shape") {
    LlamaScheduleConfig cfg;
    cfg.mode = "layer";
    cfg.seq_len = 2;
    cfg.num_q_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.gqa_group_size = 0;
    cfg.head_dim = 0;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.vocab_size = 128;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;

    Schedule s = build_transformer_layer_schedule(cfg);

    const Instruction* q = find_label_op(s, "L0 Q projection", "gemm");
    const Instruction* k = find_label_op(s, "L0 K projection", "gemm");
    const Instruction* o = find_label_op(s, "L0 output projection", "gemm");
    const Instruction* merge = find_label_op(s, "L0 merge attention tiles", "attention_merge");
    REQUIRE(q != nullptr);
    REQUIRE(k != nullptr);
    REQUIRE(o != nullptr);
    REQUIRE(merge != nullptr);

    CHECK(pget_int(q->params, "N") == 32); // num_q_heads * derived head_dim
    CHECK(pget_int(k->params, "N") == 16); // num_kv_heads * derived head_dim
    CHECK(pget_int(o->params, "K") == 32);
    CHECK(pget_int(merge->params, "num_q_heads") == 4);
    CHECK(pget_int(merge->params, "head_dim") == 8);
}

TEST_CASE("LLaMA config parser derives omitted head_dim and GQA group size") {
    LlamaScheduleConfig cfg = llama_config_from_yaml_string(R"(
llama:
  mode: attention
  seq_len: 2
  num_q_heads: 8
  num_kv_heads: 2
  hidden_dim: 64
  intermediate_dim: 128
  tile_rows: 1
  tile_cols: 1
)");
    CHECK(cfg.head_dim == 8);
    CHECK(cfg.gqa_group_size == 4);

    Schedule s = build_llama_schedule(cfg);
    const Instruction* q = find_label_op(s, "L0 Q projection", "gemm");
    const Instruction* k = find_label_op(s, "L0 K projection", "gemm");
    REQUIRE(q != nullptr);
    REQUIRE(k != nullptr);
    CHECK(pget_int(q->params, "N") == 64);
    CHECK(pget_int(k->params, "N") == 16);
}

TEST_CASE("LLaMA schedule rejects inconsistent attention dimensions") {
    LlamaScheduleConfig cfg;
    cfg.num_q_heads = 3;
    cfg.num_kv_heads = 1;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    CHECK_THROWS_WITH_AS(build_attention_schedule(cfg),
                         doctest::Contains("hidden_dim must be divisible by num_q_heads"),
                         std::runtime_error);

    cfg.num_q_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.hidden_dim = 32;
    cfg.head_dim = 16;
    CHECK_THROWS_WITH_AS(build_attention_schedule(cfg),
                         doctest::Contains("head_dim must equal hidden_dim/num_q_heads"),
                         std::runtime_error);

    cfg.head_dim = 8;
    cfg.gqa_group_size = 3;
    CHECK_THROWS_WITH_AS(build_attention_schedule(cfg),
                         doctest::Contains("gqa_group_size must equal num_q_heads/num_kv_heads"),
                         std::runtime_error);
}

TEST_CASE("attention_merge handler charges for tile contributions plus output assembly") {
    ArchConfig arch;
    arch.vector_core.simd_width = 8;

    Schedule s;
    Instruction merge;
    merge.id = 0;
    merge.op = "attention_merge";
    merge.unit = "vector_core";
    merge.label = "merge four tile contributions";
    merge.params["source"] = std::string("shared_obuf.tile_outputs");
    merge.params["destination"] = std::string("shared_obuf.attention_heads");
    merge.params["rows"] = static_cast<int64_t>(2);
    merge.params["cols"] = static_cast<int64_t>(4);
    merge.params["input_elements"] = static_cast<int64_t>(32);
    merge.params["output_elements"] = static_cast<int64_t>(8);
    s.instructions.push_back(std::move(merge));
    s.validate();

    TensorStore ts;
    EventEngine engine(arch.clock_ghz);
    engine.register_unit(std::make_unique<VectorUnit>("vector_core_0", arch.vector_core, nullptr, &ts));
    OpRegistry reg;
    register_builtin_ops(reg, arch);
    Scheduler sched(engine, reg, s);
    wire(engine, sched, ts);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    CHECK(final_cycle == 5); // ceil((32 input + 8 output elements) / SIMD 8)
}

TEST_CASE("transformer layer schedule contains attention and MLP residual structure") {
    LlamaScheduleConfig cfg;
    cfg.seq_len = 2;
    cfg.num_q_heads = 1;
    cfg.num_kv_heads = 1;
    cfg.gqa_group_size = 1;
    cfg.head_dim = 8;
    cfg.hidden_dim = 8;
    cfg.intermediate_dim = 16;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;

    Schedule s = build_transformer_layer_schedule(cfg);
    REQUIRE(has_op(s, "norm_scale"));
    REQUIRE(has_op(s, "elementwise_mul"));
    REQUIRE(count_op(s, "residual_add") == 2);
    REQUIRE(count_op(s, "gemm") >= 7); // Q/K/V/O + gate/up/down
    REQUIRE(has_op(s, "embedding_lookup"));
    REQUIRE(has_op(s, "sample_top1"));
}

TEST_CASE("prefill decode schedule writes cache once then decodes repeatedly") {
    LlamaScheduleConfig cfg;
    cfg.prompt_len = 3;
    cfg.seq_len = 3;
    cfg.generation_steps = 2;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 8;
    cfg.hidden_dim = 16;
    cfg.intermediate_dim = 32;
    cfg.tile_rows = 1;
    cfg.tile_cols = 2;
    cfg.kv_cache_location = "hbm";

    Schedule s = build_prefill_decode_schedule(cfg);
    REQUIRE(has_op(s, "dma_store"));
    REQUIRE(has_op(s, "dma_load"));
    REQUIRE(count_op(s, "residual_add") == 6); // one layer: 2 residuals * (prefill + 2 decode)

    const auto first_decode = std::find_if(s.instructions.begin(), s.instructions.end(),
        [](const Instruction& inst) {
            return inst.label.find("S3") != std::string::npos
                || pget_str(inst.params, "destination").find("S3") != std::string::npos;
        });
    REQUIRE(first_decode != s.instructions.end());
}

TEST_CASE("decode cache reads use the same range address scheme as cache writes") {
    LlamaScheduleConfig cfg;
    cfg.mode = "prefill_decode";
    cfg.prompt_len = 2;
    cfg.seq_len = 2;
    cfg.generation_steps = 1;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 8;
    cfg.hidden_dim = 16;
    cfg.intermediate_dim = 32;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "hbm";

    Schedule s = build_prefill_decode_schedule(cfg);

    bool wrote_decode_k = false;
    bool read_decode_k = false;
    for (const Instruction& inst : s.instructions) {
        const std::string src = pget_str(inst.params, "source");
        const std::string dst = pget_str(inst.params, "destination");
        if (inst.op == "dma_store" && dst == "HBM.kv_cache.L0.kv0.K.range2_3")
            wrote_decode_k = true;
        if (inst.op == "dma_load" && src == "HBM.kv_cache.L0.kv0.K.range2_3")
            read_decode_k = true;
    }
    REQUIRE(wrote_decode_k);
    REQUIRE(read_decode_k);
}

TEST_CASE("SRAM KV cache capacity is validated when configured") {
    LlamaScheduleConfig cfg;
    cfg.mode = "attention";
    cfg.seq_len = 8;
    cfg.num_layers = 3;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 16;
    cfg.hidden_dim = 32;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.max_seq_len = 8;
    cfg.dtype_bytes = 2;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "sram";
    cfg.sram_kv_capacity_kb = 1;

    CHECK_THROWS_WITH_AS(build_attention_schedule(cfg),
                         doctest::Contains("SRAM KV cache capacity exceeded"),
                         std::runtime_error);

    cfg.sram_kv_capacity_kb = 4;
    CHECK_NOTHROW(build_attention_schedule(cfg));
}

TEST_CASE("llama workload YAML parser feeds public schedule dispatcher") {
    LlamaScheduleConfig cfg = llama_config_from_yaml_string(R"(
llama:
  mode: prefill
  prompt_len: 3
  seq_len: 9
  num_q_heads: 4
  num_kv_heads: 2
  gqa_group_size: 2
  head_dim: 8
  hidden_dim: 32
  intermediate_dim: 64
  tile_rows: 1
  tile_cols: 2
  kv_cache_enabled: true
  kv_cache_location: sram
)");
    Schedule s = build_llama_schedule(cfg);
    REQUIRE(cfg.mode == "prefill");
    REQUIRE(cfg.prompt_len == 3);
    REQUIRE(has_op(s, "sram_copy"));
    REQUIRE(has_op(s, "norm_scale"));
    REQUIRE(has_op(s, "causal_mask"));
}

TEST_CASE("full prefill/decode pipeline has diagram-level events with explicit dimensions") {
    LlamaScheduleConfig cfg;
    cfg.mode = "prefill_decode";
    cfg.prompt_len = 4;
    cfg.seq_len = 4;
    cfg.generation_steps = 1;
    cfg.num_layers = 2;
    cfg.num_q_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.gqa_group_size = 2;
    cfg.head_dim = 8;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.vocab_size = 128;
    cfg.tile_rows = 2;
    cfg.tile_cols = 2;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "hbm";

    Schedule s = build_prefill_decode_schedule(cfg);

    REQUIRE(has_op(s, "embedding_lookup"));
    REQUIRE(has_op(s, "attention_merge"));
    REQUIRE(has_op(s, "sample_top1"));
    REQUIRE(has_op(s, "token_feedback"));
    REQUIRE(count_op(s, "residual_add") == 8); // 2 residuals * 2 layers * (prefill + 1 decode)

    const Instruction* emb = find_label(s, "token embedding lookup");
    REQUIRE(emb != nullptr);
    REQUIRE(emb->unit == "dma");
    REQUIRE(pget_int(emb->params, "rows") == 4);
    REQUIRE(pget_int(emb->params, "cols") == 32);

    const Instruction* lm = find_label_op(s, "LM head linear logits projection", "gemm");
    REQUIRE(lm != nullptr);
    REQUIRE(lm->unit == "systolic");
    REQUIRE(pget_int(lm->params, "M") == 1);
    REQUIRE(pget_int(lm->params, "K") == 32);
    REQUIRE(pget_int(lm->params, "N") == 128);

    const Instruction* sm = find_label(s, "logits softmax normalize");
    REQUIRE(sm != nullptr);
    REQUIRE(sm->unit == "vector_core");
    REQUIRE(pget_int(sm->params, "rows") == 1);
    REQUIRE(pget_int(sm->params, "cols") == 128);

    const Instruction* merge = find_label(s, "merge attention tiles");
    REQUIRE(merge != nullptr);
    REQUIRE(merge->unit == "vector_core");
    REQUIRE(pget_int(merge->params, "rows") == 4);
    REQUIRE(pget_int(merge->params, "cols") == 32);
    REQUIRE(pget_int(merge->params, "q_tiles") == 2);
    REQUIRE(pget_int(merge->params, "kv_tiles") == 2);
    REQUIRE(pget_int(merge->params, "num_q_heads") == 4);
    REQUIRE(pget_int(merge->params, "head_dim") == 8);
    REQUIRE(pget_int(merge->params, "input_elements") == 256);
    REQUIRE(pget_int(merge->params, "output_elements") == 128);

    const Instruction* feedback = find_label(s, "feed sampled token back");
    REQUIRE(feedback != nullptr);
    REQUIRE(feedback->unit == "vector_core");
    REQUIRE(pget_int(feedback->params, "rows") == 1);
    REQUIRE(pget_int(feedback->params, "cols") == 1);
}
