#include <doctest/doctest.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "config/arch_config.h"
#include "core/event_engine.h"
#include "schedule/llama_schedule.h"
#include "schedule/op_handlers.h"
#include "schedule/schedule.h"
#include "schedule/scheduler.h"
#include "schedule/tiler.h"
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

bool depends_on(const Instruction& inst, InstructionId dependency) {
    return std::find(inst.depends_on.begin(), inst.depends_on.end(), dependency)
           != inst.depends_on.end();
}

void register_real_units(EventEngine& engine, const ArchConfig& arch) {
    for (uint32_t i = 0; i < arch.systolic_units; i++)
        engine.register_unit(std::make_unique<SystolicUnit>(
            "systolic_" + std::to_string(i), arch.systolic));
    engine.register_unit(std::make_unique<DmaUnit>("dma_0", arch));
    engine.register_unit(std::make_unique<VectorUnit>("vector_core_0", arch.vector_core));
    engine.register_unit(std::make_unique<VectorUnit>("vector_core_1", arch.vector_core));
    engine.register_unit(std::make_unique<AccessUnit>("access_core_0", arch.access_core));
}

void wire(EventEngine& engine, Scheduler& sched) {
    for (UnitId uid = 0; uid < static_cast<UnitId>(engine.num_units()); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<SystolicUnit*>(u)) x->set_scheduler(&sched);
        if (auto* x = dynamic_cast<DmaUnit*>(u))      x->set_scheduler(&sched);
        if (auto* x = dynamic_cast<VectorUnit*>(u))   x->set_scheduler(&sched);
        if (auto* x = dynamic_cast<AccessUnit*>(u))   x->set_scheduler(&sched);
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

    Schedule schedule = Schedule::from_yaml_file(
        std::string(SIM_PROJECT_ROOT) + "/schedules/fa2_single_tile.yaml");
    EventEngine engine(arch.clock_ghz);
    register_real_units(engine, arch);
    OpRegistry reg;
    register_builtin_ops(reg, arch);
    Scheduler sched(engine, reg, schedule);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(final_cycle > 0);
}

TEST_CASE("multiple systolic units execute independent GEMMs in parallel") {
    ArchConfig arch;
    arch.systolic.rows = 4;
    arch.systolic.cols = 4;
    arch.systolic.d_head = 16;
    arch.systolic.bidirectional = false;
    arch.systolic_units = 2;

    Schedule schedule;
    schedule.instructions = {
        Instruction{1, "gemm", "systolic",
                    {{"M", int64_t{4}}, {"K", int64_t{16}}, {"N", int64_t{4}}},
                    {}, "independent GEMM 0"},
        Instruction{2, "gemm", "systolic",
                    {{"M", int64_t{4}}, {"K", int64_t{16}}, {"N", int64_t{4}}},
                    {}, "independent GEMM 1"},
    };

    EventEngine engine(arch.clock_ghz);
    register_real_units(engine, arch);
    OpRegistry reg;
    register_builtin_ops(reg, arch);
    Scheduler sched(engine, reg, schedule);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    // Two independent GEMMs on two units run in parallel, so the run finishes at
    // a single GEMM's latency (weight-stationary model), NOT 2x (serial).
    const Cycle one = systolic_gemm_latency(arch.systolic, 4, 16, 4);
    REQUIRE(final_cycle == one);
    REQUIRE(final_cycle < 2 * one);
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
    cfg.causal_block_skip = false;  // this case validates the full FA2 inner-loop
                                    // chain; causal-skip is covered separately.

    Schedule s = build_attention_schedule(cfg);
    REQUIRE(!s.instructions.empty());
    REQUIRE(has_op(s, "causal_mask"));
    REQUIRE(has_op(s, "rope_rotate"));
    REQUIRE(has_op(s, "dma_load"));
    REQUIRE(has_op(s, "dma_store"));
    REQUIRE(count_op(s, "transpose") == 4); // num_kv_heads * kv_tiles
    REQUIRE(count_op(s, "normalize") == 8); // num_q_heads * q_tiles, after all KV tiles
    REQUIRE(count_op(s, "logsumexp") == 8);

    const Instruction* init_o = find_label_op(s, "init O L0.S128.kv0.qh0.qt0", "init_fill");
    const Instruction* init_m = find_label_op(s, "init m L0.S128.kv0.qh0.qt0", "init_fill");
    const Instruction* init_l = find_label_op(s, "init l L0.S128.kv0.qh0.qt0", "init_fill");
    REQUIRE(init_o != nullptr);
    REQUIRE(init_m != nullptr);
    REQUIRE(init_l != nullptr);

    const std::vector<std::pair<std::string, std::string>> fa2_inner = {
        {"stage Q L0.S128.kv0.tile0.qh0.qt0", "dma_stage"},
        {"QK L0.S128.kv0.tile0.qh0.qt0", "gemm"},
        {"scale L0.S128.kv0.tile0.qh0.qt0", "scale"},
        {"causal mask L0.S128.kv0.tile0.qh0.qt0", "causal_mask"},
        {"rowmax L0.S128.kv0.tile0.qh0.qt0", "rowmax"},
        {"update m L0.S128.kv0.tile0.qh0.qt0", "update_rowmax"},
        {"softmax exp L0.S128.kv0.tile0.qh0.qt0", "exp_shift"},
        {"update l L0.S128.kv0.tile0.qh0.qt0", "update_rowsum"},
        {"rescale O L0.S128.kv0.tile0.qh0.qt0", "scale"},
        {"stage P L0.S128.kv0.tile0.qh0.qt0", "dma_stage"},
        {"PV L0.S128.kv0.tile0.qh0.qt0", "gemm"},
        {"accumulate O L0.S128.kv0.tile0.qh0.qt0", "accumulate"},
    };

    std::vector<const Instruction*> inner_events;
    inner_events.reserve(fa2_inner.size());
    for (const auto& expected : fa2_inner) {
        const Instruction* inst = find_label_op(s, expected.first, expected.second);
        REQUIRE(inst != nullptr);
        inner_events.push_back(inst);
    }
    CHECK(depends_on(*inner_events[1], inner_events[0]->id));  // QK waits on staged Q.
    CHECK(depends_on(*inner_events[2], inner_events[1]->id));  // scale waits on QK.
    CHECK(depends_on(*inner_events[3], inner_events[2]->id));  // causal mask waits on scale.
    CHECK(depends_on(*inner_events[4], inner_events[3]->id));  // rowmax waits on mask.
    CHECK(depends_on(*inner_events[5], inner_events[4]->id));  // m update waits on rowmax.
    CHECK(depends_on(*inner_events[5], init_m->id));
    CHECK(depends_on(*inner_events[6], inner_events[5]->id));  // exp waits on m update.
    CHECK(depends_on(*inner_events[7], inner_events[6]->id));  // l update waits on exp.
    CHECK(depends_on(*inner_events[7], init_l->id));
    CHECK(depends_on(*inner_events[8], inner_events[5]->id));  // O rescale waits on m update.
    CHECK(depends_on(*inner_events[8], init_o->id));
    CHECK(depends_on(*inner_events[9], inner_events[6]->id));  // stage P waits on exp.
    CHECK(depends_on(*inner_events[10], inner_events[9]->id)); // PV waits on staged P.
    CHECK(depends_on(*inner_events[11], inner_events[8]->id)); // accumulate waits on rescale.
    CHECK(depends_on(*inner_events[11], inner_events[10]->id));

    const Instruction* upd0 = find_label_op(s, "update m L0.S128.kv0.tile0.qh0.qt0", "update_rowmax");
    const Instruction* upd1 = find_label_op(s, "update m L0.S128.kv0.tile1.qh0.qt0", "update_rowmax");
    const Instruction* l0 = find_label_op(s, "update l L0.S128.kv0.tile0.qh0.qt0", "update_rowsum");
    const Instruction* l1 = find_label_op(s, "update l L0.S128.kv0.tile1.qh0.qt0", "update_rowsum");
    const Instruction* acc0 = find_label_op(s, "accumulate O L0.S128.kv0.tile0.qh0.qt0", "accumulate");
    const Instruction* acc1 = find_label_op(s, "accumulate O L0.S128.kv0.tile1.qh0.qt0", "accumulate");
    const Instruction* rescale1 = find_label_op(s, "rescale O L0.S128.kv0.tile1.qh0.qt0", "scale");
    const Instruction* norm = find_label_op(s, "normalize L0.S128.kv0.qh0.qt0", "normalize");
    const Instruction* lse = find_label_op(s, "logsumexp L0.S128.kv0.qh0.qt0", "logsumexp");
    REQUIRE(upd0 != nullptr);
    REQUIRE(upd1 != nullptr);
    REQUIRE(l0 != nullptr);
    REQUIRE(l1 != nullptr);
    REQUIRE(acc0 != nullptr);
    REQUIRE(acc1 != nullptr);
    REQUIRE(rescale1 != nullptr);
    REQUIRE(norm != nullptr);
    REQUIRE(lse != nullptr);
    CHECK(depends_on(*upd1, upd0->id));
    CHECK(depends_on(*l1, l0->id));
    CHECK(depends_on(*rescale1, acc0->id));
    CHECK(depends_on(*norm, acc1->id));
    CHECK(depends_on(*norm, l1->id));
    CHECK(depends_on(*lse, upd1->id));
    CHECK(depends_on(*lse, l1->id));
    CHECK(depends_on(*lse, norm->id));
}

TEST_CASE("P1.3 causal block-skip removes fully-future KV tiles and masks") {
    LlamaScheduleConfig cfg;
    cfg.mode = "attention";
    cfg.seq_len = 4;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 2;
    cfg.gqa_group_size = 1;
    cfg.head_dim = 8;
    cfg.hidden_dim = 16;
    cfg.tile_rows = 2;
    cfg.tile_cols = 2;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "hbm";

    cfg.causal_block_skip = false;
    Schedule full = build_attention_schedule(cfg);
    cfg.causal_block_skip = true;
    Schedule skip = build_attention_schedule(cfg);

    // q-tile qt0 covers positions 0..1; KV tile1 covers 2..3 (entirely future).
    // Present without skip, gone with skip.
    CHECK(find_label_op(full, "update m L0.S128.kv0.tile1.qh0.qt0", "update_rowmax") != nullptr);
    CHECK(find_label_op(skip, "update m L0.S128.kv0.tile1.qh0.qt0", "update_rowmax") == nullptr);

    // q-tile qt1 covers 2..3 and straddles the diagonal at KV tile1 -> kept.
    CHECK(find_label_op(skip, "update m L0.S128.kv0.tile1.qh0.qt1", "update_rowmax") != nullptr);

    // Fully-below-diagonal block (qt1 q=2..3, KV tile0 kv=0..1) needs no mask.
    CHECK(find_label_op(skip, "causal mask L0.S128.kv0.tile0.qh0.qt1", "causal_mask") == nullptr);
    // Diagonal block still masked.
    CHECK(find_label_op(skip, "causal mask L0.S128.kv0.tile1.qh0.qt1", "causal_mask") != nullptr);

    // Skipping strictly reduces work.
    CHECK(skip.instructions.size() < full.instructions.size());
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

TEST_CASE("oversized generated GEMM tiles expand through the original Tiler") {
    LlamaScheduleConfig cfg;
    cfg.mode = "layer";
    cfg.seq_len = 4;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.hidden_dim = 16;
    cfg.intermediate_dim = 32;
    cfg.vocab_size = 64;
    cfg.tile_rows = 2;
    cfg.tile_cols = 2;
    cfg.linear_tile_rows = 4;
    cfg.linear_tile_cols = 16;

    Schedule logical = build_transformer_layer_schedule(cfg);
    REQUIRE(count_label_op(logical, "L0 Q projection tile r0 c0", "gemm") == 1);

    ArchConfig arch;
    arch.systolic.rows = 2;
    arch.systolic.cols = 8;
    arch.systolic.d_head = 16;

    Schedule expanded = Tiler::expand_gemm_subtiles(logical, arch);

    CHECK(count_label_op(expanded, "L0 Q projection tile r0 c0 / STAGE Q_sub_r",
                         "dma_stage") == 2);
    CHECK(count_label_op(expanded, "L0 Q projection tile r0 c0 / S[r",
                         "gemm") == 4);

    for (const Instruction& inst : expanded.instructions) {
        if (inst.op != "gemm"
            || inst.label.find("L0 Q projection tile r0 c0 / S[r") == std::string::npos)
            continue;
        CHECK(pget_int(inst.params, "M") <= 2);
        CHECK(pget_int(inst.params, "N") <= 8);
        CHECK(pget_int(inst.params, "K") == 16);
        CHECK(pget_int(inst.params, "subtiled_from") >= 0);
    }
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

TEST_CASE("LLaMA MLP streams a tile-level SwiGLU down-projection kernel") {
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

    CHECK(count_label_op(s, "L0 MLP gate tile r", "gemm") == 4);
    CHECK(count_label_op(s, "L0 MLP up tile r", "gemm") == 4);
    CHECK(count_label_op(s, "L0 MLP SwiGLU SiLU tile r", "silu") == 4);
    CHECK(count_label_op(s, "L0 MLP SwiGLU multiply tile r", "elementwise_mul") == 4);
    CHECK(count_label_op(s, "L0 MLP down tile r", "gemm") == 4);
    CHECK(count_label_op(s, "L0 MLP down accumulate tile r", "accumulate") == 4);
    CHECK(find_label_op(s, "L0 MLP down assemble output", "sram_copy") != nullptr);
    CHECK(find_label_op(s, "L0 MLP gate assemble output", "sram_copy") == nullptr);
    CHECK(find_label_op(s, "L0 MLP up assemble output", "sram_copy") == nullptr);

    const Instruction* gate = find_label_op(s, "L0 MLP gate tile r0 i0", "gemm");
    const Instruction* up = find_label_op(s, "L0 MLP up tile r0 i0", "gemm");
    const Instruction* silu = find_label_op(s, "L0 MLP SwiGLU SiLU tile r0 i0", "silu");
    const Instruction* mul = find_label_op(s, "L0 MLP SwiGLU multiply tile r0 i0", "elementwise_mul");
    const Instruction* down0 = find_label_op(s, "L0 MLP down tile r0 i0 c0", "gemm");
    const Instruction* down1 = find_label_op(s, "L0 MLP down tile r0 i1 c0", "gemm");
    const Instruction* acc0 = find_label_op(s, "L0 MLP down accumulate tile r0 i0 c0", "accumulate");
    const Instruction* acc1 = find_label_op(s, "L0 MLP down accumulate tile r0 i1 c0", "accumulate");
    REQUIRE(gate != nullptr);
    REQUIRE(up != nullptr);
    REQUIRE(silu != nullptr);
    REQUIRE(mul != nullptr);
    REQUIRE(down0 != nullptr);
    REQUIRE(down1 != nullptr);
    REQUIRE(acc0 != nullptr);
    REQUIRE(acc1 != nullptr);

    CHECK(pget_int(gate->params, "M") == 1);
    CHECK(pget_int(gate->params, "K") == 16);
    CHECK(pget_int(gate->params, "N") == 16);
    CHECK(pget_int(up->params, "K") == 16);
    CHECK(pget_int(up->params, "N") == 16);
    CHECK(pget_int(down0->params, "M") == 1);
    CHECK(pget_int(down0->params, "K") == 16);
    CHECK(pget_int(down0->params, "N") == 16);
    CHECK(pget_int(down1->params, "K") == 16);

    CHECK(depends_on(*silu, gate->id));
    CHECK(depends_on(*mul, silu->id));
    CHECK(depends_on(*mul, up->id));
    CHECK(depends_on(*acc0, down0->id));
    CHECK(depends_on(*acc1, acc0->id));
    CHECK(depends_on(*acc1, down1->id));
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

    cfg.gqa_group_size = 2;
    cfg.schedule_granularity = "coarse";
    CHECK_THROWS_WITH_AS(build_attention_schedule(cfg),
                         doctest::Contains("only detailed schedule generation is supported"),
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

    EventEngine engine(arch.clock_ghz);
    engine.register_unit(std::make_unique<VectorUnit>("vector_core_0", arch.vector_core));
    OpRegistry reg;
    register_builtin_ops(reg, arch);
    Scheduler sched(engine, reg, s);
    wire(engine, sched);

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
        if (inst.op == "dma_store" && dst == "HBM.kv_cache.L0.kv0.K.page2.block2.range2_3")
            wrote_decode_k = true;
        if (inst.op == "dma_load" && src == "HBM.kv_cache.L0.kv0.K.page2.block2.range2_3")
            read_decode_k = true;
    }
    REQUIRE(wrote_decode_k);
    REQUIRE(read_decode_k);
}

TEST_CASE("KV cache prefetch alternates staging slots and read-aheads independent K/V tiles") {
    LlamaScheduleConfig cfg;
    cfg.mode = "attention";
    cfg.seq_len = 3;
    cfg.num_q_heads = 1;
    cfg.num_kv_heads = 1;
    cfg.hidden_dim = 8;
    cfg.intermediate_dim = 16;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "hbm";
    cfg.kv_prefetch = "double_buffer";
    cfg.kv_stage_buffers = 2;

    Schedule s = build_attention_schedule(cfg);

    const Instruction* k0 = find_label_op(s, "KV cache read HBM.kv_cache.L0.kv0.K.page0.block0.range0_1", "dma_load");
    const Instruction* v0 = find_label_op(s, "KV cache read HBM.kv_cache.L0.kv0.V.page0.block0.range0_1", "dma_load");
    const Instruction* k1 = find_label_op(s, "KV cache read HBM.kv_cache.L0.kv0.K.page1.block1.range1_2", "dma_load");
    const Instruction* k2 = find_label_op(s, "KV cache read HBM.kv_cache.L0.kv0.K.page2.block2.range2_3", "dma_load");
    const Instruction* release0 = find_label_op(s, "release KV stage slot0 after L0.S128.kv0.tile0", "kv_stage_release");
    const Instruction* qk0 = find_label_op(s, "QK L0.S128.kv0.tile0.qh0.qt0", "gemm");
    const Instruction* pv0 = find_label_op(s, "PV L0.S128.kv0.tile0.qh0.qt0", "gemm");
    REQUIRE(k0 != nullptr);
    REQUIRE(v0 != nullptr);
    REQUIRE(k1 != nullptr);
    REQUIRE(k2 != nullptr);
    REQUIRE(release0 != nullptr);
    REQUIRE(qk0 != nullptr);
    REQUIRE(pv0 != nullptr);

    CHECK(pget_str(k0->params, "destination") == "shared_ibuf.kv_stage.kv0.slot0.K");
    CHECK(pget_str(k1->params, "destination") == "shared_ibuf.kv_stage.kv0.slot1.K");
    CHECK(pget_str(k2->params, "destination") == "shared_ibuf.kv_stage.kv0.slot0.K");

    CHECK(std::find(v0->depends_on.begin(), v0->depends_on.end(), k0->id) == v0->depends_on.end());
    CHECK(std::find(k1->depends_on.begin(), k1->depends_on.end(), qk0->id) == k1->depends_on.end());
    CHECK(std::find(k1->depends_on.begin(), k1->depends_on.end(), pv0->id) == k1->depends_on.end());
    CHECK(std::find(k2->depends_on.begin(), k2->depends_on.end(), release0->id) != k2->depends_on.end());
}

TEST_CASE("GQA reuses one staged K/V tile across query heads in a group") {
    LlamaScheduleConfig cfg;
    cfg.mode = "attention";
    cfg.seq_len = 1;
    cfg.num_q_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "hbm";

    Schedule s = build_attention_schedule(cfg);

    CHECK(count_label_op(s, "KV cache read HBM.kv_cache.L0.kv0.K.page0.block0.range0_1", "dma_load") == 1);
    CHECK(count_label_op(s, "QK L0.S128.kv0.tile0.qh0", "gemm") == 1);
    CHECK(count_label_op(s, "QK L0.S128.kv0.tile0.qh1", "gemm") == 1);
}

TEST_CASE("SRAM KV cache spill policy emits mixed SRAM and HBM page accesses") {
    LlamaScheduleConfig cfg;
    cfg.mode = "attention";
    cfg.seq_len = 4;
    cfg.num_q_heads = 1;
    cfg.num_kv_heads = 1;
    cfg.hidden_dim = 128;
    cfg.intermediate_dim = 256;
    cfg.tile_rows = 1;
    cfg.tile_cols = 1;
    cfg.max_seq_len = 4;
    cfg.dtype_bytes = 2;
    cfg.sram_kv_capacity_kb = 1;
    cfg.kv_cache_enabled = true;
    cfg.kv_cache_location = "sram";
    cfg.kv_cache_eviction_policy = "spill_to_hbm";

    Schedule s = build_attention_schedule(cfg);

    REQUIRE(has_cache_label(s, "dma_store", "KV cache write HBM.kv_cache"));
    REQUIRE(has_cache_label(s, "sram_copy", "KV cache write SRAM.kv_cache"));
    REQUIRE(has_cache_label(s, "dma_load", "KV cache read HBM.kv_cache"));
    REQUIRE(has_cache_label(s, "sram_copy", "KV cache read SRAM.kv_cache"));
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
    REQUIRE(pget_int(merge->params, "input_elements") == 128);
    REQUIRE(pget_int(merge->params, "output_elements") == 128);

    const Instruction* feedback = find_label(s, "feed sampled token back");
    REQUIRE(feedback != nullptr);
    REQUIRE(feedback->unit == "vector_core");
    REQUIRE(pget_int(feedback->params, "rows") == 1);
    REQUIRE(pget_int(feedback->params, "cols") == 1);
}
