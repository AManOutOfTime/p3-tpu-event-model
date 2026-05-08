#include <doctest/doctest.h>
#include "schedule/tiler.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include "core/event_engine.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/systolic_unit.h"
#include "units/dma_unit.h"
#include <cmath>
#include <sstream>

using namespace sim;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> ref_gemm(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> C(M * N, 0.f);
    for (uint32_t i=0;i<M;i++)
        for (uint32_t k=0;k<K;k++)
            for (uint32_t j=0;j<N;j++)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
    return C;
}

static float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float mx=0;
    for (size_t i=0;i<a.size();i++) mx=std::max(mx,std::abs(a[i]-b[i]));
    return mx;
}

static ArchConfig make_arch(uint32_t r, uint32_t c) {
    ArchConfig a;
    a.systolic.rows = r; a.systolic.cols = c;
    a.sram.banking_factor = 8;
    return a;
}

// Run a TileDecomposition through the full simulator and assemble output.
static void run_decomp(TileDecomposition& td, TensorStore& ts,
                       const ArchConfig& arch) {
    Schedule sched;
    sched.instructions = td.instructions;

    std::ostringstream log;
    EventEngine engine;

    auto* su = new SystolicUnit("systolic", arch.systolic, nullptr, &ts, log);
    auto* du = new DmaUnit("dma", arch, &ts, nullptr, log);
    engine.register_unit(std::unique_ptr<Unit>(su));
    engine.register_unit(std::unique_ptr<Unit>(du));

    OpRegistry reg;

    // gemm op
    reg.register_op("gemm", [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        GemmShape s;
        s.M     = (uint32_t)pget_int(ctx.inst.params,"M",arch.systolic.rows);
        s.K     = (uint32_t)pget_int(ctx.inst.params,"K",arch.systolic.rows);
        s.N     = (uint32_t)pget_int(ctx.inst.params,"N",arch.systolic.cols);
        s.src_a = pget_str(ctx.inst.params,"source_a");
        s.src_b = pget_str(ctx.inst.params,"source_b");
        s.dst_c = pget_str(ctx.inst.params,"destination");
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=s; ctx.engine.schedule(std::move(e));
    });

    // stage op (IBUF → array register)
    reg.register_op("stage", [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        DmaTransfer xfer;
        xfer.bytes   = (uint64_t)pget_int(ctx.inst.params,"bytes",0);
        xfer.on_chip = pget_bool(ctx.inst.params,"on_chip",true);
        xfer.src_buf = pget_str(ctx.inst.params,"src_buf");
        xfer.dst_buf = pget_str(ctx.inst.params,"dst_buf");
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=xfer; ctx.engine.schedule(std::move(e));
    });

    Scheduler scheduler(engine, reg, sched);
    su->set_scheduler(&scheduler);
    du->set_scheduler(&scheduler);
    scheduler.launch();
    engine.run();
    CHECK(scheduler.all_done());

    Tiler::assemble_output(td, ts);
}

// =============================================================================
// TEST 1 — Single tile: fits array exactly, one STAGE + one GEMM
// =============================================================================
TEST_CASE("Tiler: single tile — fits array exactly") {
    ArchConfig arch = make_arch(8, 8);
    TensorStore ts;
    WorkloadGemm wl; wl.M=8; wl.K=8; wl.N=8;
    wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
    wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

    auto td = Tiler::decompose(wl, arch, ts);

    // Exactly 1 STAGE + 1 GEMM
    CHECK(td.tiles_m == 1);
    CHECK(td.tiles_n == 1);
    CHECK(td.tiles.size() == 1);
    // instructions: 1 stage + 1 gemm = 2
    CHECK(td.instructions.size() == 2);
    CHECK(td.instructions[0].op == "stage");
    CHECK(td.instructions[1].op == "gemm");
}

// =============================================================================
// TEST 2 — Instruction count for tiled case
// =============================================================================
TEST_CASE("Tiler: instruction count") {
    // 2×2 tiles → 2 STAGE + 4 GEMM = 6 instructions
    ArchConfig arch = make_arch(8, 8);
    TensorStore ts;
    WorkloadGemm wl; wl.M=16; wl.K=8; wl.N=16;
    wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
    wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

    auto td = Tiler::decompose(wl, arch, ts);

    CHECK(td.tiles_m == 2);
    CHECK(td.tiles_n == 2);
    // 2 STAGE (one per Q row sub-tile) + 4 GEMM (2×2) = 6
    CHECK(td.instructions.size() == 6);

    // Check alternating pattern: STAGE GEMM GEMM STAGE GEMM GEMM
    CHECK(td.instructions[0].op == "stage");
    CHECK(td.instructions[1].op == "gemm");
    CHECK(td.instructions[2].op == "gemm");
    CHECK(td.instructions[3].op == "stage");
    CHECK(td.instructions[4].op == "gemm");
    CHECK(td.instructions[5].op == "gemm");
}

// =============================================================================
// TEST 3 — Buffer naming convention
// =============================================================================
TEST_CASE("Tiler: buffer naming") {
    ArchConfig arch = make_arch(8, 8);
    TensorStore ts;
    WorkloadGemm wl; wl.M=16; wl.K=8; wl.N=16;
    wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
    wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

    auto td = Tiler::decompose(wl, arch, ts);

    // STAGE 0: copies Q_sub_r0 → systolic_array.Q_operand
    CHECK(pget_str(td.instructions[0].params,"src_buf")
          == "shared_ibuf.Q_sub_r0");
    CHECK(pget_str(td.instructions[0].params,"dst_buf")
          == "systolic_array.Q_operand");

    // GEMM (0,0): reads Q_operand and KT_sub_c0
    CHECK(pget_str(td.instructions[1].params,"source_a")
          == "systolic_array.Q_operand");
    CHECK(pget_str(td.instructions[1].params,"source_b")
          == "shared_ibuf.KT_sub_c0");
    CHECK(pget_str(td.instructions[1].params,"destination")
          == "shared_obuf.S_sub_r0_c0");

    // GEMM (0,1): same Q_operand, different KT sub-tile
    CHECK(pget_str(td.instructions[2].params,"source_b")
          == "shared_ibuf.KT_sub_c1");
    CHECK(pget_str(td.instructions[2].params,"destination")
          == "shared_obuf.S_sub_r0_c1");

    // STAGE 1: loads next Q sub-tile
    CHECK(pget_str(td.instructions[3].params,"src_buf")
          == "shared_ibuf.Q_sub_r1");

    // IBUF slices exist in TensorStore
    CHECK(ts.has("shared_ibuf.Q_sub_r0"));
    CHECK(ts.has("shared_ibuf.Q_sub_r1"));
    CHECK(ts.has("shared_ibuf.KT_sub_c0"));
    CHECK(ts.has("shared_ibuf.KT_sub_c1"));
}

// =============================================================================
// TEST 4 — Q-stationary: KT sub-tiles sliced only once (reused across Q rows)
// =============================================================================
TEST_CASE("Tiler: KT sub-tiles sliced once, reused across Q sub-tiles") {
    ArchConfig arch = make_arch(8, 8);
    TensorStore ts;
    WorkloadGemm wl; wl.M=24; wl.K=8; wl.N=16;
    wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
    wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

    // tiles_m=3, tiles_n=2 → 3 STAGE + 6 GEMM = 9 instructions
    auto td = Tiler::decompose(wl, arch, ts);
    CHECK(td.tiles_m == 3);
    CHECK(td.tiles_n == 2);

    // All GEMM[*, j=0] reference the same KT sub-tile buffer
    // (not three separate copies)
    CHECK(pget_str(td.instructions[1].params,"source_b")
          == "shared_ibuf.KT_sub_c0");
    CHECK(pget_str(td.instructions[4].params,"source_b")
          == "shared_ibuf.KT_sub_c0");
    CHECK(pget_str(td.instructions[7].params,"source_b")
          == "shared_ibuf.KT_sub_c0");
}

// =============================================================================
// TEST 5 — Dependency chain: STAGE serializes GEMMs correctly
// =============================================================================
TEST_CASE("Tiler: dependency chain") {
    ArchConfig arch = make_arch(8, 8);
    TensorStore ts;
    WorkloadGemm wl; wl.M=16; wl.K=8; wl.N=16;
    wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
    wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

    auto td = Tiler::decompose(wl, arch, ts);
    // id=0 STAGE_0, id=1 GEMM_0_0, id=2 GEMM_0_1, id=3 STAGE_1, id=4 GEMM_1_0, id=5 GEMM_1_1

    // STAGE_0 has no dependencies (first instruction, no prev GEMM)
    CHECK(td.instructions[0].depends_on.empty());

    // GEMM_0_0 depends only on STAGE_0 (no prev GEMM yet)
    CHECK(td.instructions[1].depends_on.size() == 1);
    CHECK(td.instructions[1].depends_on[0] == 0);  // STAGE_0

    // GEMM_0_1 depends on STAGE_0 AND GEMM_0_0 (array serialization)
    CHECK(td.instructions[2].depends_on.size() == 2);
    CHECK(td.instructions[2].depends_on[0] == 0);  // STAGE_0
    CHECK(td.instructions[2].depends_on[1] == 1);  // GEMM_0_0

    // STAGE_1 depends on GEMM_0_1 (array must drain before re-staging)
    CHECK(td.instructions[3].depends_on.size() == 1);
    CHECK(td.instructions[3].depends_on[0] == 2);  // GEMM_0_1

    // GEMM_1_0 depends on STAGE_1 AND GEMM_0_1
    CHECK(td.instructions[4].depends_on[0] == 3);  // STAGE_1
    CHECK(td.instructions[4].depends_on[1] == 2);  // GEMM_0_1
}

// =============================================================================
// TEST 6 — End-to-end correctness: assembled S_tile matches reference GEMM
// =============================================================================
TEST_CASE("Tiler: end-to-end correctness") {
    SUBCASE("1x1 tiles — single execution") {
        ArchConfig arch = make_arch(8, 8);
        TensorStore ts;
        WorkloadGemm wl; wl.M=8; wl.K=8; wl.N=8;
        wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
        wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

        ts.init_random("shared_ibuf.Q_tile",  8*8, -1.f, 1.f, 1);
        ts.init_random("shared_ibuf.KT_tile", 8*8, -1.f, 1.f, 2);

        auto td = Tiler::decompose(wl, arch, ts);
        run_decomp(td, ts, arch);

        REQUIRE(ts.has("shared_obuf.S_tile"));
        auto ref = ref_gemm(ts.get("shared_ibuf.Q_tile"),
                            ts.get("shared_ibuf.KT_tile"), 8, 8, 8);
        CHECK(max_diff(ts.get("shared_obuf.S_tile"), ref) < 1e-4f);
    }

    SUBCASE("2x2 tiles — Q_tile larger than array") {
        ArchConfig arch = make_arch(8, 8);
        TensorStore ts;
        WorkloadGemm wl; wl.M=16; wl.K=8; wl.N=16;
        wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
        wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

        ts.init_random("shared_ibuf.Q_tile",  16*8,  -1.f, 1.f, 10);
        ts.init_random("shared_ibuf.KT_tile",  8*16, -1.f, 1.f, 11);

        auto td = Tiler::decompose(wl, arch, ts);
        CHECK(td.instructions.size() == 6);  // 2 STAGE + 4 GEMM
        run_decomp(td, ts, arch);

        REQUIRE(ts.has("shared_obuf.S_tile"));
        auto ref = ref_gemm(ts.get("shared_ibuf.Q_tile"),
                            ts.get("shared_ibuf.KT_tile"), 16, 8, 16);
        CHECK(max_diff(ts.get("shared_obuf.S_tile"), ref) < 1e-4f);
    }

    SUBCASE("FA2: Br=256 d_head=128 Bc=256 on 128x128 array") {
        ArchConfig arch = make_arch(128, 128);
        TensorStore ts;
        WorkloadGemm wl; wl.M=256; wl.K=128; wl.N=256;
        wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
        wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

        ts.init_random("shared_ibuf.Q_tile",  256*128, -1.f, 1.f, 20);
        ts.init_random("shared_ibuf.KT_tile", 128*256, -1.f, 1.f, 21);

        auto td = Tiler::decompose(wl, arch, ts);
        // 2 STAGE + 4 GEMM = 6 instructions
        CHECK(td.tiles_m == 2);
        CHECK(td.tiles_n == 2);
        CHECK(td.instructions.size() == 6);

        run_decomp(td, ts, arch);
        REQUIRE(ts.has("shared_obuf.S_tile"));

        auto ref = ref_gemm(ts.get("shared_ibuf.Q_tile"),
                            ts.get("shared_ibuf.KT_tile"), 256, 128, 256);
        CHECK(max_diff(ts.get("shared_obuf.S_tile"), ref) < 1e-3f);
    }
}

// =============================================================================
// TEST 7 — Partial tiles (M and N not multiples of array size)
// =============================================================================
TEST_CASE("Tiler: partial tiles") {
    ArchConfig arch = make_arch(8, 8);
    TensorStore ts;
    // M=12 → tiles_m=2 (tile0:8rows, tile1:4rows)
    // N=10 → tiles_n=2 (tile0:8cols, tile1:2cols)
    WorkloadGemm wl; wl.M=12; wl.K=8; wl.N=10;
    wl.src_a="shared_ibuf.Q_tile"; wl.src_b="shared_ibuf.KT_tile";
    wl.dst_c="shared_obuf.S_tile"; wl.fill="random";

    ts.init_random("shared_ibuf.Q_tile",  12*8,  -1.f, 1.f, 30);
    ts.init_random("shared_ibuf.KT_tile",  8*10, -1.f, 1.f, 31);

    auto td = Tiler::decompose(wl, arch, ts);

    // Check partial tile shapes
    CHECK(td.tiles[0].tm == 8); CHECK(td.tiles[0].tn == 8);  // full
    CHECK(td.tiles[1].tm == 8); CHECK(td.tiles[1].tn == 2);  // partial N
    CHECK(td.tiles[2].tm == 4); CHECK(td.tiles[2].tn == 8);  // partial M
    CHECK(td.tiles[3].tm == 4); CHECK(td.tiles[3].tn == 2);  // partial both

    run_decomp(td, ts, arch);
    REQUIRE(ts.has("shared_obuf.S_tile"));

    auto ref = ref_gemm(ts.get("shared_ibuf.Q_tile"),
                        ts.get("shared_ibuf.KT_tile"), 12, 8, 10);
    CHECK(max_diff(ts.get("shared_obuf.S_tile"), ref) < 1e-4f);
}

// =============================================================================
// TEST 8 — YAML parsing with FA2 field names
// =============================================================================
TEST_CASE("Tiler: YAML parsing FA2 field names") {
    const std::string yaml = R"(
workload:
  Br: 256
  d_head: 128
  Bc: 256
  src_a: "shared_ibuf.Q_tile"
  src_b: "shared_ibuf.KT_tile"
  dst_c: "shared_obuf.S_tile"
  fill: random
)";
    WorkloadGemm wl = Tiler::from_yaml_string(yaml);
    CHECK(wl.M == 256);
    CHECK(wl.K == 128);
    CHECK(wl.N == 256);
    CHECK(wl.src_a == "shared_ibuf.Q_tile");
    CHECK(wl.src_b == "shared_ibuf.KT_tile");
    CHECK(wl.dst_c == "shared_obuf.S_tile");
}

// =============================================================================
// TEST 9 — Stage latency: on-chip SRAM read, no HBM penalty
// =============================================================================
TEST_CASE("DmaUnit: stage_latency uses banking_factor, not HBM") {
    ArchConfig arch = make_arch(128, 128);
    arch.sram.banking_factor = 8;
    arch.hbm.latency_cycles  = 200;

    TensorStore ts;
    DmaUnit du("dma", arch, &ts);

    // 128x128 BF16 = 32768 bytes
    // stage_latency = ceil(32768 / 8) = 4096  (no HBM penalty)
    CHECK(du.stage_latency(32768) == 4096);

    // transfer_latency includes HBM penalty
    // ceil(32768 / 2000) = 17  → 200 + 17 = 217
    CHECK(du.transfer_latency(32768) == 217);

    // stage is always cheaper than HBM transfer for any byte count
    CHECK(du.stage_latency(32768) != du.transfer_latency(32768));
}
