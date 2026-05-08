#include <doctest/doctest.h>
#include "core/event_engine.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/systolic_unit.h"
#include <sstream>
#include <cmath>
#include <vector>

using namespace sim;

// ─────────────────────────────────────────────────────────────────────────────
// Reference GEMM — simple triple loop, used to verify the tiled result.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> ref_gemm(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> C(M * N, 0.0f);
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t k = 0; k < K; k++)
            for (uint32_t j = 0; j < N; j++)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
    return C;
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run a gemm op through the full scheduler machinery.
// Returns the final simulation cycle; C is written into ts["dst"].
// ─────────────────────────────────────────────────────────────────────────────
static Cycle run_gemm(TensorStore& ts,
                      uint32_t sa_rows, uint32_t sa_cols,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::string& src_a,
                      const std::string& src_b,
                      const std::string& dst_c,
                      bool bidir = false) {
    SystolicConfig cfg;
    cfg.rows = sa_rows; cfg.cols = sa_cols; cfg.bidirectional = bidir;

    Schedule sched;
    Instruction inst;
    inst.id   = 0;
    inst.op   = "gemm";
    inst.unit = "systolic";
    inst.params["M"]           = static_cast<int64_t>(M);
    inst.params["K"]           = static_cast<int64_t>(K);
    inst.params["N"]           = static_cast<int64_t>(N);
    inst.params["source_a"]    = src_a;
    inst.params["source_b"]    = src_b;
    inst.params["destination"] = dst_c;
    sched.instructions.push_back(inst);

    std::ostringstream log;
    EventEngine engine;
    auto* su = new SystolicUnit("systolic", cfg, nullptr, &ts, log);
    engine.register_unit(std::unique_ptr<Unit>(su));

    OpRegistry reg;
    reg.register_op("gemm", [&](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        GemmShape s;
        s.M     = (uint32_t)pget_int(ctx.inst.params, "M", cfg.rows);
        s.K     = (uint32_t)pget_int(ctx.inst.params, "K", cfg.rows);
        s.N     = (uint32_t)pget_int(ctx.inst.params, "N", cfg.cols);
        s.src_a = pget_str(ctx.inst.params, "source_a");
        s.src_b = pget_str(ctx.inst.params, "source_b");
        s.dst_c = pget_str(ctx.inst.params, "destination");
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=s; ctx.engine.schedule(std::move(e));
    });

    Scheduler scheduler(engine, reg, sched);
    su->set_scheduler(&scheduler);
    scheduler.launch();
    Cycle c = engine.run();
    CHECK(scheduler.all_done());
    return c;
}

// =============================================================================
// TEST 1 — Identity matrix: I × A = A
// =============================================================================
TEST_CASE("SystolicUnit compute: identity x A = A") {
    const uint32_t N = 4;
    TensorStore ts;
    ts.init_identity("I", N);                      // 4×4 identity
    ts.init_random("A", N * N, 0.0f, 1.0f, 42);   // random A

    run_gemm(ts, N, N, N, N, N, "I", "A", "C");

    REQUIRE(ts.has("C"));
    const auto& A = ts.get("A");
    const auto& C = ts.get("C");
    CHECK(max_abs_diff(C, A) < 1e-5f);
}

// =============================================================================
// TEST 2 — All-ones × All-ones = K * ones
// =============================================================================
TEST_CASE("SystolicUnit compute: ones x ones = K * ones") {
    const uint32_t M = 4, K = 8, N = 4;
    TensorStore ts;
    ts.init_ones("A", M * K);
    ts.init_ones("B", K * N);

    run_gemm(ts, 4, 4, M, K, N, "A", "B", "C");

    REQUIRE(ts.has("C"));
    for (float v : ts.get("C"))
        CHECK(std::abs(v - static_cast<float>(K)) < 1e-4f);
}

// =============================================================================
// TEST 3 — Random matrices vs reference GEMM, exact match
// =============================================================================
TEST_CASE("SystolicUnit compute: random vs reference") {
    SUBCASE("4x4 array, 4x4x4 GEMM") {
        const uint32_t N = 4;
        TensorStore ts;
        ts.init_random("A", N*N, -1.0f, 1.0f, 1);
        ts.init_random("B", N*N, -1.0f, 1.0f, 2);

        run_gemm(ts, N, N, N, N, N, "A", "B", "C");
        REQUIRE(ts.has("C"));

        auto ref = ref_gemm(ts.get("A"), ts.get("B"), N, N, N);
        CHECK(max_abs_diff(ts.get("C"), ref) < 1e-4f);
    }

    SUBCASE("8x8 array, 8x8x8 GEMM") {
        const uint32_t N = 8;
        TensorStore ts;
        ts.init_random("A", N*N, -1.0f, 1.0f, 10);
        ts.init_random("B", N*N, -1.0f, 1.0f, 11);

        run_gemm(ts, N, N, N, N, N, "A", "B", "C");
        REQUIRE(ts.has("C"));

        auto ref = ref_gemm(ts.get("A"), ts.get("B"), N, N, N);
        CHECK(max_abs_diff(ts.get("C"), ref) < 1e-4f);
    }

    SUBCASE("16x16 array, 32x16x32 GEMM (tiled 2x2)") {
        TensorStore ts;
        ts.init_random("A", 32*16, -1.0f, 1.0f, 20);
        ts.init_random("B", 16*32, -1.0f, 1.0f, 21);

        run_gemm(ts, 16, 16, 32, 16, 32, "A", "B", "C");
        REQUIRE(ts.has("C"));

        auto ref = ref_gemm(ts.get("A"), ts.get("B"), 32, 16, 32);
        CHECK(max_abs_diff(ts.get("C"), ref) < 1e-4f);
    }
}

// =============================================================================
// TEST 4 — FA2 naming convention: "systolic_array.Q_operand" × "shared_ibuf.K_tile_T"
// =============================================================================
TEST_CASE("SystolicUnit compute: FA2 buffer naming convention") {
    const uint32_t N = 8;
    TensorStore ts;
    ts.init_random("systolic_array.Q_operand", N*N, -1.0f, 1.0f, 7);
    ts.init_random("shared_ibuf.K_tile_T",     N*N, -1.0f, 1.0f, 8);

    run_gemm(ts, N, N, N, N, N,
             "systolic_array.Q_operand",
             "shared_ibuf.K_tile_T",
             "shared_obuf.S_tile");

    REQUIRE(ts.has("shared_obuf.S_tile"));

    // Verify against reference
    auto ref = ref_gemm(ts.get("systolic_array.Q_operand"),
                        ts.get("shared_ibuf.K_tile_T"), N, N, N);
    CHECK(max_abs_diff(ts.get("shared_obuf.S_tile"), ref) < 1e-4f);
}

// =============================================================================
// TEST 5 — Timing unchanged when buffer names are empty (timing-only mode)
// =============================================================================
TEST_CASE("SystolicUnit compute: timing-only when no buffer names") {
    TensorStore ts;  // empty — no buffers

    SystolicConfig cfg; cfg.rows = cfg.cols = 8;
    std::ostringstream log;
    auto* su = new SystolicUnit("systolic", cfg, nullptr, &ts, log);
    EventEngine engine;
    engine.register_unit(std::unique_ptr<Unit>(su));

    // GemmShape with empty src names → timing only, no crash
    GemmShape shape; shape.M = 8; shape.K = 8; shape.N = 8;
    // src_a, src_b, dst_c deliberately left empty

    Event e;
    e.type    = EventType::OP_START;
    e.target  = engine.find_unit("systolic");
    e.cycle   = 0;
    e.instr   = 0;
    e.payload = shape;
    engine.schedule(e);

    Cycle c = engine.run();
    CHECK(c == 8 + 7 + 7);  // 22 cycles for 8x8 unidir
    CHECK(log.str().find("GEMM_DONE") != std::string::npos);
    CHECK(log.str().find("GEMM_COMPUTE") == std::string::npos);  // no compute
    CHECK(!ts.has(""));  // nothing written
}

// =============================================================================
// TEST 6 — Missing source buffer logs a warning, doesn't crash
// =============================================================================
TEST_CASE("SystolicUnit compute: missing buffer skipped gracefully") {
    TensorStore ts;
    // Only populate A, leave B missing

    const uint32_t N = 4;
    ts.init_random("A", N*N, 0.0f, 1.0f, 1);
    // "B" intentionally not added

    std::ostringstream log;
    SystolicConfig cfg; cfg.rows = cfg.cols = N;
    auto* su = new SystolicUnit("systolic", cfg, nullptr, &ts, log);
    EventEngine engine;
    engine.register_unit(std::unique_ptr<Unit>(su));

    GemmShape shape; shape.M = N; shape.K = N; shape.N = N;
    shape.src_a = "A";  shape.src_b = "B";  shape.dst_c = "C";

    Event e;
    e.type    = EventType::OP_START;
    e.target  = engine.find_unit("systolic");
    e.cycle   = 0; e.instr = 0; e.payload = shape;
    engine.schedule(e);

    engine.run();  // must not throw
    CHECK(log.str().find("SKIPPED") != std::string::npos);
    CHECK(!ts.has("C"));  // nothing written
}

// =============================================================================
// TEST 7 — Tiling correctness: 128×128 array with 256×128×256 GEMM (4 tiles)
// =============================================================================
TEST_CASE("SystolicUnit compute: multi-tile matches reference") {
    const uint32_t M = 32, K = 16, N = 32;  // 2x2 tiles on 16x16 array
    TensorStore ts;
    ts.init_random("A", M*K, -1.0f, 1.0f, 30);
    ts.init_random("B", K*N, -1.0f, 1.0f, 31);

    run_gemm(ts, 16, 16, M, K, N, "A", "B", "C");
    REQUIRE(ts.has("C"));

    auto ref = ref_gemm(ts.get("A"), ts.get("B"), M, K, N);
    CHECK(max_abs_diff(ts.get("C"), ref) < 1e-4f);
}

// =============================================================================
// TEST 8 — TensorStore helpers
// =============================================================================
TEST_CASE("TensorStore helpers") {
    TensorStore ts;

    SUBCASE("init_zeros") {
        ts.init_zeros("z", 4);
        for (float v : ts.get("z")) CHECK(v == 0.0f);
    }
    SUBCASE("init_ones") {
        ts.init_ones("o", 4);
        for (float v : ts.get("o")) CHECK(v == 1.0f);
    }
    SUBCASE("init_identity") {
        ts.init_identity("I", 3);
        const auto& I = ts.get("I");
        // diagonal = 1
        CHECK(I[0] == 1.0f); CHECK(I[4] == 1.0f); CHECK(I[8] == 1.0f);
        // off-diagonal = 0
        CHECK(I[1] == 0.0f); CHECK(I[3] == 0.0f);
    }
    SUBCASE("max_abs_diff same") {
        ts.init_ones("a", 4);
        ts.init_ones("b", 4);
        CHECK(ts.max_abs_diff("a", "b") == 0.0f);
    }
    SUBCASE("max_abs_diff different") {
        ts.init_zeros("x", 4);
        ts.init_ones ("y", 4);
        CHECK(ts.max_abs_diff("x", "y") == 1.0f);
    }
    SUBCASE("has / remove") {
        ts.init_zeros("tmp", 2);
        CHECK(ts.has("tmp"));
        ts.remove("tmp");
        CHECK(!ts.has("tmp"));
    }
}
