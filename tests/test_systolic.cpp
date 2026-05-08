#include <doctest/doctest.h>
#include "core/event_engine.h"
#include "config/arch_config.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/systolic_unit.h"
#include <sstream>
#include <string>

using namespace sim;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

// Build a minimal ArchConfig with the given systolic dimensions.
ArchConfig make_cfg(uint32_t rows, uint32_t cols) {
    ArchConfig c;
    c.systolic.rows = rows;
    c.systolic.cols = cols;
    return c;
}

// Register the "gemm" op (mirrors sim_main.cpp).
void add_gemm_op(OpRegistry& reg, const ArchConfig& arch) {
    reg.register_op("gemm", [&arch](const IssueCtx& ctx) {
        UnitId target = ctx.engine.find_unit(ctx.inst.unit);
        REQUIRE(target != INVALID_UNIT);

        GemmShape shape;
        shape.M = static_cast<uint32_t>(pget_int(ctx.inst.params, "M", arch.systolic.rows));
        shape.K = static_cast<uint32_t>(pget_int(ctx.inst.params, "K", arch.systolic.rows));
        shape.N = static_cast<uint32_t>(pget_int(ctx.inst.params, "N", arch.systolic.cols));

        Event e;
        e.type    = EventType::OP_START;
        e.target  = target;
        e.cycle   = ctx.engine.current_cycle();
        e.instr   = ctx.inst.id;
        e.label   = ctx.inst.label;
        e.payload = shape;
        ctx.engine.schedule(std::move(e));
    });
}

// Run a single-instruction GEMM schedule; return the final simulation cycle.
Cycle run_gemm(uint32_t sa_rows, uint32_t sa_cols,
               uint32_t M, uint32_t K, uint32_t N) {
    ArchConfig arch = make_cfg(sa_rows, sa_cols);

    // Build a one-instruction schedule programmatically.
    Schedule sched;
    Instruction inst;
    inst.id   = 0;
    inst.op   = "gemm";
    inst.unit = "systolic";
    inst.params["M"] = static_cast<int64_t>(M);
    inst.params["K"] = static_cast<int64_t>(K);
    inst.params["N"] = static_cast<int64_t>(N);
    sched.instructions.push_back(inst);

    EventEngine engine(arch.clock_ghz);
    std::ostringstream log;

    auto* su = new SystolicUnit("systolic", arch.systolic, nullptr, nullptr, log);
    engine.register_unit(std::unique_ptr<Unit>(su));

    OpRegistry reg;
    add_gemm_op(reg, arch);

    Scheduler scheduler(engine, reg, sched);
    su->set_scheduler(&scheduler);

    scheduler.launch();
    Cycle final_cycle = engine.run();

    CHECK(scheduler.all_done());
    return final_cycle;
}

}  // namespace

// ---------------------------------------------------------------------------
// TEST 1 – compute_latency formula
// ---------------------------------------------------------------------------
TEST_CASE("SystolicUnit::compute_latency formula") {
    SUBCASE("Single tile — array matches GEMM dims exactly") {
        // 128×128 array, 128×128×128 GEMM  → 1 tile
        // per_tile = 128 + 127 + 127 = 382
        SystolicUnit su("sa", make_cfg(128, 128).systolic);
        CHECK(su.compute_latency(128, 128, 128) == 382);
    }

    SUBCASE("Single tile — non-square, fits") {
        // 256×256 array, 256×512×256 GEMM → 1 tile
        // per_tile = 512 + 255 + 255 = 1022
        SystolicUnit su("sa", make_cfg(256, 256).systolic);
        CHECK(su.compute_latency(256, 512, 256) == 1022);
    }

    SUBCASE("Four tiles — M and N each doubled") {
        // 128×128 array, 256×128×256 GEMM → 2×2 = 4 tiles
        // per_tile = 128 + 127 + 127 = 382,  total = 4 × 382 = 1528
        SystolicUnit su("sa", make_cfg(128, 128).systolic);
        CHECK(su.compute_latency(256, 128, 256) == 1528);
    }

    SUBCASE("Partial last tile — M not a multiple of rows") {
        // 128×128 array, 130×128×128 GEMM → ceil(130/128)=2 tiles in M, 1 in N
        // tiles = 2×1 = 2,  per_tile = 382,  total = 764
        SystolicUnit su("sa", make_cfg(128, 128).systolic);
        CHECK(su.compute_latency(130, 128, 128) == 764);
    }

    SUBCASE("Zero dimension returns 0") {
        SystolicUnit su("sa", make_cfg(128, 128).systolic);
        CHECK(su.compute_latency(0,   128, 128) == 0);
        CHECK(su.compute_latency(128,   0, 128) == 0);
        CHECK(su.compute_latency(128, 128,   0) == 0);
    }
}

// ---------------------------------------------------------------------------
// TEST 2 – event flow through the engine
// ---------------------------------------------------------------------------
TEST_CASE("SystolicUnit event flow — OP_START -> OP_DONE") {
    ArchConfig arch = make_cfg(128, 128);

    EventEngine engine(arch.clock_ghz);
    std::ostringstream log;
    SystolicUnit* su = new SystolicUnit("systolic", arch.systolic, nullptr, nullptr, log);
    engine.register_unit(std::unique_ptr<Unit>(su));
    UnitId sa_id = engine.find_unit("systolic");
    REQUIRE(sa_id != INVALID_UNIT);

    // Manually inject OP_START with a 128×128×128 shape.
    Event start;
    start.type    = EventType::OP_START;
    start.target  = sa_id;
    start.cycle   = 0;
    start.instr   = 42;
    start.payload = GemmShape{128, 128, 128};
    engine.schedule(start);

    // Run until OP_DONE fires (no scheduler — unit won't crash without one).
    Cycle done_cycle = engine.run();

    CHECK(done_cycle == 382);  // 128+127+127

    // Log must mention both START and DONE.
    CHECK(log.str().find("GEMM_START") != std::string::npos);
    CHECK(log.str().find("GEMM_DONE")  != std::string::npos);
    CHECK(log.str().find("128x128x128") != std::string::npos);
}

// ---------------------------------------------------------------------------
// TEST 3 – full scheduler integration with gemm op
// ---------------------------------------------------------------------------
TEST_CASE("SystolicUnit scheduler integration — gemm op") {
    SUBCASE("128x128 array, 128x128x128 GEMM (1 tile)") {
        Cycle c = run_gemm(128, 128, 128, 128, 128);
        CHECK(c == 382);
    }

    SUBCASE("128x128 array, 256x256x256 GEMM (4 tiles)") {
        // per_tile=510 (256+127+127), 4 tiles → 2040
        Cycle c = run_gemm(128, 128, 256, 256, 256);
        CHECK(c == 2040);
    }

    SUBCASE("128x128 array, 512x512x512 GEMM (16 tiles)") {
        // per_tile=766 (512+127+127), 16 tiles → 12256
        Cycle c = run_gemm(128, 128, 512, 512, 512);
        CHECK(c == 12256);
    }
}

// ---------------------------------------------------------------------------
// TEST 4 – Target 2: parametric array size scaling
// ---------------------------------------------------------------------------
TEST_CASE("SystolicUnit scales correctly across array sizes") {
    // 512×512×512 GEMM on three different array sizes.
    // Bigger array → fewer tiles → fewer total cycles.

    SUBCASE("256x256 array, 512x512x512 GEMM (4 tiles)") {
        // tiles=2×2=4,  per_tile=512+255+255=1022,  total=4088
        Cycle c = run_gemm(256, 256, 512, 512, 512);
        CHECK(c == 4088);
    }

    SUBCASE("512x512 array, 512x512x512 GEMM (1 tile)") {
        // tiles=1×1=1,  per_tile=512+511+511=1534,  total=1534
        Cycle c = run_gemm(512, 512, 512, 512, 512);
        CHECK(c == 1534);
    }

    SUBCASE("1024x1024 array, 512x512x512 GEMM (1 tile — array oversized)") {
        // M=N=512 < 1024 → still 1 tile in each dim
        // per_tile=512+1023+1023=2558,  total=2558
        // (larger pipeline fill cost even though 1 tile — undersized GEMM)
        Cycle c = run_gemm(1024, 1024, 512, 512, 512);
        CHECK(c == 2558);
    }

    SUBCASE("Larger GEMM benefits from bigger array: 1024x1024x1024") {
        // 256×256: tiles=4×4=16, per=1024+255+255=1534, total=24544
        // 512×512: tiles=2×2=4,  per=1024+511+511=2046, total=8184
        // 1024×1024: tiles=1×1=1, per=1024+1023+1023=3070, total=3070
        Cycle c256  = run_gemm(256,  256,  1024, 1024, 1024);
        Cycle c512  = run_gemm(512,  512,  1024, 1024, 1024);
        Cycle c1024 = run_gemm(1024, 1024, 1024, 1024, 1024);

        CHECK(c256  == 24544);
        CHECK(c512  == 8184);
        CHECK(c1024 == 3070);

        // Bigger array → fewer cycles for same GEMM.
        CHECK(c1024 < c512);
        CHECK(c512  < c256);
    }
}

// ---------------------------------------------------------------------------
// TEST 6 – fill_latency() correctness for both modes
// ---------------------------------------------------------------------------
TEST_CASE("SystolicUnit fill_latency unidirectional vs bidirectional") {
    SUBCASE("128x128 unidirectional") {
        SystolicUnit su("sa", make_cfg(128,128).systolic);
        CHECK(su.fill_latency() == 254);   // 127 + 127
    }
    SUBCASE("128x128 bidirectional") {
        auto cfg = make_cfg(128,128);
        cfg.systolic.bidirectional = true;
        SystolicUnit su("sa", cfg.systolic);
        CHECK(su.fill_latency() == 128);   // ceil(127/2)+ceil(127/2) = 64+64
    }
    SUBCASE("256x256 unidirectional") {
        SystolicUnit su("sa", make_cfg(256,256).systolic);
        CHECK(su.fill_latency() == 510);   // 255 + 255
    }
    SUBCASE("256x256 bidirectional") {
        auto cfg = make_cfg(256,256);
        cfg.systolic.bidirectional = true;
        SystolicUnit su("sa", cfg.systolic);
        CHECK(su.fill_latency() == 256);   // ceil(255/2)+ceil(255/2) = 128+128
    }
    SUBCASE("512x512 bidirectional") {
        auto cfg = make_cfg(512,512);
        cfg.systolic.bidirectional = true;
        SystolicUnit su("sa", cfg.systolic);
        CHECK(su.fill_latency() == 512);   // 256+256
    }
    SUBCASE("1024x1024 bidirectional") {
        auto cfg = make_cfg(1024,1024);
        cfg.systolic.bidirectional = true;
        SystolicUnit su("sa", cfg.systolic);
        CHECK(su.fill_latency() == 1024);  // 512+512
    }
    SUBCASE("Odd dimension — fill rounds up correctly") {
        // 5×5 array: ceil(4/2)+ceil(4/2) = 2+2 = 4  (exact, 4 is even)
        // 7×7 array: ceil(6/2)+ceil(6/2) = 3+3 = 6  (exact)
        auto cfg5 = make_cfg(5,5);  cfg5.systolic.bidirectional = true;
        auto cfg7 = make_cfg(7,7);  cfg7.systolic.bidirectional = true;
        SystolicUnit su5("sa5", cfg5.systolic);
        SystolicUnit su7("sa7", cfg7.systolic);
        CHECK(su5.fill_latency() == 4);
        CHECK(su7.fill_latency() == 6);
    }
}

// ---------------------------------------------------------------------------
// TEST 7 – bidirectional compute_latency values
// ---------------------------------------------------------------------------
TEST_CASE("SystolicUnit::compute_latency bidirectional") {
    SUBCASE("128x128 bidir, 128x128x128 — 1 tile") {
        // per_tile = 128 + 128 = 256  (fill=128)
        auto cfg = make_cfg(128,128);
        cfg.systolic.bidirectional = true;
        SystolicUnit su("sa", cfg.systolic);
        CHECK(su.compute_latency(128, 128, 128) == 256);
    }
    SUBCASE("128x128 bidir, 256x256x256 — 4 tiles") {
        // per_tile = 256 + 128 = 384,  4 tiles → 1536
        auto cfg = make_cfg(128,128);
        cfg.systolic.bidirectional = true;
        SystolicUnit su("sa", cfg.systolic);
        CHECK(su.compute_latency(256, 256, 256) == 1536);
    }
    SUBCASE("256x256 bidir, 256x256x256 — 1 tile") {
        // per_tile = 256 + 256 = 512
        auto cfg = make_cfg(256,256);
        cfg.systolic.bidirectional = true;
        SystolicUnit su("sa", cfg.systolic);
        CHECK(su.compute_latency(256, 256, 256) == 512);
    }
}

// ---------------------------------------------------------------------------
// TEST 8 – speedup: bidir always <= unidir, benefit shrinks as K grows
// ---------------------------------------------------------------------------
TEST_CASE("SystolicUnit bidirectional is always faster than unidirectional") {
    auto uni_cfg  = make_cfg(128,128);
    auto bidir_cfg = make_cfg(128,128);
    bidir_cfg.systolic.bidirectional = true;

    SystolicUnit uni  ("sa_uni",   uni_cfg.systolic);
    SystolicUnit bidir("sa_bidir", bidir_cfg.systolic);

    // Test across several K values — speedup must be in (1, 2]
    for (uint32_t K : {16u, 64u, 128u, 256u, 512u, 1024u, 4096u}) {
        Cycle u = uni.compute_latency(256, K, 256);
        Cycle b = bidir.compute_latency(256, K, 256);

        CHECK(b < u);          // bidir is strictly faster

        double speedup = static_cast<double>(u) / static_cast<double>(b);
        CHECK(speedup > 1.0);
        CHECK(speedup <= 2.0);
    }

    // Speedup decreases monotonically as K grows
    Cycle u_small = uni.compute_latency(128, 16,   128);
    Cycle b_small = bidir.compute_latency(128, 16,  128);
    Cycle u_large = uni.compute_latency(128, 4096, 128);
    Cycle b_large = bidir.compute_latency(128, 4096, 128);

    double speedup_small = static_cast<double>(u_small) / b_small;
    double speedup_large = static_cast<double>(u_large) / b_large;
    CHECK(speedup_small > speedup_large);
}

// ---------------------------------------------------------------------------
// TEST 9 – YAML round-trip: bidirectional flag parsed from config string
// ---------------------------------------------------------------------------
TEST_CASE("SystolicConfig bidirectional parsed from YAML") {
    const std::string yaml_bidir = R"(
clock_ghz: 1.0
systolic:
  rows: 256
  cols: 256
  precision: BF16
  bidirectional: true
)";
    const std::string yaml_unidir = R"(
clock_ghz: 1.0
systolic:
  rows: 256
  cols: 256
  precision: BF16
  bidirectional: false
)";
    const std::string yaml_default = R"(
clock_ghz: 1.0
systolic:
  rows: 256
  cols: 256
)";

    auto cfg_bi  = ArchConfig::from_yaml_string(yaml_bidir);
    auto cfg_uni = ArchConfig::from_yaml_string(yaml_unidir);
    auto cfg_def = ArchConfig::from_yaml_string(yaml_default);

    CHECK(cfg_bi.systolic.bidirectional  == true);
    CHECK(cfg_uni.systolic.bidirectional == false);
    CHECK(cfg_def.systolic.bidirectional == false);  // default = unidir

    // Latency differs between the two
    SystolicUnit su_bi ("sa", cfg_bi.systolic);
    SystolicUnit su_uni("sa", cfg_uni.systolic);
    CHECK(su_bi.compute_latency(256, 256, 256)  < su_uni.compute_latency(256, 256, 256));
    CHECK(su_bi.fill_latency()  == 256);   // bidir 256×256
    CHECK(su_uni.fill_latency() == 510);   // unidir 256×256
}
TEST_CASE("SystolicUnit default shape when no payload set") {
    ArchConfig arch = make_cfg(64, 64);

    EventEngine engine(arch.clock_ghz);
    std::ostringstream log;
    SystolicUnit* su = new SystolicUnit("systolic", arch.systolic, nullptr, nullptr, log);
    engine.register_unit(std::unique_ptr<Unit>(su));
    UnitId sa_id = engine.find_unit("systolic");

    // OP_START with NO GemmShape payload → falls back to cfg.rows×cfg.rows×cfg.cols
    Event start;
    start.type   = EventType::OP_START;
    start.target = sa_id;
    start.cycle  = 0;
    start.instr  = 1;
    // payload intentionally left empty
    engine.schedule(start);

    Cycle done_cycle = engine.run();
    // Latency for 64×64×64 on 64×64 array: 64 + 63 + 63 = 190
    CHECK(done_cycle == 190);
}
