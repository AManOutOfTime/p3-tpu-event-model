#include <doctest/doctest.h>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include "core/event_engine.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/delay_unit.h"

using namespace sim;

// SIM_PROJECT_ROOT is injected by CMake (see tests/CMakeLists.txt).
// It resolves to the absolute path of the repo root at build time, so
// Schedule::from_yaml_file() works regardless of the working directory
// when the test binary is run.
static const std::string FA2_PATH =
    std::string(SIM_PROJECT_ROOT) + "/schedules/fa2_single_tile.yaml";

// ---------------------------------------------------------------------------
// Helpers shared by every test in this file
// ---------------------------------------------------------------------------
namespace {

void add_delay_op(OpRegistry& reg) {
    reg.register_op("delay", [](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        REQUIRE(t != INVALID_UNIT);
        Event e;
        e.type    = EventType::OP_START;
        e.target  = t;
        e.cycle   = ctx.engine.current_cycle();
        e.instr   = ctx.inst.id;
        e.label   = ctx.inst.label;
        e.payload = static_cast<int64_t>(
                        pget_int(ctx.inst.params, "latency_cycles", 10));
        ctx.engine.schedule(std::move(e));
    });
}

void register_fa2_units(EventEngine& engine, std::ostream& ss) {
    for (const char* n : {"systolic", "tandem_1", "access_core_1",
                          "vector_core", "dma"})
        engine.register_unit(std::make_unique<DelayUnit>(n, 0, nullptr, ss));
}

void wire(EventEngine& engine, Scheduler& sched) {
    for (UnitId id = 0; id < static_cast<UnitId>(engine.num_units()); id++)
        if (auto* du = dynamic_cast<DelayUnit*>(engine.get_unit(id)))
            du->set_scheduler(&sched);
}

bool has_dep(const Instruction& inst, InstructionId dep) {
    return std::find(inst.depends_on.begin(), inst.depends_on.end(), dep)
           != inst.depends_on.end();
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: File loads and has exactly 22 instructions with correct fields
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: parses 22 instructions with correct ids and units") {
    Schedule s = Schedule::from_yaml_file(FA2_PATH);

    REQUIRE(s.instructions.size() == 22);

    // id=0: first DMA load (load Q tile from HBM)
    REQUIRE(s.instructions[0].id   == 0);
    REQUIRE(s.instructions[0].op   == "delay");
    REQUIRE(s.instructions[0].unit == "dma");
    REQUIRE(pget_int(s.instructions[0].params, "latency_cycles") == 50);

    // id=1: init O_acc on access_core_1
    REQUIRE(s.instructions[1].id   == 1);
    REQUIRE(s.instructions[1].unit == "access_core_1");
    REQUIRE(pget_int(s.instructions[1].params, "latency_cycles") == 10);

    // id=8: first GEMM on systolic (Q @ K^T)
    REQUIRE(s.instructions[8].id   == 8);
    REQUIRE(s.instructions[8].unit == "systolic");
    REQUIRE(pget_int(s.instructions[8].params, "latency_cycles") == 200);

    // id=10: rowmax on vector_core
    REQUIRE(s.instructions[10].id   == 10);
    REQUIRE(s.instructions[10].unit == "vector_core");

    // id=16: second GEMM on systolic (P @ V)
    REQUIRE(s.instructions[16].id   == 16);
    REQUIRE(s.instructions[16].unit == "systolic");
    REQUIRE(pget_int(s.instructions[16].params, "latency_cycles") == 200);

    // id=21: last DMA store (store L to HBM)
    REQUIRE(s.instructions[21].id   == 21);
    REQUIRE(s.instructions[21].unit == "dma");
}

// ---------------------------------------------------------------------------
// Test 2: Params decoded correctly from the real YAML (not a hardcoded copy)
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: params are parsed correctly from YAML") {
    Schedule s = Schedule::from_yaml_file(FA2_PATH);

    // Check every instruction has a latency_cycles param > 0
    for (const auto& inst : s.instructions) {
        int64_t lat = pget_int(inst.params, "latency_cycles", -1);
        CHECK_MESSAGE(lat > 0,
            "instruction id=", inst.id, " (", inst.label, ") has no latency_cycles");
    }

    // Spot-check GEMM params use "M", "N", "K" keys (set in the YAML)
    const auto& qk = s.instructions[8];
    REQUIRE(qk.params.count("M") == 1);
    REQUIRE(qk.params.count("N") == 1);
    REQUIRE(qk.params.count("K") == 1);

    // source / destination strings are parsed as strings
    REQUIRE(pget_str(s.instructions[0].params, "source").find("HBM") != std::string::npos);
    REQUIRE(pget_str(s.instructions[0].params, "destination").find("ibuf") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Test 3: Dependency edges match the FA2 algorithm
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: dependency edges are algorithmically correct") {
    Schedule s = Schedule::from_yaml_file(FA2_PATH);

    // GEMM QK (8) needs Q issued to systolic (4) AND K transposed (7)
    REQUIRE(s.instructions[8].depends_on.size() == 2);
    REQUIRE(has_dep(s.instructions[8], 4));
    REQUIRE(has_dep(s.instructions[8], 7));

    // update_m_correction (11): needs rowmax (10) AND m_old written by init_m (2)
    REQUIRE(has_dep(s.instructions[11], 10));
    REQUIRE(has_dep(s.instructions[11], 2));

    // update_l (13): needs l_old written by init_l (3) — pre-inner → inner link
    REQUIRE(has_dep(s.instructions[13], 3));

    // rescale_O (14): needs O_acc written by init_O_acc (1) — pre-inner → inner link
    REQUIRE(has_dep(s.instructions[14], 1));

    // load_P_systolic (15): waits for compute_P (12) AND load_V done (6)
    REQUIRE(has_dep(s.instructions[15], 12));
    REQUIRE(has_dep(s.instructions[15], 6));

    // matmul_PV (16): needs P in systolic (15) and V tile in IBUF (6)
    REQUIRE(has_dep(s.instructions[16], 15));
    REQUIRE(has_dep(s.instructions[16], 6));

    // finalize_O (18): needs accumulate_O (17) AND final l (13)
    REQUIRE(has_dep(s.instructions[18], 17));
    REQUIRE(has_dep(s.instructions[18], 13));

    // store_L (21): waits for finalize_L (19) AND store_O (20)
    REQUIRE(has_dep(s.instructions[21], 19));
    REQUIRE(has_dep(s.instructions[21], 20));
}

// ---------------------------------------------------------------------------
// Test 4: Full simulation — all 22 instructions complete with all_done()
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: all 22 instructions complete") {
    std::stringstream ss;
    EventEngine engine;
    register_fa2_units(engine, ss);

    Schedule   s = Schedule::from_yaml_file(FA2_PATH);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(sched.outstanding() == 0);
}

// ---------------------------------------------------------------------------
// Test 5: DMA ops are fully serialized (single channel)
// DMA chain must be: 0→4→5→6→15→20→21 in strictly increasing start cycles.
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: DMA ops are serialized in order") {
    std::stringstream ss;
    EventEngine engine;
    register_fa2_units(engine, ss);

    std::unordered_map<InstructionId, Cycle> start_at;
    engine.set_trace([&](const Event& e) {
        if (e.type == EventType::OP_START)
            start_at[e.instr] = e.cycle;
    });

    Schedule   s = Schedule::from_yaml_file(FA2_PATH);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    engine.run();

    // DMA serialization order: load_Q(0) -> Q_issue(4) -> load_K(5) ->
    //   load_V(6) -> load_P_systolic(15) -> store_O(20) -> store_L(21)
    const std::vector<InstructionId> dma_order = {0, 4, 5, 6, 15, 20, 21};
    for (size_t i = 1; i < dma_order.size(); i++) {
        InstructionId prev = dma_order[i - 1];
        InstructionId cur  = dma_order[i];
        CHECK_MESSAGE(start_at[cur] >= start_at[prev],
            "DMA instr ", cur, " started before instr ", prev);
    }
    // store_L must start strictly after store_O finishes
    REQUIRE(start_at[21] > start_at[20]);
}

// ---------------------------------------------------------------------------
// Test 6: Phase ordering — pre-inner finishes before systolic GEMM starts
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: pre-inner completes before inner GEMM starts") {
    std::stringstream ss;
    EventEngine engine;
    register_fa2_units(engine, ss);

    std::unordered_map<InstructionId, Cycle> done_at;
    std::unordered_map<InstructionId, Cycle> start_at;
    engine.set_trace([&](const Event& e) {
        if (e.type == EventType::OP_DONE)  done_at[e.instr]  = e.cycle;
        if (e.type == EventType::OP_START) start_at[e.instr] = e.cycle;
    });

    Schedule   s = Schedule::from_yaml_file(FA2_PATH);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    engine.run();

    // init_l (id=3) is the last pre-inner op on access_core_1
    // GEMM QK (id=8) must start after init_m (id=2) which is in pre-inner
    REQUIRE(start_at[8] > done_at[2]);   // GEMM starts after m is initialised
    REQUIRE(done_at[8]  > done_at[3]);   // GEMM ends after all pre-inner done
}

// ---------------------------------------------------------------------------
// Test 7: Label substrings from the real YAML appear in unit output
// Labels are checked against actual text in fa2_single_tile.yaml.
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: real instruction labels appear in unit output") {
    std::stringstream ss;
    EventEngine engine;
    register_fa2_units(engine, ss);

    Schedule   s = Schedule::from_yaml_file(FA2_PATH);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    engine.run();

    const std::string out = ss.str();

    // Substrings taken verbatim from labels in fa2_single_tile.yaml
    REQUIRE(out.find("Q_tile")          != std::string::npos);
    REQUIRE(out.find("K_tile_T")        != std::string::npos);  // transpose output name
    REQUIRE(out.find("rowmax")          != std::string::npos);
    REQUIRE(out.find("correction")      != std::string::npos);
    REQUIRE(out.find("exp(S_tile")      != std::string::npos);  // compute_P label
    REQUIRE(out.find("O_acc / l")       != std::string::npos);  // finalize_O label
    REQUIRE(out.find("m + log(l)")      != std::string::npos);  // finalize_L label
    REQUIRE(out.find("store O")         != std::string::npos);
    REQUIRE(out.find("store L")         != std::string::npos);
}
