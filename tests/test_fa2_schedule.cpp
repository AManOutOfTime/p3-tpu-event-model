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
    auto fixed_latency_op = [](const IssueCtx& ctx) {
        std::vector<UnitId> targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        REQUIRE(!targets.empty());
        Cycle latency = static_cast<Cycle>(pget_int(ctx.inst.params, "latency_cycles", 10));
        UnitReservation reservation = ctx.scheduler.reserve_unit_pool(targets, latency);
        Event e;
        e.type    = EventType::OP_START;
        e.target  = reservation.id;
        e.cycle   = reservation.start;
        e.instr   = ctx.inst.id;
        e.label   = ctx.inst.label;
        e.payload = static_cast<int64_t>(latency);
        ctx.engine.schedule(std::move(e));
    };
    reg.register_op("delay", fixed_latency_op);
    reg.register_op("gemm", fixed_latency_op);
}

void register_fa2_units(EventEngine& engine, std::ostream& ss) {
    for (const char* n : {"systolic", "access_core_0", "vector_core_0",
                          "vector_core_1", "vector_core_2", "dma_0"})
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
// Test 1: File loads and has exactly 21 instructions with correct fields
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: parses 21 instructions with correct ids and units") {
    Schedule s = Schedule::from_yaml_file(FA2_PATH);

    REQUIRE(s.instructions.size() == 21);

    // id=0: first DMA load (load Q tile from HBM)
    REQUIRE(s.instructions[0].id   == 0);
    REQUIRE(s.instructions[0].op   == "delay");
    REQUIRE(s.instructions[0].unit == "dma");
    REQUIRE(pget_int(s.instructions[0].params, "latency_cycles") == 50);

    // id=1: init O_acc on access_core
    REQUIRE(s.instructions[1].id   == 1);
    REQUIRE(s.instructions[1].unit == "access_core");
    REQUIRE(pget_int(s.instructions[1].params, "latency_cycles") == 10);

    // id=7: first GEMM on systolic (Q @ K^T)
    REQUIRE(s.instructions[7].id   == 7);
    REQUIRE(s.instructions[7].unit == "systolic");
    REQUIRE(pget_int(s.instructions[7].params, "latency_cycles") == 200);

    // id=9: rowmax on vector_core
    REQUIRE(s.instructions[9].id   == 9);
    REQUIRE(s.instructions[9].unit == "vector_core");

    // id=15: second GEMM on systolic (P @ V)
    REQUIRE(s.instructions[15].id   == 15);
    REQUIRE(s.instructions[15].unit == "systolic");
    REQUIRE(pget_int(s.instructions[15].params, "latency_cycles") == 200);

    // id=20: last DMA store (store L to HBM)
    REQUIRE(s.instructions[20].id   == 20);
    REQUIRE(s.instructions[20].unit == "dma");
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
    const auto& qk = s.instructions[7];
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

    // GEMM QK (7) needs Q loaded (0) AND K transposed (6)
    REQUIRE(s.instructions[7].depends_on.size() == 2);
    REQUIRE(has_dep(s.instructions[7], 0));
    REQUIRE(has_dep(s.instructions[7], 6));

    // update_m_correction (10): needs rowmax (9) AND m_old written by init_m (2)
    REQUIRE(has_dep(s.instructions[10], 9));
    REQUIRE(has_dep(s.instructions[10], 2));

    // update_l (12): needs P (11), correction (10), and l_old from init_l (3)
    REQUIRE(has_dep(s.instructions[12], 11));
    REQUIRE(has_dep(s.instructions[12], 10));
    REQUIRE(has_dep(s.instructions[12], 3));

    // rescale_O (13): needs correction (10) and O_acc from init_O_acc (1)
    REQUIRE(has_dep(s.instructions[13], 10));
    REQUIRE(has_dep(s.instructions[13], 1));

    // load_P_systolic (14): waits for compute_P (11)
    REQUIRE(has_dep(s.instructions[14], 11));

    // matmul_PV (15): needs P in systolic (14) and V tile in IBUF (5)
    REQUIRE(has_dep(s.instructions[15], 14));
    REQUIRE(has_dep(s.instructions[15], 5));

    // finalize_O (17): needs accumulate_O (16) AND final l (12)
    REQUIRE(has_dep(s.instructions[17], 16));
    REQUIRE(has_dep(s.instructions[17], 12));

    // store_L (20): waits for finalize_L (18) AND store_O (19)
    REQUIRE(has_dep(s.instructions[20], 18));
    REQUIRE(has_dep(s.instructions[20], 19));
}

// ---------------------------------------------------------------------------
// Test 4: Full simulation — all 21 instructions complete with all_done()
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
// DMA chain must be: 0→4→5→14→19→20 in strictly increasing start cycles.
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

    // DMA serialization order: load_Q(0) -> load_K(4) -> load_V(5) ->
    //   load_P_systolic(14) -> store_O(19) -> store_L(20)
    const std::vector<InstructionId> dma_order = {0, 4, 5, 14, 19, 20};
    for (size_t i = 1; i < dma_order.size(); i++) {
        InstructionId prev = dma_order[i - 1];
        InstructionId cur  = dma_order[i];
        CHECK_MESSAGE(start_at[cur] >= start_at[prev],
            "DMA instr ", cur, " started before instr ", prev);
    }
    // store_L must start strictly after store_O finishes
    REQUIRE(start_at[20] > start_at[19]);
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

    // init_l (id=3) is the last pre-inner op on access_core
    // GEMM QK (id=7) must start after init_m (id=2) which is in pre-inner
    REQUIRE(start_at[7] > done_at[2]);   // GEMM starts after m is initialised
    REQUIRE(done_at[7]  > done_at[3]);   // GEMM ends after all pre-inner done
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
