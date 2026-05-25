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
    // Register all FA2 ops as delay-ops so the scheduler can drive them
    // without requiring the real hardware units in schedule-level tests.
    for (const char* op : {"delay","gemm","dma_load","dma_store","dma_stage",
                           "init_fill","transpose","scale","rowmax",
                           "update_rowmax","exp_shift","update_rowsum",
                           "accumulate","normalize","logsumexp",
                           "weight_load","sram_read","sram_write"})
        reg.register_op(op, fixed_latency_op);
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
// Test 1: File loads and has exactly 22 instructions with correct fields
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: parses 22 instructions with correct ids and units") {
    Schedule s = Schedule::from_yaml_file(FA2_PATH);

    REQUIRE(s.instructions.size() == 24);

    // id=0: dma_load Q tile from HBM
    REQUIRE(s.instructions[0].id   == 0);
    REQUIRE(s.instructions[0].op   == "dma_load");
    REQUIRE(s.instructions[0].unit == "dma");

    // id=1: init_fill O_acc on access_core
    REQUIRE(s.instructions[1].id   == 1);
    REQUIRE(s.instructions[1].op   == "init_fill");
    REQUIRE(s.instructions[1].unit == "access_core");

    // id=4: dma_stage Q tile into array register
    REQUIRE(s.instructions[4].id   == 4);
    REQUIRE(s.instructions[4].op   == "dma_stage");
    REQUIRE(s.instructions[4].unit == "dma");

    // id=7: transpose K tile on access_core
    REQUIRE(s.instructions[7].id   == 7);
    REQUIRE(s.instructions[7].op   == "transpose");
    REQUIRE(s.instructions[7].unit == "access_core");

    // id=8: weight_load K tile into PE registers (NEW)
    REQUIRE(s.instructions[8].id   == 8);
    REQUIRE(s.instructions[8].op   == "weight_load");
    REQUIRE(s.instructions[8].unit == "systolic");

    // id=9: gemm Q @ K^T on systolic
    REQUIRE(s.instructions[9].id   == 9);
    REQUIRE(s.instructions[9].op   == "gemm");
    REQUIRE(s.instructions[9].unit == "systolic");

    // id=11: rowmax on vector_core
    REQUIRE(s.instructions[11].id   == 11);
    REQUIRE(s.instructions[11].op   == "rowmax");
    REQUIRE(s.instructions[11].unit == "vector_core");

    // id=12: update_rowmax on vector_core
    REQUIRE(s.instructions[12].op == "update_rowmax");

    // id=13: exp_shift on vector_core
    REQUIRE(s.instructions[13].op == "exp_shift");

    // id=14: update_rowsum on vector_core
    REQUIRE(s.instructions[14].op == "update_rowsum");

    // id=17: weight_load V tile into PE registers (NEW)
    REQUIRE(s.instructions[17].op   == "weight_load");
    REQUIRE(s.instructions[17].unit == "systolic");

    // id=18: gemm P @ V on systolic
    REQUIRE(s.instructions[18].id   == 18);
    REQUIRE(s.instructions[18].op   == "gemm");
    REQUIRE(s.instructions[18].unit == "systolic");

    // id=20: normalize on vector_core
    REQUIRE(s.instructions[20].op == "normalize");

    // id=21: logsumexp on vector_core
    REQUIRE(s.instructions[21].op == "logsumexp");

    // id=22: dma_store O tile
    REQUIRE(s.instructions[22].op   == "dma_store");
    REQUIRE(s.instructions[22].unit == "dma");

    // id=23: dma_store L tile
    REQUIRE(s.instructions[23].id   == 23);
    REQUIRE(s.instructions[23].op   == "dma_store");
    REQUIRE(s.instructions[23].unit == "dma");
}

// ---------------------------------------------------------------------------
// Test 2: Params decoded correctly from the real YAML
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: params are parsed correctly from YAML") {
    Schedule s = Schedule::from_yaml_file(FA2_PATH);

    // GEMM params use M/K/N keys with symbolic values (id=9)
    const auto& qk = s.instructions[9];
    REQUIRE(qk.params.count("M") == 1);
    REQUIRE(qk.params.count("K") == 1);
    REQUIRE(qk.params.count("N") == 1);
    REQUIRE(pget_str(qk.params, "source_a") == "systolic_array.Q_operand");
    REQUIRE(pget_str(qk.params, "source_b") == "shared_ibuf.K_tile_T");
    REQUIRE(pget_str(qk.params, "destination") == "shared_obuf.S_tile");

    // weight_load has source/destination params (id=8)
    REQUIRE(s.instructions[8].params.count("source") == 1);
    REQUIRE(s.instructions[8].params.count("destination") == 1);
    REQUIRE(pget_str(s.instructions[8].params,"source") == "shared_ibuf.K_tile_T");

    // DMA ops have source / destination strings
    REQUIRE(pget_str(s.instructions[0].params, "source").find("HBM") != std::string::npos);
    REQUIRE(pget_str(s.instructions[0].params, "destination").find("ibuf") != std::string::npos);

    // dma_stage has source in IBUF and destination in systolic register
    REQUIRE(pget_str(s.instructions[4].params, "source").find("ibuf") != std::string::npos);
    REQUIRE(pget_str(s.instructions[4].params, "destination").find("systolic_array") != std::string::npos);

    // init_fill has init_value param
    REQUIRE(s.instructions[1].params.count("init_value") == 1);
    REQUIRE(s.instructions[2].params.count("init_value") == 1);  // -inf

    // vector ops have rows/cols or length params
    REQUIRE(s.instructions[10].params.count("rows") == 1);  // scale has rows/cols
    REQUIRE(s.instructions[11].params.count("rows") == 1);  // rowmax has rows/cols
    REQUIRE(s.instructions[12].params.count("length") == 1); // update_rowmax has length
}

// ---------------------------------------------------------------------------
// Test 3: Dependency edges match the FA2 algorithm
// ---------------------------------------------------------------------------
TEST_CASE("fa2 file: dependency edges are algorithmically correct") {
    Schedule s = Schedule::from_yaml_file(FA2_PATH);

    // weight_load K (8) needs K transposed (7)
    REQUIRE(has_dep(s.instructions[8], 7));

    // GEMM QK (9) needs Q staged (4) AND weight_load K done (8)
    REQUIRE(s.instructions[9].depends_on.size() == 2);
    REQUIRE(has_dep(s.instructions[9], 4));
    REQUIRE(has_dep(s.instructions[9], 8));

    // update_rowmax (12): needs rowmax (11) AND m_old from init_m (2)
    REQUIRE(has_dep(s.instructions[12], 11));
    REQUIRE(has_dep(s.instructions[12], 2));

    // update_rowsum (14): needs l_old from init_l (3)
    REQUIRE(has_dep(s.instructions[14], 3));

    // scale_O (15): needs O_acc from init_O (1)
    REQUIRE(has_dep(s.instructions[15], 1));

    // dma_stage P (16): waits for exp_shift (13) AND V loaded (6)
    REQUIRE(has_dep(s.instructions[16], 13));
    REQUIRE(has_dep(s.instructions[16], 6));

    // weight_load V (17): needs GEMM S done (9) AND V loaded (6)
    REQUIRE(has_dep(s.instructions[17], 9));
    REQUIRE(has_dep(s.instructions[17], 6));

    // GEMM PV (18): needs P staged (16) AND weight_load V done (17)
    REQUIRE(has_dep(s.instructions[18], 16));
    REQUIRE(has_dep(s.instructions[18], 17));

    // normalize (20): needs accumulate (19) AND final l (14)
    REQUIRE(has_dep(s.instructions[20], 19));
    REQUIRE(has_dep(s.instructions[20], 14));

    // store_L (23): waits for logsumexp (21) AND store_O (22)
    REQUIRE(has_dep(s.instructions[23], 21));
    REQUIRE(has_dep(s.instructions[23], 22));
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

    // init_l (id=3) is the last pre-inner op on access_core
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
