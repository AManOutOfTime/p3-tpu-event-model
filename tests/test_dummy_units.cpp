#include <doctest/doctest.h>
#include "core/event_engine.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/printing_unit.h"
#include "units/delay_unit.h"
#include <sstream>

using namespace sim;

// ---------------------------------------------------------------------------
// Helpers shared by all tests in this file.
// ---------------------------------------------------------------------------
namespace {

// Register the built-in "delay" op that fires a single OP_START on inst.unit.
void add_delay_op(OpRegistry& reg) {
    reg.register_op("delay", [](const IssueCtx& ctx) {
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
    });
}

// Wire a Scheduler pointer into every DelayUnit on the engine.
void wire(EventEngine& eng, Scheduler& sched) {
    for (UnitId id = 0; id < static_cast<UnitId>(eng.num_units()); id++)
        if (auto* du = dynamic_cast<DelayUnit*>(eng.get_unit(id)))
            du->set_scheduler(&sched);
}

}  // namespace

// ---------------------------------------------------------------------------
TEST_CASE("serial chain: total cycles = sum of individual latencies") {
    std::stringstream ss;
    EventEngine engine;
    engine.register_unit(std::make_unique<DelayUnit>("u0", 0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("u1", 0, nullptr, ss));

    const char* yaml = R"(
schedule:
  - id: 0
    op: delay
    unit: u0
    params: { latency_cycles: 100 }
    label: first
  - id: 1
    op: delay
    unit: u1
    params: { latency_cycles: 50 }
    depends_on: [0]
    label: second
)";
    Schedule   s = Schedule::from_yaml_string(yaml);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(final_cycle == 150);  // 100 + 50
}

TEST_CASE("parallel independent ops: completes at max latency") {
    std::stringstream ss;
    EventEngine engine;
    engine.register_unit(std::make_unique<DelayUnit>("u0", 0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("u1", 0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("u2", 0, nullptr, ss));

    const char* yaml = R"(
schedule:
  - id: 0
    op: delay
    unit: u0
    params: { latency_cycles: 100 }
  - id: 1
    op: delay
    unit: u1
    params: { latency_cycles: 50 }
  - id: 2
    op: delay
    unit: u2
    params: { latency_cycles: 75 }
)";
    Schedule   s = Schedule::from_yaml_string(yaml);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(final_cycle == 100);  // max of {100, 50, 75}
}

TEST_CASE("independent ops on one unit are serialized by scheduler reservation") {
    std::stringstream ss;
    EventEngine engine;
    engine.register_unit(std::make_unique<DelayUnit>("vector_core", 0, nullptr, ss));

    std::vector<Cycle> starts;
    engine.set_trace([&](const Event& e) {
        if (e.type == EventType::OP_START)
            starts.push_back(e.cycle);
    });

    const char* yaml = R"(
schedule:
  - id: 0
    op: delay
    unit: vector_core
    params: { latency_cycles: 100 }
  - id: 1
    op: delay
    unit: vector_core
    params: { latency_cycles: 50 }
  - id: 2
    op: delay
    unit: vector_core
    params: { latency_cycles: 75 }
)";
    Schedule   s = Schedule::from_yaml_string(yaml);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(starts == std::vector<Cycle>{0, 100, 150});
    REQUIRE(final_cycle == 225);
}

TEST_CASE("logical unit pool picks the earliest free physical unit") {
    std::stringstream ss;
    EventEngine engine;
    engine.register_unit(std::make_unique<DelayUnit>("vector_core_0", 0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("vector_core_1", 0, nullptr, ss));

    std::vector<Cycle> starts;
    std::vector<std::string> targets;
    engine.set_trace([&](const Event& e) {
        if (e.type == EventType::OP_START) {
            starts.push_back(e.cycle);
            targets.push_back(engine.get_unit(e.target)->name());
        }
    });

    const char* yaml = R"(
schedule:
  - id: 0
    op: delay
    unit: vector_core
    params: { latency_cycles: 100 }
  - id: 1
    op: delay
    unit: vector_core
    params: { latency_cycles: 50 }
  - id: 2
    op: delay
    unit: vector_core
    params: { latency_cycles: 75 }
)";
    Schedule   s = Schedule::from_yaml_string(yaml);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(starts == std::vector<Cycle>{0, 0, 50});
    REQUIRE(targets == std::vector<std::string>{
        "vector_core_0", "vector_core_1", "vector_core_1"});
    REQUIRE(final_cycle == 125);
}

TEST_CASE("diamond dependency graph: critical path determines final cycle") {
    // A(10) -> B(20), A(10) -> C(30) -> D(5)
    // Critical path: A + C + D = 10 + 30 + 5 = 45
    std::stringstream ss;
    EventEngine engine;
    for (const char* n : {"uA", "uB", "uC", "uD"})
        engine.register_unit(std::make_unique<DelayUnit>(n, 0, nullptr, ss));

    const char* yaml = R"(
schedule:
  - id: 0
    op: delay
    unit: uA
    params: { latency_cycles: 10 }
  - id: 1
    op: delay
    unit: uB
    params: { latency_cycles: 20 }
    depends_on: [0]
  - id: 2
    op: delay
    unit: uC
    params: { latency_cycles: 30 }
    depends_on: [0]
  - id: 3
    op: delay
    unit: uD
    params: { latency_cycles: 5 }
    depends_on: [1, 2]
)";
    Schedule   s = Schedule::from_yaml_string(yaml);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(final_cycle == 45);
}

TEST_CASE("coarse op fires events across multiple units simultaneously") {
    // A composite 'fa2_tile' op issues OP_START to dma, systolic, and vector_core
    // at the same cycle. The longest (systolic, 60 cycles) determines the end.
    std::stringstream ss;
    EventEngine engine;
    engine.register_unit(std::make_unique<DelayUnit>("dma",      0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("systolic", 0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("vector_core", 0, nullptr, ss));

    // Track how many events fire at which cycle.
    std::vector<Cycle> event_cycles;
    engine.set_trace([&](const Event& e) { event_cycles.push_back(e.cycle); });

    OpRegistry reg;
    reg.register_op("fa2_tile", [](const IssueCtx& ctx) {
        struct Sub { const char* unit; Cycle lat; const char* label; };
        for (auto sub : std::vector<Sub>{
                {"dma",      30, "load_K"},
                {"systolic", 60, "QK_T"},
                {"vector_core", 20, "scale"}}) {
            Event e;
            e.type    = EventType::OP_START;
            e.target  = ctx.engine.find_unit(sub.unit);
            e.cycle   = ctx.engine.current_cycle();
            e.instr   = ctx.inst.id;
            e.label   = sub.label;
            e.payload = static_cast<int64_t>(sub.lat);
            ctx.engine.schedule(std::move(e));
        }
        // Note: no notify_done here (smoke-test only; not tracking scheduler).
    });

    const char* yaml = R"(
schedule:
  - id: 0
    op: fa2_tile
    label: "FA-2 tile 0"
)";
    Schedule  s = Schedule::from_yaml_string(yaml);
    Scheduler sched(engine, reg, s);
    sched.launch();
    engine.run();

    // 3 OP_START (all at cycle 0) + 3 OP_DONE = 6 trace events total.
    REQUIRE(event_cycles.size() == 6);
    // Simulation ends when the slowest (systolic, lat=60) fires its OP_DONE.
    REQUIRE(engine.current_cycle() == 60);
}

TEST_CASE("PrintingUnit handles all EventTypes without crashing") {
    std::stringstream ss;
    EventEngine engine;
    UnitId u = engine.register_unit(std::make_unique<PrintingUnit>("printer", ss));

    for (EventType t : {EventType::OP_START, EventType::OP_DONE,
                        EventType::DMA_DONE, EventType::BUFFER_SWAP,
                        EventType::BARRIER,  EventType::CUSTOM}) {
        Event e; e.type = t; e.target = u; e.label = "test";
        engine.schedule(e);
    }
    REQUIRE_NOTHROW(engine.run());
    // Every event should have printed the unit name.
    REQUIRE(ss.str().find("printer") != std::string::npos);
}

TEST_CASE("dummy round-trip: sim_main style setup runs without errors") {
    // Mirrors what sim_main does with the dummy_example schedule.
    std::stringstream ss;
    EventEngine engine(1.0);

    engine.register_unit(std::make_unique<DelayUnit>("systolic",    0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("access_core_0", 0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("dma_0",         0, nullptr, ss));
    engine.register_unit(std::make_unique<DelayUnit>("vector_core_0", 0, nullptr, ss));

    const char* yaml = R"(
schedule:
  - id: 0
    op: delay
    unit: dma
    params: { latency_cycles: 50 }
    label: "DMA load K_tile"
  - id: 1
    op: delay
    unit: access_core
    params: { latency_cycles: 30 }
    depends_on: [0]
    label: "transpose"
  - id: 2
    op: delay
    unit: systolic
    params: { latency_cycles: 200 }
    depends_on: [1]
    label: "GEMM"
  - id: 3
    op: delay
    unit: vector_core
    params: { latency_cycles: 40 }
    depends_on: [2]
    label: "softmax"
)";
    Schedule   s = Schedule::from_yaml_string(yaml);
    OpRegistry reg; add_delay_op(reg);
    Scheduler  sched(engine, reg, s);
    wire(engine, sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(final_cycle == 320);  // 50 + 30 + 200 + 40
}
