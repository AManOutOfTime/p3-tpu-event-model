#include "core/event_engine.h"
#include "core/logger.h"
#include "config/arch_config.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/delay_unit.h"
#include <iostream>
#include <string>

using namespace sim;

// ---------------------------------------------------------------------------
// Built-in ops. Register additional ops here (or in your own module) before
// constructing the Scheduler.
// ---------------------------------------------------------------------------

// "delay" op: issues one OP_START on inst.unit; latency from
// params["latency_cycles"] (default 10). Works with any DelayUnit target.
static void register_builtin_ops(OpRegistry& reg) {
    reg.register_op("delay", [](const IssueCtx& ctx) {
        UnitId target = ctx.engine.find_unit(ctx.inst.unit);
        if (target == INVALID_UNIT)
            throw std::runtime_error("delay op: unknown unit '" + ctx.inst.unit + "'");

        Event e;
        e.type    = EventType::OP_START;
        e.target  = target;
        e.cycle   = ctx.engine.current_cycle();
        e.instr   = ctx.inst.id;
        e.label   = ctx.inst.label;
        e.payload = static_cast<int64_t>(pget_int(ctx.inst.params, "latency_cycles", 10));
        ctx.engine.schedule(std::move(e));
    });
}

int main(int argc, char** argv) {
    std::string config_path = "configs/default.yaml";
    std::string sched_path  = "schedules/dummy_example.yaml";
    bool        trace       = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--config"   && i + 1 < argc) config_path = argv[++i];
        else if (a == "--schedule" && i + 1 < argc) sched_path  = argv[++i];
        else if (a == "--no-trace")                 trace = false;
        else {
            std::cerr << "Usage: sim_main [--config FILE] [--schedule FILE] [--no-trace]\n";
            return 1;
        }
    }

    // Load config + schedule.
    ArchConfig arch     = ArchConfig::from_yaml_file(config_path);
    Schedule   schedule = Schedule::from_yaml_file(sched_path);

    std::cout << "clock=" << arch.clock_ghz << " GHz  "
              << "systolic=" << arch.systolic.rows << "x" << arch.systolic.cols
              << "  precision=" << arch.systolic.precision << "\n"
              << "hbm_bw=" << arch.hbm.bandwidth_tb_s << " TB/s  "
              << "hbm_bytes_per_cycle=" << arch.hbm_bytes_per_cycle() << "\n\n";

    // Build engine and register hardware units.
    // Unit names must match the 'unit:' fields in your schedule YAML.
    // Swap DelayUnit for a real hardware model (SystolicUnit, DmaUnit, etc.)
    // when that phase is ready -- everything else stays the same.
    EventEngine engine(arch.clock_ghz);
    engine.register_unit(std::make_unique<DelayUnit>("systolic",      0));
    engine.register_unit(std::make_unique<DelayUnit>("tandem_1",      0));
    engine.register_unit(std::make_unique<DelayUnit>("tandem_2",      0));
    engine.register_unit(std::make_unique<DelayUnit>("tandem_3",      0));
    engine.register_unit(std::make_unique<DelayUnit>("access_core_1", 0));
    engine.register_unit(std::make_unique<DelayUnit>("access_core_2", 0));
    engine.register_unit(std::make_unique<DelayUnit>("dma",           0));
    engine.register_unit(std::make_unique<DelayUnit>("vector_core",   0));

    // Register ops and build scheduler.
    OpRegistry reg;
    register_builtin_ops(reg);
    Scheduler scheduler(engine, reg, schedule);

    // Wire scheduler back into every DelayUnit so OP_DONE calls notify_done.
    for (UnitId id = 0; id < static_cast<UnitId>(engine.num_units()); id++)
        if (auto* du = dynamic_cast<DelayUnit*>(engine.get_unit(id)))
            du->set_scheduler(&scheduler);

    // Optional per-event trace to stdout.
    ConsoleLogger logger(engine);
    if (trace) engine.set_trace([&](const Event& e) { logger(e); });

    // Run.
    std::cout << "== simulation start  instructions=" << schedule.instructions.size() << " ==\n";
    scheduler.launch();
    Cycle final_cycle = engine.run();
    std::cout << "== simulation done"
              << "  cycle=" << final_cycle
              << "  (" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns)"
              << "  outstanding=" << scheduler.outstanding() << " ==\n";

    return scheduler.all_done() ? 0 : 1;
}
