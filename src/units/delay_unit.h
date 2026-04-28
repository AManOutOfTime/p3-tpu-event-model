#pragma once
#include "core/unit.h"
#include <iostream>

namespace sim {

class Scheduler;

// Models a fixed-latency hardware unit.
//
//   OP_START received -> print, schedule OP_DONE (default_latency cycles later)
//   OP_DONE  received -> print, call scheduler.notify_done(instr)
//
// Per-event latency override: set event.payload = int64_t before scheduling
// OP_START (the 'delay' op handler does this using params["latency_cycles"]).
//
// This is the starting template for real units (systolic array, DMA, tandem).
// Real units override handle() and replace the fixed latency with a computed
// one based on operation shape, SRAM availability, backpressure, etc.
class DelayUnit : public Unit {
public:
    DelayUnit(std::string name, Cycle default_latency,
              Scheduler* sched = nullptr, std::ostream& os = std::cout)
        : Unit(std::move(name))
        , default_latency_(default_latency)
        , sched_(sched)
        , os_(os) {}

    void set_scheduler(Scheduler* s) { sched_ = s; }

    void handle(const Event& e, EventEngine& engine) override;

private:
    Cycle         default_latency_;
    Scheduler*    sched_;
    std::ostream& os_;
};

}  // namespace sim
