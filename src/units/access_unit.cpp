#include "units/access_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

AccessUnit::AccessUnit(std::string name, const AccessCoreConfig& cfg,
                       Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

Cycle AccessUnit::compute_latency(uint64_t elements) const {
    if (!elements || !cfg_.bandwidth) return 0;
    return static_cast<Cycle>(std::ceil(
        static_cast<double>(elements) / cfg_.bandwidth));
}

void AccessUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat = 0;
        if (const auto* op = std::any_cast<AccessOp>(&e.payload)) {
            lat = compute_latency(op->elements);
            if (verbose_)
                os_ << "  [" << name() << "]  ACCESS_START"
                    << "  instr=" << e.instr << "  @cycle=" << e.cycle
                    << "  kind=" << op->kind << "  elems=" << op->elements
                    << "  lat=" << lat;
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);
            if (verbose_)
                os_ << "  [" << name() << "]  ACCESS_START"
                    << "  instr=" << e.instr << "  @cycle=" << e.cycle
                    << "  lat=" << lat;
        }
        if (verbose_)
            os_ << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        Event done = e;
        done.type  = EventType::OP_DONE;
        done.cycle = e.cycle + lat;
        done.seq   = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        if (verbose_)
            os_ << "  [" << name() << "]  ACCESS_DONE"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
