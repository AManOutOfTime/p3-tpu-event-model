#include "units/access_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

AccessUnit::AccessUnit(std::string name, const AccessCoreConfig& cfg,
                       Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

Cycle AccessUnit::compute_latency(uint64_t elements) const {
    if (elements == 0 || cfg_.bandwidth == 0) return 0;
    return static_cast<Cycle>(
        std::ceil(static_cast<double>(elements) /
                  static_cast<double>(cfg_.bandwidth)));
}

void AccessUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle    lat      = 0;
        bool     from_cfg = false;
        uint64_t elements = 0;
        std::string kind  = "?";

        if (const auto* op = std::any_cast<AccessOp>(&e.payload)) {
            lat      = compute_latency(op->elements);
            from_cfg = true;
            elements = op->elements;
            kind     = op->kind.empty() ? "?" : op->kind;
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);
        }

        os_ << "  [" << name() << "]  ACCESS_START"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle;
        if (from_cfg) {
            os_ << "  kind=" << kind
                << "  elems=" << elements
                << "  lat=" << lat;
        } else {
            os_ << "  lat=" << lat;
        }
        os_ << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        Event done = e;
        done.type  = EventType::OP_DONE;
        done.cycle = e.cycle + lat;
        done.seq   = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        os_ << "  [" << name() << "]  ACCESS_DONE"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";
        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
