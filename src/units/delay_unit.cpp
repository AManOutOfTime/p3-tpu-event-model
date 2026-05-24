#include "units/delay_unit.h"
#include "schedule/scheduler.h"

namespace sim {

void DelayUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        // Latency: use payload override if present, otherwise the default.
        Cycle lat = default_latency_;
        uint64_t buffer_bytes = 0;
        if (const auto* p = std::any_cast<int64_t>(&e.payload))
            lat = static_cast<Cycle>(*p);
        if (const auto* p = std::any_cast<DelayPayload>(&e.payload)) {
            lat = static_cast<Cycle>(p->latency_cycles);
            buffer_bytes = p->buffer_bytes;
        }

        os_ << "  [" << name() << "]  START  instr=" << e.instr
            << "  @cycle=" << e.cycle << "  lat=" << lat
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        Event done  = e;
        done.type   = EventType::OP_DONE;
        done.cycle  = e.cycle + lat;
        done.seq    = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        os_ << "  [" << name() << "]  DONE   instr=" << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        if (const auto* p = std::any_cast<DelayPayload>(&e.payload))
            engine.release_unit_buffer(e.target, p->buffer_bytes);

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
