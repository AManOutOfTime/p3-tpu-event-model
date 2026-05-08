#include "units/buffer_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <algorithm>
#include <cmath>

namespace sim {

BufferUnit::BufferUnit(std::string name, const SramConfig& cfg,
                       Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

// ---------------------------------------------------------------------------
// access_latency
//   Raw SRAM access time (ignoring which bank is free).
//   banking_factor parallel ports → ceil(bytes / banking_factor) cycles.
//
//   Example: banking_factor=8
//     32 KB = 32768 bytes → ceil(32768 / 8) = 4096 cycles
// ---------------------------------------------------------------------------
Cycle BufferUnit::access_latency(uint64_t bytes) const {
    if (bytes == 0 || cfg_.banking_factor == 0) return 0;
    return static_cast<Cycle>(
        std::ceil(static_cast<double>(bytes) /
                  static_cast<double>(cfg_.banking_factor)));
}

void BufferUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle    lat       = 0;
        bool     from_cfg  = false;
        bool     is_write  = false;
        uint64_t bytes     = 0;

        if (const auto* a = std::any_cast<SramAccess>(&e.payload)) {
            lat      = access_latency(a->bytes);
            from_cfg = true;
            is_write = a->is_write;
            bytes    = a->bytes;
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);   // op: delay backward compat
        }

        // ── Double-buffer bank selection ────────────────────────────────
        // Pick the bank that becomes free soonest.  Accesses that arrive
        // while a bank is still busy will stall on that bank until it frees.
        const uint8_t bank = (bank_free_at_[0] <= bank_free_at_[1]) ? 0 : 1;
        const Cycle   eff_start = std::max(e.cycle, bank_free_at_[bank]);
        bank_free_at_[bank] = eff_start + lat;

        os_ << "  [" << name() << "]  SRAM_" << (is_write ? "WRITE" : "READ")
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle;
        if (from_cfg) {
            os_ << "  bytes=" << bytes
                << "  bank="  << static_cast<int>(bank)
                << "  eff_start=" << eff_start
                << "  lat=" << lat;
        } else {
            os_ << "  lat=" << lat;
        }
        os_ << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        Event done = e;
        done.type  = EventType::OP_DONE;
        done.cycle = eff_start + lat;
        done.seq   = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        os_ << "  [" << name() << "]  SRAM_DONE"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";
        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
