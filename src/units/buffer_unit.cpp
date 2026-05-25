#include "units/buffer_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>
#include <algorithm>

namespace sim {

BufferUnit::BufferUnit(std::string name, const SramConfig& cfg,
                       Scheduler* sched, TensorStore* ts, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), ts_(ts), os_(os) {}

// ---------------------------------------------------------------------------
// access_latency
//   ceil(bytes / banking_factor)
//   banking_factor parallel ports → one port transfers one "stripe" per cycle.
//   e.g. banking_factor=8, bytes=32768 → 4096 cycles
// ---------------------------------------------------------------------------
Cycle BufferUnit::access_latency(uint64_t bytes) const {
    if (!bytes || !cfg_.banking_factor) return 0;
    return static_cast<Cycle>(
        std::ceil(static_cast<double>(bytes) / cfg_.banking_factor));
}

// ---------------------------------------------------------------------------
// select_bank
//   Double-buffer half assignment:
//     Writes → producer half  (write_count_ % 2)
//     Reads  → consumer half  (the other one)
//   This ensures DMA (writing the next tile) and systolic (reading the
//   current tile) are always on different halves and never contend.
// ---------------------------------------------------------------------------
int BufferUnit::select_bank(bool is_write) {
    if (is_write) {
        int bank = static_cast<int>(write_count_ % 2);
        write_count_++;
        return bank;
    }
    // Consumer reads from whichever half the last write went to
    // (write_count_ has already been incremented, so -1)
    return static_cast<int>((write_count_ == 0 ? 0 : (write_count_ - 1)) % 2);
}

void BufferUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat = 0;

        if (const auto* a = std::any_cast<SramAccess>(&e.payload)) {
            lat = access_latency(a->bytes);

            // ── Double-buffer bank selection ──────────────────────────────
            const int   bank        = select_bank(a->is_write);
            const Cycle eff_start   = std::max(e.cycle, bank_free_at_[bank]);
            bank_free_at_[bank]     = eff_start + lat;

            os_ << "  [" << name() << "]  SRAM_"
                << (a->is_write ? "WRITE" : "READ ")
                << "  instr="      << e.instr
                << "  @cycle="     << e.cycle
                << "  bytes="      << a->bytes
                << "  bank="       << bank
                << "  eff_start="  << eff_start
                << "  lat="        << lat
                << "  free_at="    << bank_free_at_[bank];
            if (!a->src_buf.empty())
                os_ << "  " << a->src_buf << " → " << a->dst_buf;
            os_ << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

            // Schedule OP_DONE at eff_start + lat (may be later than e.cycle)
            Event done   = e;
            done.type    = EventType::OP_DONE;
            done.cycle   = eff_start + lat;
            done.seq     = engine.next_seq();
            engine.schedule(done);

        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            // backward compat: op: delay
            lat = static_cast<Cycle>(*p);
            os_ << "  [" << name() << "]  SRAM_START"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << "  lat=" << lat << "\n";
            Event done = e;
            done.type  = EventType::OP_DONE;
            done.cycle = e.cycle + lat;
            done.seq   = engine.next_seq();
            engine.schedule(done);
        }

    } else if (e.type == EventType::OP_DONE) {
        // Copy buffer in TensorStore (data has arrived at destination)
        if (ts_)
            if (const auto* a = std::any_cast<SramAccess>(&e.payload))
                if (!a->src_buf.empty() && !a->dst_buf.empty()
                    && ts_->has(a->src_buf))
                    ts_->copy(a->src_buf, a->dst_buf);

        os_ << "  [" << name() << "]  SRAM_DONE"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
