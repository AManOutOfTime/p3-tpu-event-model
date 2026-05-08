#include "units/dma_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

DmaUnit::DmaUnit(std::string name, const ArchConfig& cfg,
                 Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

// ---------------------------------------------------------------------------
// transfer_latency
//
//   effective_bw = hbm_bytes_per_cycle × channels
//   latency = hbm.latency_cycles + ceil(bytes / effective_bw)
//
//   Example (defaults): 2 TB/s at 1 GHz, 1 channel → 2000 bytes/cycle
//     32 KB tile: 200 + ceil(32768 / 2000) = 200 + 17 = 217 cycles
//     64 KB tile: 200 + ceil(65536 / 2000) = 200 + 33 = 233 cycles
// ---------------------------------------------------------------------------
Cycle DmaUnit::transfer_latency(uint64_t bytes) const {
    if (bytes == 0) return 0;
    const double effective_bw = cfg_.hbm_bytes_per_cycle()
                               * static_cast<double>(cfg_.dma.channels);
    const Cycle xfer = static_cast<Cycle>(
        std::ceil(static_cast<double>(bytes) / effective_bw));
    return static_cast<Cycle>(cfg_.hbm.latency_cycles) + xfer;
}

void DmaUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat = 0;
        bool  from_cfg = false;

        if (const auto* t = std::any_cast<DmaTransfer>(&e.payload)) {
            lat      = transfer_latency(t->bytes);
            from_cfg = true;
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);   // op: delay backward compat
        }

        os_ << "  [" << name() << "]  DMA_START"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle;
        if (from_cfg) {
            const auto* t = std::any_cast<DmaTransfer>(&e.payload);
            os_ << "  bytes=" << t->bytes
                << "  lat="   << lat;
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
        os_ << "  [" << name() << "]  DMA_DONE"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";
        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
