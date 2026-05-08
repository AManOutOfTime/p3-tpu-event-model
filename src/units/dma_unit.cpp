#include "units/dma_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

DmaUnit::DmaUnit(std::string name, const ArchConfig& cfg,
                 TensorStore* ts, Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), ts_(ts), sched_(sched), os_(os) {}

// ---------------------------------------------------------------------------
// transfer_latency  —  HBM → IBUF
//   hbm.latency_cycles + ceil(bytes / (hbm_bw × channels))
// ---------------------------------------------------------------------------
Cycle DmaUnit::transfer_latency(uint64_t bytes) const {
    if (bytes == 0) return 0;
    const double bw  = cfg_.hbm_bytes_per_cycle()
                     * static_cast<double>(cfg_.dma.channels);
    const Cycle  xfer = static_cast<Cycle>(std::ceil(
                            static_cast<double>(bytes) / bw));
    return static_cast<Cycle>(cfg_.hbm.latency_cycles) + xfer;
}

// ---------------------------------------------------------------------------
// stage_latency  —  IBUF → systolic array PE registers (on-chip)
//   ceil(bytes / banking_factor)
//   banking_factor parallel SRAM ports, no HBM latency penalty.
// ---------------------------------------------------------------------------
Cycle DmaUnit::stage_latency(uint64_t bytes) const {
    if (bytes == 0 || cfg_.sram.banking_factor == 0) return 0;
    return static_cast<Cycle>(std::ceil(
        static_cast<double>(bytes) /
        static_cast<double>(cfg_.sram.banking_factor)));
}

// ---------------------------------------------------------------------------
// handle
// ---------------------------------------------------------------------------
void DmaUnit::handle(const Event& e, EventEngine& engine) {

    if (e.type == EventType::OP_START) {
        Cycle lat = 0;

        if (const auto* t = std::any_cast<DmaTransfer>(&e.payload)) {
            lat = t->on_chip ? stage_latency(t->bytes)
                             : transfer_latency(t->bytes);

            os_ << "  [" << name() << "]  "
                << (t->on_chip ? "STAGE_START" : "DMA_START")
                << "  instr="  << e.instr
                << "  @cycle=" << e.cycle
                << "  bytes="  << t->bytes
                << "  lat="    << lat;
            if (!t->src_buf.empty())
                os_ << "  " << t->src_buf << " → " << t->dst_buf;
            os_ << (e.label.empty() ? "" : "  \"" + e.label + "\"")
                << "\n";

        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            // backward compat: op: delay
            lat = static_cast<Cycle>(*p);
            os_ << "  [" << name() << "]  DMA_START"
                << "  instr="  << e.instr
                << "  @cycle=" << e.cycle
                << "  lat="    << lat
                << (e.label.empty() ? "" : "  \"" + e.label + "\"")
                << "\n";
        }

        Event done  = e;
        done.type   = EventType::OP_DONE;
        done.cycle  = e.cycle + lat;
        done.seq    = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {

        // ── Buffer copy in TensorStore ─────────────────────────────────
        // Represents data arriving at destination (IBUF or array register).
        if (ts_) {
            if (const auto* t = std::any_cast<DmaTransfer>(&e.payload)) {
                if (!t->src_buf.empty() && !t->dst_buf.empty()
                    && ts_->has(t->src_buf)) {
                    // Copy src → dst in TensorStore
                    ts_->set(t->dst_buf, ts_->get(t->src_buf));
                    os_ << "  [" << name() << "]  "
                        << (t->on_chip ? "STAGE_COPY" : "DMA_COPY")
                        << "  \"" << t->src_buf
                        << "\" → \"" << t->dst_buf << "\"\n";
                }
            }
        }

        os_ << "  [" << name() << "]  "
            << "DMA_DONE"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"")
            << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
