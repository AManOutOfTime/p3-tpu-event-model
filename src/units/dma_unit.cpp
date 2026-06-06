#include "units/dma_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

static uint32_t stage_dtype_bytes(const std::string& precision) {
    if (precision == "FP8")  return 1;
    if (precision == "FP32") return 4;
    return 2;  // BF16 / FP16
}

DmaUnit::DmaUnit(std::string name, const ArchConfig& cfg,
                 Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

// ---------------------------------------------------------------------------
// transfer_latency  —  HBM ↔ IBUF
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
//   ceil(elements / array_rows)   [no HBM latency penalty]
//
//   Staging feeds the array over its WIDE operand bus: `rows` input lanes
//   (one per PE row) accept one column of `rows` elements per cycle. So the
//   ingest bandwidth is `rows` elements/cycle = rows * dtype_bytes per cycle.
//   This matches the K-cycle GEMM streaming model and is intentionally NOT
//   bounded by the narrow SRAM banking_factor (that governs BufferUnit SRAM
//   r/w, a separate path).
// ---------------------------------------------------------------------------
Cycle DmaUnit::stage_latency(uint64_t bytes) const {
    const uint32_t lanes  = cfg_.systolic.rows;
    const uint32_t dbytes = stage_dtype_bytes(cfg_.systolic.precision);
    if (bytes == 0 || lanes == 0 || dbytes == 0) return 0;
    const double bytes_per_cycle = static_cast<double>(lanes)
                                 * static_cast<double>(dbytes);
    return static_cast<Cycle>(std::ceil(
        static_cast<double>(bytes) / bytes_per_cycle));
}

void DmaUnit::handle(const Event& e, EventEngine& engine) {

    if (e.type == EventType::OP_START) {
        Cycle lat = 0;

        if (const auto* t = std::any_cast<DmaTransfer>(&e.payload)) {
            lat = t->on_chip ? stage_latency(t->bytes)
                             : transfer_latency(t->bytes);

            if (verbose_) {
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
            }

        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            // backward compat: op: delay
            lat = static_cast<Cycle>(*p);
            if (verbose_)
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

        if (verbose_)
            os_ << "  [" << name() << "]  DMA_DONE"
                << "  instr="  << e.instr
                << "  @cycle=" << e.cycle
                << (e.label.empty() ? "" : "  \"" + e.label + "\"")
                << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
