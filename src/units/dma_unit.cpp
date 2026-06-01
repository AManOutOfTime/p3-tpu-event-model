#include "units/dma_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

DmaUnit::DmaUnit(std::string name, const ArchConfig& cfg,
                 TensorStore*, Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

Cycle DmaUnit::load_store_latency(uint64_t bytes) const {
    if (!bytes) return 0;
    double bw = cfg_.hbm_bytes_per_cycle() * cfg_.dma.channels;
    return static_cast<Cycle>(cfg_.hbm.latency_cycles) +
           static_cast<Cycle>(std::ceil(static_cast<double>(bytes) / bw));
}

Cycle DmaUnit::stage_latency(uint64_t bytes) const {
    if (!bytes || !cfg_.sram.banking_factor) return 0;
    return static_cast<Cycle>(std::ceil(
        static_cast<double>(bytes) / cfg_.sram.banking_factor));
}

void DmaUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat = 0;
        std::string kind = "DMA";

        if (const auto* t = std::any_cast<DmaTransfer>(&e.payload)) {
            kind = t->on_chip ? "STAGE" : "DMA";
            lat  = t->on_chip ? stage_latency(t->bytes)
                              : load_store_latency(t->bytes);
            os_ << "  [" << name() << "]  " << kind << "_START"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << "  bytes=" << t->bytes << "  lat=" << lat;
            if (!t->src_buf.empty())
                os_ << "  " << t->src_buf << " → " << t->dst_buf;
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);  // backward compat: op: delay
            os_ << "  [" << name() << "]  DMA_START"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << "  lat=" << lat;
        }
        os_ << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        Event done = e;
        done.type  = EventType::OP_DONE;
        done.cycle = e.cycle + lat;
        done.seq   = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        os_ << "  [" << name() << "]  DMA_DONE"
            << "  instr=" << e.instr << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
