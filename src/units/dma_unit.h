#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// DmaTransfer — payload for dma_load / dma_store / dma_stage ops.
//
//   bytes   : number of bytes to transfer (determines latency)
//   src_buf : symbolic source buffer name (used in trace output only)
//   dst_buf : symbolic destination buffer name
//   on_chip : true = IBUF→array staging (no HBM latency penalty)
//
// Transfer latency formulas:
//   HBM ↔ IBUF  : hbm_latency_cycles + ceil(bytes / (hbm_bw × channels))
//   IBUF → array: ceil(elements / array_rows)   (wide on-chip operand bus)
// ---------------------------------------------------------------------------
struct DmaTransfer {
    uint64_t    bytes   = 0;
    std::string src_buf;
    std::string dst_buf;
    bool        on_chip = false;
};

// ---------------------------------------------------------------------------
// DmaUnit — models an HBM <-> SRAM DMA channel.
//
//   dma_load    HBM → IBUF
//     latency = hbm.latency_cycles + ceil(bytes / (hbm_bw × channels))
//
//   dma_store   OBUF → HBM
//     latency = same formula (symmetric bandwidth)
//
//   dma_stage   IBUF → systolic_array PE registers (on-chip)
//     latency = ceil(elements / systolic.rows)   (wide operand bus, rows lanes)
// ---------------------------------------------------------------------------
class DmaUnit : public Unit {
public:
    DmaUnit(std::string name, const ArchConfig& cfg,
            Scheduler*    sched = nullptr,
            std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }

    void handle(const Event& e, EventEngine& engine) override;

    // HBM ↔ IBUF latency: hbm_latency + ceil(bytes / (hbm_bw × channels))
    Cycle transfer_latency(uint64_t bytes) const;

    // IBUF → array latency: ceil(elements / systolic.rows) (wide operand bus)
    Cycle stage_latency(uint64_t bytes) const;

private:
    const ArchConfig& cfg_;
    Scheduler*        sched_;
    std::ostream&     os_;
};

}  // namespace sim
