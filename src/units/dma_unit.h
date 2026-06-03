#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// DmaTransfer — payload for dma_load / dma_store / dma_stage ops.
//
//   bytes    : number of bytes to transfer (determines latency)
//   src_buf  : TensorStore key to copy FROM  (empty = no TensorStore copy)
//   dst_buf  : TensorStore key to copy TO
//   on_chip  : true = IBUF→array staging (no HBM latency penalty)
//
// Transfer latency formulas:
//   HBM ↔ IBUF  : hbm_latency_cycles + ceil(bytes / (hbm_bw × channels))
//   IBUF → array: ceil(bytes / banking_factor)
// ---------------------------------------------------------------------------
struct DmaTransfer {
    uint64_t    bytes    = 0;
    std::string src_buf;       // TensorStore source key (empty = no copy)
    std::string dst_buf;       // TensorStore destination key
    bool        on_chip  = false;  // true = IBUF→array staging
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
//     latency = ceil(bytes / sram.banking_factor)
//
// On OP_DONE the unit copies src_buf → dst_buf in the attached TensorStore
// to model data arriving at destination.
// ---------------------------------------------------------------------------
class DmaUnit : public Unit {
public:
    DmaUnit(std::string name, const ArchConfig& cfg,
            TensorStore* ts    = nullptr,
            Scheduler*   sched = nullptr,
            std::ostream& os   = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_ = ts; }

    void handle(const Event& e, EventEngine& engine) override;

    // HBM ↔ IBUF latency: hbm_latency + ceil(bytes / (hbm_bw × channels))
    Cycle transfer_latency(uint64_t bytes) const;

    // IBUF → array latency: ceil(bytes / banking_factor)
    Cycle stage_latency(uint64_t bytes) const;

private:
    const ArchConfig& cfg_;
    TensorStore*      ts_;
    Scheduler*        sched_;
    std::ostream&     os_;
};

}  // namespace sim
