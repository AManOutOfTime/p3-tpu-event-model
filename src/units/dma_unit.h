#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {
class Scheduler;

struct DmaTransfer {
    uint64_t    bytes    = 0;
    std::string src_buf;       // TensorStore source key (empty = no copy)
    std::string dst_buf;       // TensorStore destination key
    bool        on_chip  = false;  // true = IBUF→array (dma_stage)
};

// ---------------------------------------------------------------------------
// DmaUnit — models an HBM <-> SRAM DMA channel.
//
// Three op kinds, all routed through this unit:
//
//   dma_load    HBM → IBUF
//     latency = hbm.latency_cycles + ceil(bytes / (hbm_bw × channels))
//
//   dma_store   OBUF/IBUF → HBM
//     latency = same formula (symmetric bandwidth)
//
//   dma_stage   IBUF → systolic_array.Q_operand / P_operand
//     Reads from on-chip SRAM, no HBM penalty.
//     latency = ceil(bytes / sram.banking_factor)
//
// At OP_DONE, if src_buf and dst_buf are set in the payload, the unit copies
// the buffer in TensorStore — this models data arriving at its destination.
// ---------------------------------------------------------------------------
class DmaUnit : public Unit {
public:
    DmaUnit(std::string name, const ArchConfig& cfg,
            TensorStore*  ts    = nullptr,
            Scheduler*    sched = nullptr,
            std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_    = ts; }

    void handle(const Event& e, EventEngine& engine) override;

    Cycle load_store_latency(uint64_t bytes) const;  // HBM latency
    Cycle stage_latency(uint64_t bytes) const;        // on-chip latency

private:
    const ArchConfig& cfg_;
    TensorStore*      ts_;
    Scheduler*        sched_;
    std::ostream&     os_;
};

}  // namespace sim
