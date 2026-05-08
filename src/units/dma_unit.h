#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// DmaTransfer — payload for dma_load / dma_store / stage ops.
//
//   bytes    : number of bytes to transfer (determines latency)
//   src_buf  : TensorStore key to copy FROM  (empty = no TensorStore copy)
//   dst_buf  : TensorStore key to copy TO
//
// If src_buf and dst_buf are both set, the DmaUnit copies the buffer in the
// TensorStore at OP_DONE time — modelling data arriving at destination.
//
// Transfer kinds:
//   HBM → IBUF   : hbm_latency + ceil(bytes / hbm_bw)        (long)
//   IBUF → array : ceil(bytes / banking_factor)               (short, on-chip)
//
// The `on_chip` flag selects which latency formula to use.
// ---------------------------------------------------------------------------
struct DmaTransfer {
    uint64_t    bytes    = 0;
    std::string src_buf;       // TensorStore source key (empty = no copy)
    std::string dst_buf;       // TensorStore destination key
    bool        on_chip  = false;  // true = IBUF→array (no HBM latency)
};

// ---------------------------------------------------------------------------
// DmaUnit
// ---------------------------------------------------------------------------
class DmaUnit : public Unit {
public:
    DmaUnit(std::string name, const ArchConfig& cfg,
            TensorStore* ts    = nullptr,
            Scheduler*   sched = nullptr,
            std::ostream& os   = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_    = ts; }

    void handle(const Event& e, EventEngine& engine) override;

    // HBM → IBUF latency
    Cycle transfer_latency(uint64_t bytes) const;

    // IBUF → array (on-chip) latency: ceil(bytes / banking_factor)
    Cycle stage_latency(uint64_t bytes) const;

private:
    const ArchConfig& cfg_;
    TensorStore*      ts_;
    Scheduler*        sched_;
    std::ostream&     os_;
};

}  // namespace sim
