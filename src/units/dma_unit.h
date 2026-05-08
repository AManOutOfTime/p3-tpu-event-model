#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// DmaTransfer — payload for dma_load / dma_store ops.
// ---------------------------------------------------------------------------
struct DmaTransfer {
    uint64_t bytes    = 0;    // bytes to transfer
};

// ---------------------------------------------------------------------------
// DmaUnit — models a single HBM <-> on-chip SRAM DMA channel.
//
// TIMING MODEL
// ─────────────────────────────────────────────────────────────────────────
//   latency = hbm.latency_cycles + ceil(bytes / hbm_bytes_per_cycle)
//
//   • hbm.latency_cycles  : fixed HBM access latency (DRAM row open, etc.)
//   • hbm_bytes_per_cycle : derived from bandwidth_tb_s / clock_ghz
//                           e.g. 2 TB/s at 1 GHz = 2000 bytes/cycle
//
// MULTI-CHANNEL
//   dma.channels > 1 scales effective bandwidth: effective BW = channels × BW.
//   latency = hbm.latency_cycles + ceil(bytes / (channels × hbm_bytes_per_cycle))
//
// BACKWARD COMPATIBILITY
//   If payload is int64_t (from op: delay), that value is used as-is.
//   If payload is DmaTransfer, latency is computed from arch config.
//
// EVENT PROTOCOL
//   OP_START → compute latency → schedule OP_DONE
//   OP_DONE  → notify scheduler
// ---------------------------------------------------------------------------
class DmaUnit : public Unit {
public:
    DmaUnit(std::string name, const ArchConfig& cfg,
            Scheduler* sched = nullptr, std::ostream& os = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }
    void handle(const Event& e, EventEngine& engine) override;

    // Compute transfer latency for `bytes` bytes.
    Cycle transfer_latency(uint64_t bytes) const;

private:
    const ArchConfig& cfg_;
    Scheduler*        sched_;
    std::ostream&     os_;
};

}  // namespace sim
