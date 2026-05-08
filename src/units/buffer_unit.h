#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <array>
#include <iostream>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// SramAccess — payload for sram_read / sram_write ops.
// ---------------------------------------------------------------------------
struct SramAccess {
    uint64_t bytes    = 0;
    bool     is_write = false;
};

// ---------------------------------------------------------------------------
// BufferUnit — shared double-buffered on-chip SRAM (IBUF or OBUF).
//
// TIMING MODEL
// ─────────────────────────────────────────────────────────────────────────
//   access_latency = ceil(bytes / banking_factor)
//
//   banking_factor: number of parallel SRAM ports (read or write per cycle).
//   e.g. 8 KB/cycle at banking_factor=8 with 1-byte-wide ports.
//
// DOUBLE-BUFFERING
//   The buffer has two banks.  Each bank tracks when it is next free:
//     bank_free_at[0], bank_free_at[1]
//
//   On every access (read or write) at cycle C:
//     1. Choose bank b = argmin(bank_free_at[i])
//     2. Effective start = max(C, bank_free_at[b])   (wait if bank busy)
//     3. bank_free_at[b] = effective_start + access_latency
//     4. Schedule OP_DONE at effective_start + access_latency
//
//   This models the producer-consumer overlap of double-buffering:
//   while the systolic array reads from bank 0, DMA can write into bank 1.
//   The unit correctly serializes accesses that land on the same bank.
//
// BACKWARD COMPATIBILITY
//   If payload is int64_t (op: delay), that value is used as latency.
//   If payload is SramAccess, latency is computed from banking_factor.
//
// EVENT PROTOCOL
//   OP_START → select bank → compute latency → schedule OP_DONE
//   OP_DONE  → notify scheduler
// ---------------------------------------------------------------------------
class BufferUnit : public Unit {
public:
    BufferUnit(std::string name, const SramConfig& cfg,
               Scheduler* sched = nullptr, std::ostream& os = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }
    void handle(const Event& e, EventEngine& engine) override;

    // Access latency for `bytes` bytes (ignoring bank availability).
    Cycle access_latency(uint64_t bytes) const;

    // Reset double-buffer state (useful for tests).
    void reset_banks() { bank_free_at_[0] = bank_free_at_[1] = 0; }

private:
    SramConfig            cfg_;
    Scheduler*            sched_;
    std::ostream&         os_;
    std::array<Cycle, 2>  bank_free_at_ = {0, 0};
};

}  // namespace sim
