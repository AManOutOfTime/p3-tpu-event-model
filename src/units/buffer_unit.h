#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include <array>
#include <iostream>
#include <string>

namespace sim {
class Scheduler;

// ---------------------------------------------------------------------------
// SramAccess — payload for sram_read / sram_write ops.
// ---------------------------------------------------------------------------
struct SramAccess {
    uint64_t    bytes    = 0;
    bool        is_write = false;
    std::string src_buf;    // TensorStore key to read from (writes)
    std::string dst_buf;    // TensorStore key to write to  (reads)
};

// ---------------------------------------------------------------------------
// BufferUnit — double-buffered banked SRAM (IBUF or OBUF).
//
// BANKING CONTENTION
// ──────────────────
//   The SRAM has `banking_factor` parallel ports (banks).
//   Each access is assigned to bank = hash(address) % banking_factor.
//   If that bank is busy, the access stalls until it is free:
//
//     effective_start = max(now, bank_free_at[bank])
//     bank_free_at[bank] = effective_start + access_latency
//     access_latency = ceil(bytes / banking_factor)
//
//   Two accesses to DIFFERENT banks can overlap freely.
//   Two accesses to the SAME bank are serialized.
//
// DOUBLE-BUFFERING
// ────────────────
//   The buffer has two logical halves (ping / pong).
//   Producer (DMA) writes to the idle half.
//   Consumer (systolic / vector) reads from the active half.
//   They can proceed simultaneously — no stall — as long as they
//   target different halves.
//
//   Half assignment:
//     producer_bank = write_count % 2        (alternates 0, 1, 0, 1 …)
//     consumer_bank = 1 - producer_bank      (always the other one)
//
//   bank_free_at[0] and bank_free_at[1] track each half independently.
//   A read stalls only if consumer_bank is still being written.
//   A write stalls only if producer_bank is still being read.
//
// ACCESS LATENCY
// ──────────────
//   latency = ceil(bytes / banking_factor)
//   (same formula as dma_stage; banking_factor = parallel SRAM ports)
// ---------------------------------------------------------------------------
class BufferUnit : public Unit {
public:
    BufferUnit(std::string name, const SramConfig& cfg,
               Scheduler*    sched = nullptr,
               TensorStore*  ts    = nullptr,
               std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_    = ts; }

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle access_latency(uint64_t bytes) const;

    // Expose bank state for logging / tests
    Cycle bank_free_at(int bank) const { return bank_free_at_[bank]; }
    void  reset_banks() { bank_free_at_[0] = bank_free_at_[1] = 0;
                          write_count_ = 0; }

private:
    // Pick which double-buffer half this access targets.
    // Writes always go to producer half; reads go to consumer half.
    int select_bank(bool is_write);

    SramConfig           cfg_;
    Scheduler*           sched_;
    TensorStore*         ts_;
    std::ostream&        os_;
    std::array<Cycle, 2> bank_free_at_ = {0, 0};
    uint32_t             write_count_  = 0;
};

}  // namespace sim
