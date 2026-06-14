#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// GemmShape — GEMM dimensions passed as event payload.
//
//   M, K, N : matrix dimensions  C[M×N] = A[M×K] × B[K×N]
//   src_a, src_b, dst_c : symbolic buffer names (used in labels only)
// ---------------------------------------------------------------------------
struct GemmShape {
    uint32_t    M = 0, K = 0, N = 0;
    std::string src_a;
    std::string src_b;
    std::string dst_c;
    uint64_t    buffer_bytes  = 0;  // P1.2: SRAM working set held during the GEMM
    uint32_t    spill_penalty = 0;  // P1.2: cycles added if the set overflows SRAM
};

// ---------------------------------------------------------------------------
// Shared GEMM latency model (single source of truth — used by both SystolicUnit
// and the `gemm` op handler so the OP_DONE timing and the reservation duration
// can never desync).
//
//   systolic_fill_latency : pipeline fill/drain ((r-1)+(c-1), halved if bidir)
//   systolic_gemm_latency : full per-GEMM latency for the configured dataflow
// ---------------------------------------------------------------------------
Cycle systolic_fill_latency(const SystolicConfig& cfg);
Cycle systolic_gemm_latency(const SystolicConfig& cfg,
                            uint32_t M, uint32_t K, uint32_t N);

// ---------------------------------------------------------------------------
// SystolicUnit  —  2-D systolic array model (unidirectional OR bidirectional).
//
// ── UNIDIRECTIONAL (default, bidirectional=false) ────────────────────────
//   Data feeds from one edge (A: left→right, B: top→bottom).
//   The wavefront must traverse the full array before the last PE starts:
//
//     fill_latency = (SA_rows − 1) + (SA_cols − 1)
//
//   128×128: fill = 127 + 127 = 254 cycles
//
// ── BIDIRECTIONAL (bidirectional=true) ───────────────────────────────────
//   Data feeds from BOTH edges simultaneously (A: left→center + right→center,
//   B: top→center + bottom→center). The two wavefronts meet in the middle,
//   halving the fill latency:
//
//     fill_latency = ceil((SA_rows − 1)/2) + ceil((SA_cols − 1)/2)
//
//   128×128: fill = ceil(127/2) + ceil(127/2) = 64 + 64 = 128 cycles
//
//   This requires that each PE can receive and forward data in both directions,
//   which adds MUX/routing hardware but cuts the fill penalty ~2×.
//   The benefit is largest when fill_latency >> K (big array, small K).
//
// ── COMMON TILING MODEL ──────────────────────────────────────────────────
//   Array size  : SA_rows × SA_cols  (from SystolicConfig)
//   GEMM shape  : M × K × N
//   Tiles       : tiles_m = ceil(M/SA_rows)  ×  tiles_n = ceil(N/SA_cols)
//
//   per_tile = K + fill_latency
//   total    = tiles_m × tiles_n × per_tile
//
// ── EVENT PROTOCOL ───────────────────────────────────────────────────────
//   OP_START → decode GemmShape → compute_latency → schedule OP_DONE
//   OP_DONE  → scheduler.notify_done(instr)
// ---------------------------------------------------------------------------
class SystolicUnit : public Unit {
public:
    SystolicUnit(std::string name, const SystolicConfig& cfg,
                 Scheduler*    sched = nullptr,
                 std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }

    void handle(const Event& e, EventEngine& engine) override;

    Cycle fill_latency() const;
    Cycle compute_latency(uint32_t M, uint32_t K, uint32_t N) const;

    const SystolicConfig& config() const { return cfg_; }

private:
    SystolicConfig cfg_;
    Scheduler*     sched_;
    std::ostream&  os_;
};

}  // namespace sim
