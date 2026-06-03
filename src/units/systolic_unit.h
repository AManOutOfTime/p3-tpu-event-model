#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include <iostream>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// GemmShape — GEMM dimensions + optional tensor buffer names.
//
//   M, K, N       : matrix dimensions  C[M×N] = A[M×K] × B[K×N]
//   src_a, src_b  : names of A and B buffers in a TensorStore
//   dst_c         : name of the output C buffer
//
//   If src_a / src_b / dst_c are non-empty AND a TensorStore is attached to
//   the SystolicUnit, the unit will compute actual float values (C = A×B) on
//   OP_DONE, in addition to modelling the cycle-accurate latency.
//
//   If the names are empty (default), the unit is timing-only.
// ---------------------------------------------------------------------------
struct GemmShape {
    uint32_t    M = 0, K = 0, N = 0;
    std::string src_a;    // TensorStore key for A [M×K]  (empty = timing-only)
    std::string src_b;    // TensorStore key for B [K×N]
    std::string dst_c;    // TensorStore key for C [M×N]
};

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
//   OP_DONE  → optional do_gemm → scheduler.notify_done(instr)
// ---------------------------------------------------------------------------
class SystolicUnit : public Unit {
public:
    SystolicUnit(std::string name, const SystolicConfig& cfg,
                 Scheduler*    sched = nullptr,
                 TensorStore*  ts    = nullptr,
                 std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_ = ts; }

    void handle(const Event& e, EventEngine& engine) override;

    Cycle fill_latency() const;
    Cycle compute_latency(uint32_t M, uint32_t K, uint32_t N) const;

    const SystolicConfig& config() const { return cfg_; }

private:
    // Blocked float GEMM: C[M×N] = A[M×K] × B[K×N].  Reads/writes ts_.
    void do_gemm(const GemmShape& shape);

    SystolicConfig cfg_;
    Scheduler*     sched_;
    TensorStore*   ts_;
    std::ostream&  os_;
};

}  // namespace sim
