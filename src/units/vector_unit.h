#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// VectorOp — payload for all vector/tandem core operations.
//
// Operations and their parameters:
//
//   kind        passes  exp_ops  description
//   ──────────  ──────  ───────  ──────────────────────────────────────
//   "scale"       1       0      element-wise scalar multiply
//   "add"         1       0      element-wise add (accumulate, rescale)
//   "exp"         1       1      element-wise exp(x)
//   "rowmax"      1       0      row-max reduction (Br passes × Bc)
//   "rowsum"      1       0      row-sum reduction
//   "softmax"     3       1      rowmax + exp+sum + normalize (fused)
//   "layer_norm"  2       0      mean+var pass + normalize pass
//   "rope"        1       0      complex multiply (RoPE encoding)
//   "normalize"   1       0      element-wise divide by scalar
//   "logsumexp"   1       1      m + log(l) — log has same cost as exp
//
// TIMING MODEL
// ─────────────────────────────────────────────────────────────────────────
//   simd_cycles = ceil(elements / simd_width) per pass
//   exp_cycles  = exp_ops × exp_latency × ceil(elements / simd_width)
//   total       = passes × simd_cycles + exp_cycles
//
//   For reductions (rowmax, rowsum): elements = rows × cols (full matrix).
//   For softmax:   elements = rows × cols, passes=3, exp_ops=1.
// ---------------------------------------------------------------------------
struct VectorOp {
    uint64_t    elements = 0;     // total number of elements
    uint32_t    passes   = 1;     // number of linear passes over elements
    uint32_t    exp_ops  = 0;     // number of exp/log calls per SIMD group
    std::string kind     = "";    // human label for logging
};

// ---------------------------------------------------------------------------
// VectorUnit — models a tandem (vector) core.
//
// Handles: scale, exp, softmax, layer_norm, RoPE, rowmax, rowsum,
//          accumulate, normalize, logsumexp.
//
// BACKWARD COMPATIBILITY
//   int64_t payload → use as latency (op: delay).
//   VectorOp payload → compute from VectorCoreConfig.
// ---------------------------------------------------------------------------
class VectorUnit : public Unit {
public:
    VectorUnit(std::string name, const VectorCoreConfig& cfg,
               Scheduler* sched = nullptr, std::ostream& os = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }
    void handle(const Event& e, EventEngine& engine) override;

    // Compute cycles for a VectorOp.
    Cycle compute_latency(const VectorOp& op) const;

private:
    VectorCoreConfig  cfg_;
    Scheduler*        sched_;
    std::ostream&     os_;
};

}  // namespace sim
