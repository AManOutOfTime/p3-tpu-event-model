#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {
class Scheduler;

// ---------------------------------------------------------------------------
// VectorOp — payload for all vector_core operations.
//
// Timing formula (all ops):
//   groups  = ceil(elements / simd_width)
//   latency = passes × groups  +  exp_ops × exp_latency × groups
//
// Buffer fields used per op are symbolic keys for event labels/dependencies:
//
//  scale         src, dst, src_scale
//  rowmax        src, dst
//  update_rowmax src_m, src_rowmax, dst_m, dst_correction
//  exp_shift     src_matrix, src_shift, dst
//  update_rowsum src_p, src_correction, src_l, dst_l
//  accumulate    src_a, src_b, dst
//  normalize     src_matrix, src_denom, dst
//  logsumexp     src_m, src_l, dst
//  causal_mask   src, dst with row/col absolute starts
//  rope          src, dst
//  rmsnorm       src, dst
//  silu_mul      src_a, src_b, dst
//  residual_add  src_a, src_b, dst
// ---------------------------------------------------------------------------
struct VectorOp {
    std::string kind;
    uint64_t    elements  = 0;
    uint32_t    passes    = 1;
    uint32_t    exp_ops   = 0;
    uint32_t    rows      = 0;  // matrix row count (for broadcast ops)
    uint32_t    cols      = 0;  // matrix col count

    // Generic source/destination (scale, rowmax, accumulate, normalize)
    std::string src;
    std::string src_a;
    std::string src_b;
    std::string dst;

    // scale: optional row-vector to broadcast (empty = use scalar 1.0)
    std::string src_scale;

    // rowmax → update_rowmax chain
    std::string src_m;
    std::string src_rowmax;
    std::string dst_m;
    std::string dst_correction;

    // exp_shift: P = exp(S - m_broadcast)
    std::string src_matrix;
    std::string src_shift;   // length-Br vector broadcast over cols

    // update_rowsum
    std::string src_p;
    std::string src_correction;
    std::string src_l;
    std::string dst_l;

    // normalize / logsumexp share src_matrix / src_denom or src_m / src_l
    std::string src_denom;

    // Optional positional metadata for masking/RoPE schedule builders.
    uint32_t row_start = 0;
    uint32_t col_start = 0;
};

// ---------------------------------------------------------------------------
// VectorUnit — SIMD vector / tandem core.
// ---------------------------------------------------------------------------
class VectorUnit : public Unit {
public:
    VectorUnit(std::string name, const VectorCoreConfig& cfg,
               Scheduler*    sched = nullptr,
               std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle compute_latency(const VectorOp& op) const;

private:
    VectorCoreConfig cfg_;
    Scheduler*       sched_;
    std::ostream&    os_;
};

}  // namespace sim
