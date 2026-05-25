#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
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
// Buffer fields used per op (all are TensorStore keys):
//
//  scale         src, dst, src_scale (optional row-vector; empty = scalar)
//  rowmax        src, dst            [Br×Bc] → [Br]
//  update_rowmax src_m, src_rowmax, dst_m, dst_correction
//  exp_shift     src_matrix, src_shift, dst   P = exp(S - m) broadcast
//  update_rowsum src_p, src_correction, src_l, dst_l
//  accumulate    src_a, src_b, dst   element-wise dst = src_a + src_b
//  normalize     src_matrix, src_denom, dst   row-wise divide
//  logsumexp     src_m, src_l, dst   element-wise dst = m + log(l)
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
};

// ---------------------------------------------------------------------------
// VectorUnit — SIMD vector / tandem core.
// ---------------------------------------------------------------------------
class VectorUnit : public Unit {
public:
    VectorUnit(std::string name, const VectorCoreConfig& cfg,
               Scheduler*    sched = nullptr,
               TensorStore*  ts    = nullptr,
               std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_    = ts; }

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle compute_latency(const VectorOp& op) const;

private:
    void do_scale        (const VectorOp& op);
    void do_rowmax       (const VectorOp& op);
    void do_update_rowmax(const VectorOp& op);
    void do_exp_shift    (const VectorOp& op);
    void do_update_rowsum(const VectorOp& op);
    void do_accumulate   (const VectorOp& op);
    void do_normalize    (const VectorOp& op);
    void do_logsumexp    (const VectorOp& op);

    VectorCoreConfig cfg_;
    Scheduler*       sched_;
    TensorStore*     ts_;
    std::ostream&    os_;
};

}  // namespace sim
