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
// Op kinds and their latency parameters (all share the same formula):
//   latency = passes × ceil(elements / simd_width)
//           + exp_ops × exp_latency × ceil(elements / simd_width)
//
//   kind            passes  exp_ops  what it computes
//   ─────────────── ──────  ───────  ──────────────────────────────────────
//   scale           1       0        x * scalar  (or x *= row_vector)
//   rowmax          1       0        max over cols → length-Br vector
//   update_rowmax   1       1        m=max(m,r); correction=exp(m_old-m_new)
//   exp_shift       1       1        P = exp(S - m_new)  [broadcast m]
//   update_rowsum   1       0        l = correction*l_old + rowsum(P)
//   accumulate      1       0        A += B  element-wise
//   normalize       1       0        A / b   row-wise divide by vector
//   logsumexp       1       1        L = m + log(l)  [log = exp cost]
//
// The exp_ops=1 entries call exp() or log() once per SIMD group, which costs
// exp_latency extra cycles per group (transcendental unit pipeline depth).
// ---------------------------------------------------------------------------
struct VectorOp {
    std::string kind;
    uint64_t    elements = 0;
    uint32_t    passes   = 1;
    uint32_t    exp_ops  = 0;
};

class VectorUnit : public Unit {
public:
    VectorUnit(std::string name, const VectorCoreConfig& cfg,
               Scheduler*    sched = nullptr,
               std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }
    void handle(const Event& e, EventEngine& engine) override;
    Cycle compute_latency(const VectorOp& op) const;

private:
    VectorCoreConfig cfg_;
    Scheduler*       sched_;
    std::ostream&    os_;
};

}  // namespace sim
