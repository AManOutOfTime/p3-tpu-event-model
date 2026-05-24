#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {
class Scheduler;

struct GemmShape {
    uint32_t    M=0, K=0, N=0;
    std::string src_a;   // TensorStore key for A (empty = timing-only)
    std::string src_b;   // TensorStore key for B
    std::string dst_c;   // TensorStore key for C output
};

// ---------------------------------------------------------------------------
// SystolicUnit — weight-stationary systolic array (unidir or bidir).
//
// TIMING:  per_tile = K + fill_latency
//          fill_latency = (rows-1)+(cols-1)          [unidir]
//                       = ceil((rows-1)/2)+ceil((cols-1)/2)  [bidir]
//          total = tiles_m × tiles_n × per_tile
//
// COMPUTE: if src_a/src_b/dst_c are set AND a TensorStore is attached,
//          do_gemm() runs the actual tiled float MAC at OP_DONE.
//          Otherwise the unit is timing-only (backward compatible).
// ---------------------------------------------------------------------------
class SystolicUnit : public Unit {
public:
    SystolicUnit(std::string name, const SystolicConfig& cfg,
                 Scheduler*    sched = nullptr,
                 TensorStore*  ts    = nullptr,
                 std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_    = ts; }

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle fill_latency() const;
    Cycle compute_latency(uint32_t M, uint32_t K, uint32_t N) const;
    const SystolicConfig& config() const { return cfg_; }

private:
    void do_gemm(const GemmShape& shape);
    SystolicConfig cfg_;
    Scheduler*     sched_;
    TensorStore*   ts_;
    std::ostream&  os_;
};
}  // namespace sim
