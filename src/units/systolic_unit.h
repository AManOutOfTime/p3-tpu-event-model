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
    std::string src_a;   // symbolic A buffer key
    std::string src_b;   // symbolic B buffer key
    std::string dst_c;   // symbolic C buffer key
};

// ---------------------------------------------------------------------------
// SystolicUnit — weight-stationary systolic array (unidir or bidir).
//
// TIMING:  per_tile = K + fill_latency
//          fill_latency = (rows-1)+(cols-1)          [unidir]
//                       = ceil((rows-1)/2)+ceil((cols-1)/2)  [bidir]
//          Oversized logical GEMMs must be decomposed into physical executions
//          by Tiler::decompose()/expand_gemm_subtiles before reaching this unit.
//
// The unit is event/timing-only. It never computes GEMM output values.
// ---------------------------------------------------------------------------
class SystolicUnit : public Unit {
public:
    SystolicUnit(std::string name, const SystolicConfig& cfg,
                 Scheduler*    sched = nullptr,
                 TensorStore*  ts    = nullptr,
                 std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore*) {}

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle fill_latency() const;
    Cycle compute_latency(uint32_t M, uint32_t K, uint32_t N) const;
    const SystolicConfig& config() const { return cfg_; }

private:
    SystolicConfig cfg_;
    Scheduler*     sched_;
    std::ostream&  os_;
};
}  // namespace sim
