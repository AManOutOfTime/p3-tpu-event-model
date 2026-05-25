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
// WeightLoad — payload for weight_load op.
//
// In weight-stationary mode, K_tile_T must be distributed into every PE's
// weight register BEFORE the GEMM streaming begins.  This is a broadcast
// write from IBUF to all SA_rows × SA_cols PEs.
//
// Latency = ceil(SA_rows × SA_cols × dtype_bytes / sram.banking_factor)
//
// Example: 128×128 array, BF16 (2 bytes), banking_factor=8:
//   = ceil(128 × 128 × 2 / 8) = ceil(4096) = 4096 cycles
//
// This is the pre-loading cost that was previously assumed to be zero.
// ---------------------------------------------------------------------------
struct WeightLoad {
    uint32_t    sa_rows    = 0;
    uint32_t    sa_cols    = 0;
    uint32_t    dtype_bytes= 2;
    uint32_t    banking_factor = 8;
    std::string src_buf;   // TensorStore key for the weight tile
    std::string dst_buf;   // destination key (e.g. "systolic_array.weight_reg")
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

    // Weight-stationary pre-load: distribute weight tile into PE registers.
    // Cost = ceil(SA_rows × SA_cols × dtype_bytes / banking_factor)
    static Cycle weight_load_latency(uint32_t sa_rows, uint32_t sa_cols,
                                     uint32_t dtype_bytes, uint32_t banking_factor);

    const SystolicConfig& config() const { return cfg_; }

private:
    void do_gemm(const GemmShape& shape);
    void do_weight_load(const WeightLoad& wl);

    SystolicConfig cfg_;
    Scheduler*     sched_;
    TensorStore*   ts_;
    std::ostream&  os_;
};
}  // namespace sim
