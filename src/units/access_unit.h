#pragma once
#include "core/unit.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>
#include <limits>

namespace sim {
class Scheduler;

// ---------------------------------------------------------------------------
// AccessOp — payload for access_core operations.
//
// init_fill:
//   Writes fill_value to every element of dst.
//   dst must already exist in TensorStore (pre-allocated).
//   elements = total element count.
//
// transpose:
//   Reads src [input_rows × input_cols], writes dst [input_cols × input_rows].
//   Row-major in/out.
// ---------------------------------------------------------------------------
struct AccessOp {
    std::string kind;           // "init_fill" | "transpose"
    uint64_t    elements = 0;

    // init_fill fields
    std::string dst;            // destination buffer key
    float       fill_value = 0.f;

    // transpose fields
    std::string src;            // source buffer key
    // dst shared with init_fill
    uint32_t    input_rows  = 0;
    uint32_t    input_cols  = 0;
};

// ---------------------------------------------------------------------------
// AccessUnit — transpose / init-fill core.
//
// TIMING:  latency = ceil(elements / bandwidth)
// COMPUTE: fires at OP_DONE if TensorStore is attached.
// ---------------------------------------------------------------------------
class AccessUnit : public Unit {
public:
    AccessUnit(std::string name, const AccessCoreConfig& cfg,
               Scheduler*    sched = nullptr,
               TensorStore*  ts    = nullptr,
               std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s)       { sched_ = s; }
    void set_tensor_store(TensorStore* ts) { ts_    = ts; }

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle compute_latency(uint64_t elements) const;

private:
    void do_init_fill(const AccessOp& op);
    void do_transpose(const AccessOp& op);

    AccessCoreConfig cfg_;
    Scheduler*       sched_;
    TensorStore*     ts_;
    std::ostream&    os_;
};

}  // namespace sim
