#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>
#include <limits>

namespace sim {
class Scheduler;

// ---------------------------------------------------------------------------
// AccessOp — payload for access_core operations.
//
// Payload fields are symbolic. The access unit models latency only.
// ---------------------------------------------------------------------------
struct AccessOp {
    std::string kind;           // "init_fill" | "transpose" | "copy"
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
// AccessUnit — transpose / init-fill / copy timing core.
//
// TIMING:  latency = ceil(elements / bandwidth)
// The unit never mutates or computes buffer contents.
// ---------------------------------------------------------------------------
class AccessUnit : public Unit {
public:
    AccessUnit(std::string name, const AccessCoreConfig& cfg,
               Scheduler*    sched = nullptr,
               std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle compute_latency(uint64_t elements) const;

private:
    AccessCoreConfig cfg_;
    Scheduler*       sched_;
    std::ostream&    os_;
};

}  // namespace sim
