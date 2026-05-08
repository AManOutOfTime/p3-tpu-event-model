#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// AccessOp — payload for transpose / scatter / gather / init_fill ops.
// ---------------------------------------------------------------------------
struct AccessOp {
    uint64_t    elements = 0;   // total elements to move or transform
    std::string kind     = "";  // "transpose" | "scatter" | "gather" | "init_fill"
};

// ---------------------------------------------------------------------------
// AccessUnit — models an access core (transpose, scatter/gather, init-fill).
//
// TIMING MODEL
// ─────────────────────────────────────────────────────────────────────────
//   latency = ceil(elements / access_bandwidth)
//
//   access_bandwidth: elements moved/transformed per cycle (from config).
//
//   Worked examples (bandwidth=64):
//     Transpose K [128×128 = 16384 elems]  → ceil(16384/64) = 256 cycles
//     Init fill  O [128×128 = 16384 elems] → 256 cycles
//     Scatter    [4096 elems]              →  64 cycles
//
//   All four operations (transpose, scatter, gather, init_fill) are modeled
//   identically — they differ only in data flow, not in on-chip bandwidth.
//
// BACKWARD COMPATIBILITY
//   int64_t payload → latency_cycles (op: delay).
//   AccessOp payload → latency from AccessCoreConfig.bandwidth.
// ---------------------------------------------------------------------------
class AccessUnit : public Unit {
public:
    AccessUnit(std::string name, const AccessCoreConfig& cfg,
               Scheduler* sched = nullptr, std::ostream& os = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }
    void handle(const Event& e, EventEngine& engine) override;

    Cycle compute_latency(uint64_t elements) const;

private:
    AccessCoreConfig  cfg_;
    Scheduler*        sched_;
    std::ostream&     os_;
};

}  // namespace sim
