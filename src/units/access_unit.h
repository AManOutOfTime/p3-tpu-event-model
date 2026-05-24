#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {
class Scheduler;

// ---------------------------------------------------------------------------
// AccessOp — payload for access_core operations.
//
// Op kinds:
//   init_fill   Fill a buffer with a constant (0, -inf, 1, etc.)
//               latency = ceil(elements / bandwidth)
//
//   transpose   Transpose a matrix tile in SRAM.
//               latency = ceil(rows × cols / bandwidth)
//               The transposed result is written to a separate destination.
//
// Both share the same latency formula — they differ only in data movement
// pattern, not bandwidth.
// ---------------------------------------------------------------------------
struct AccessOp {
    std::string kind;          // "init_fill" | "transpose"
    uint64_t    elements = 0;  // total elements processed
};

class AccessUnit : public Unit {
public:
    AccessUnit(std::string name, const AccessCoreConfig& cfg,
               Scheduler*    sched = nullptr,
               std::ostream& os    = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }
    void handle(const Event& e, EventEngine& engine) override;
    Cycle compute_latency(uint64_t elements) const;

private:
    AccessCoreConfig cfg_;
    Scheduler*       sched_;
    std::ostream&    os_;
};

}  // namespace sim
