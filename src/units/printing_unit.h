#pragma once
#include "core/unit.h"
#include <iostream>

namespace sim {

// Simplest possible unit: prints every event it receives and does nothing else.
// Useful as a smoke-test target and as a template for new units.
//
// To add a real hardware unit, copy this file, rename the class, and replace
// handle() with logic that computes latency and schedules an OP_DONE event.
class PrintingUnit : public Unit {
public:
    explicit PrintingUnit(std::string name, std::ostream& os = std::cout)
        : Unit(std::move(name)), os_(os) {}

    void handle(const Event& e, EventEngine& /*engine*/) override;

private:
    std::ostream& os_;
};

}  // namespace sim
