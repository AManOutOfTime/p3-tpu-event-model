#pragma once
#include "core/event.h"
#include <string>

namespace sim {

class EventEngine;

// Base class for any hardware unit (systolic array, vector core, DMA, ...).
// To add a new unit, derive from this and implement handle(). Register the
// instance with EventEngine::register_unit() -- the engine assigns the id.
class Unit {
public:
    explicit Unit(std::string name) : name_(std::move(name)) {}
    virtual ~Unit() = default;

    UnitId             id()   const { return id_; }
    const std::string& name() const { return name_; }

    // Called by the engine when an event targeting this unit fires. The
    // engine's current_cycle() == event.cycle inside this call. The unit may
    // schedule new events on the engine.
    virtual void handle(const Event& e, EventEngine& engine) = 0;

private:
    friend class EventEngine;          // engine assigns id_ on registration
    UnitId      id_ = INVALID_UNIT;
    std::string name_;
};

}  // namespace sim
