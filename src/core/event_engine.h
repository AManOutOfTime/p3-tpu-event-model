#pragma once
#include "core/event.h"
#include "core/unit.h"
#include <functional>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace sim {

struct UnitReservation {
    UnitId id = INVALID_UNIT;
    Cycle  start = 0;
};

// Cycle-based discrete event engine. Owns the simulation clock, the priority
// queue of pending events, and the registered units. Time advances by jumping
// to the next event's cycle -- there is no per-cycle tick loop.
class EventEngine {
public:
    explicit EventEngine(double clock_ghz = 1.0) : clock_ghz_(clock_ghz) {}

    // ---- Unit registration --------------------------------------------------
    // Engine takes ownership and assigns the unit's id. The returned id can
    // also be retrieved later via find_unit(name).
    UnitId  register_unit(std::unique_ptr<Unit> unit, uint64_t buffer_capacity_bytes = 0);
    Unit*   get_unit(UnitId id) const;
    UnitId  find_unit(const std::string& name) const;
    std::vector<UnitId> find_unit_pool(const std::string& logical_name) const;
    size_t  num_units() const { return units_.size(); }

    // ---- Hardware resource table -------------------------------------------
    // Tracks per-physical-unit timing and scratch-buffer usage. Placement picks
    // the unit with the smallest current buffer use, then earliest start cycle.
    UnitReservation reserve_unit_pool(const std::vector<UnitId>& ids,
                                      Cycle duration,
                                      uint64_t buffer_bytes);
    void     release_unit_buffer(UnitId id, uint64_t buffer_bytes);
    uint64_t unit_buffer_used(UnitId id) const;
    uint64_t unit_buffer_capacity(UnitId id) const;

    // ---- Event scheduling ---------------------------------------------------
    // Insert an event. If e.seq == 0 the engine assigns a fresh seq for stable
    // ordering. Throws if the event is in the past (cycle < current_cycle).
    EventId schedule(Event e);

    // Convenience: schedule an event 'delta' cycles from now.
    EventId schedule_after(Cycle delta, Event e);

    // ---- Driving the simulation --------------------------------------------
    // Drain the queue, optionally stopping when the next event would fire
    // after stop_at. Returns the cycle of the last event actually dispatched.
    Cycle run(Cycle stop_at = CYCLE_MAX);

    // ---- Introspection ------------------------------------------------------
    Cycle  current_cycle() const { return now_; }
    double clock_ghz()     const { return clock_ghz_; }
    size_t pending()       const { return queue_.size(); }
    EventId next_seq()           { return next_seq_++; }

    // Optional callback fired before every dispatched event. Useful for
    // tracing/logging in tests and the CLI driver.
    using TraceFn = std::function<void(const Event&)>;
    void set_trace(TraceFn fn) { trace_ = std::move(fn); }

private:
    using MinHeap = std::priority_queue<Event, std::vector<Event>, std::greater<Event>>;

    struct HardwareState {
        Cycle    available_at = 0;
        uint64_t buffer_capacity_bytes = 0;
        uint64_t buffer_used_bytes = 0;
    };

    double  clock_ghz_;
    Cycle   now_      = 0;
    EventId next_seq_ = 1;
    MinHeap queue_;
    std::vector<std::unique_ptr<Unit>>  units_;
    std::vector<HardwareState>          hardware_;
    std::unordered_map<std::string, UnitId> name_index_;
    TraceFn trace_;
};

}  // namespace sim
