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

    // ---- Metrics (P0.2) -----------------------------------------------------
    // Per-unit cycles spent reserved (occupancy). Accumulated in reserve_unit_pool.
    Cycle    unit_busy_cycles(UnitId id) const;
    // Global counters incremented by op handlers at issue time.
    void     add_macs(uint64_t n)      { total_macs_ += n; }
    void     add_hbm_bytes(uint64_t n) { total_hbm_bytes_ += n; }
    uint64_t total_macs()      const { return total_macs_; }
    uint64_t total_hbm_bytes() const { return total_hbm_bytes_; }

    // ---- SRAM-pressure accounting (P1.2) -----------------------------------
    // Shared IBUF+OBUF working-set tracker. Capacity 0 == unlimited (default).
    void     set_sram_capacity(uint64_t bytes) { sram_capacity_bytes_ = bytes; }
    uint64_t sram_capacity()  const { return sram_capacity_bytes_; }
    uint64_t sram_used()      const { return sram_used_bytes_; }
    uint64_t sram_peak()      const { return sram_peak_bytes_; }
    uint64_t sram_spills()    const { return sram_spills_; }
    // Acquire `bytes` of SRAM working set. Returns true if it fit; false if it
    // overflowed capacity (caller models a spill). Always tracks peak usage.
    bool     sram_acquire(uint64_t bytes);
    void     sram_release(uint64_t bytes);

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
        Cycle    busy_cycles = 0;          // P0.2: total reserved occupancy
    };

    double   clock_ghz_;
    Cycle    now_      = 0;
    EventId  next_seq_ = 1;
    uint64_t total_macs_      = 0;         // P0.2
    uint64_t total_hbm_bytes_ = 0;         // P0.2
    uint64_t sram_capacity_bytes_ = 0;     // P1.2 (0 == unlimited)
    uint64_t sram_used_bytes_     = 0;
    uint64_t sram_peak_bytes_     = 0;
    uint64_t sram_spills_         = 0;
    MinHeap queue_;
    std::vector<std::unique_ptr<Unit>>  units_;
    std::vector<HardwareState>          hardware_;
    std::unordered_map<std::string, UnitId> name_index_;
    TraceFn trace_;
};

}  // namespace sim
