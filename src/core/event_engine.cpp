#include "core/event_engine.h"
#include <stdexcept>
#include <algorithm>

namespace sim {

UnitId EventEngine::register_unit(std::unique_ptr<Unit> unit, uint64_t buffer_capacity_bytes) {
    UnitId id = static_cast<UnitId>(units_.size());
    unit->id_ = id;                           // assigned via friend
    name_index_[unit->name()] = id;
    units_.push_back(std::move(unit));
    hardware_.push_back(HardwareState{0, buffer_capacity_bytes, 0});
    return id;
}

Unit* EventEngine::get_unit(UnitId id) const {
    if (id == INVALID_UNIT || id >= static_cast<UnitId>(units_.size()))
        return nullptr;
    return units_[id].get();
}

UnitId EventEngine::find_unit(const std::string& name) const {
    auto it = name_index_.find(name);
    return it == name_index_.end() ? INVALID_UNIT : it->second;
}

std::vector<UnitId> EventEngine::find_unit_pool(const std::string& logical_name) const {
    std::vector<UnitId> ids;
    const std::string prefix = logical_name + "_";

    for (UnitId id = 0; id < static_cast<UnitId>(units_.size()); id++) {
        const std::string& name = units_[id]->name();
        if (name.rfind(prefix, 0) == 0)
            ids.push_back(id);
    }

    if (!ids.empty())
        return ids;

    UnitId exact = find_unit(logical_name);
    if (exact != INVALID_UNIT)
        ids.push_back(exact);
    return ids;
}

UnitReservation EventEngine::reserve_unit_pool(const std::vector<UnitId>& ids,
                                               Cycle duration,
                                               uint64_t buffer_bytes) {
    if (ids.empty())
        throw std::runtime_error("EventEngine::reserve_unit_pool: empty unit pool");

    UnitReservation best;
    best.start = CYCLE_MAX;

    for (UnitId id : ids) {
        if (id == INVALID_UNIT || id >= static_cast<UnitId>(hardware_.size()))
            continue;

        const HardwareState& state = hardware_[id];
        if (state.buffer_capacity_bytes > 0 &&
            state.buffer_used_bytes + buffer_bytes > state.buffer_capacity_bytes)
            continue;

        Cycle start = now_ > state.available_at ? now_ : state.available_at;
        if (best.id == INVALID_UNIT ||
            state.buffer_used_bytes < hardware_[best.id].buffer_used_bytes ||
            (state.buffer_used_bytes == hardware_[best.id].buffer_used_bytes &&
             (start < best.start || (start == best.start && id < best.id)))) {
            best.id = id;
            best.start = start;
        }
    }

    if (best.id == INVALID_UNIT)
        throw std::runtime_error("EventEngine::reserve_unit_pool: no unit has enough buffer capacity");

    HardwareState& chosen = hardware_[best.id];
    chosen.available_at = best.start + duration;
    chosen.buffer_used_bytes += buffer_bytes;
    return best;
}

void EventEngine::release_unit_buffer(UnitId id, uint64_t buffer_bytes) {
    if (id == INVALID_UNIT || id >= static_cast<UnitId>(hardware_.size()))
        return;
    HardwareState& state = hardware_[id];
    state.buffer_used_bytes = buffer_bytes > state.buffer_used_bytes
        ? 0
        : state.buffer_used_bytes - buffer_bytes;
}

uint64_t EventEngine::unit_buffer_used(UnitId id) const {
    if (id == INVALID_UNIT || id >= static_cast<UnitId>(hardware_.size()))
        return 0;
    return hardware_[id].buffer_used_bytes;
}

uint64_t EventEngine::unit_buffer_capacity(UnitId id) const {
    if (id == INVALID_UNIT || id >= static_cast<UnitId>(hardware_.size()))
        return 0;
    return hardware_[id].buffer_capacity_bytes;
}

EventId EventEngine::schedule(Event e) {
    if (e.cycle < now_)
        throw std::runtime_error(
            "EventEngine::schedule: event at cycle " + std::to_string(e.cycle) +
            " is in the past (now=" + std::to_string(now_) + ")");
    if (e.seq == 0) e.seq = next_seq_++;
    EventId id = e.seq;
    queue_.push(std::move(e));
    return id;
}

EventId EventEngine::schedule_after(Cycle delta, Event e) {
    e.cycle = now_ + delta;
    return schedule(std::move(e));
}

Cycle EventEngine::run(Cycle stop_at) {
    while (!queue_.empty()) {
        if (queue_.top().cycle > stop_at) break;

        Event e = queue_.top();
        queue_.pop();
        now_ = e.cycle;

        if (trace_) trace_(e);

        Unit* u = get_unit(e.target);
        if (u) u->handle(e, *this);
    }
    return now_;
}

}  // namespace sim
