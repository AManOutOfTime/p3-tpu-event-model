#include "core/event_engine.h"
#include <stdexcept>

namespace sim {

UnitId EventEngine::register_unit(std::unique_ptr<Unit> unit) {
    UnitId id = static_cast<UnitId>(units_.size());
    unit->id_ = id;                           // assigned via friend
    name_index_[unit->name()] = id;
    units_.push_back(std::move(unit));
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
