#include "engine/engine.h"

#include <stdexcept>

namespace tpu {

std::string_view to_string(EventKind k) noexcept {
    switch (k) {
        case EventKind::Tick:         return "Tick";
        case EventKind::PrintMessage: return "PrintMessage";
    }
    return "Unknown";
}

Seq EventEngine::schedule(Event e) {
    // Reject events scheduled in the past. This catches the common bug
    // where a handler computes `now() - delta` instead of `now() + delta`.
    if (e.time < now_) {
        throw std::invalid_argument("EventEngine::schedule: event in the past");
    }
    e.seq = next_seq_++;
    queue_.push(std::move(e));
    return e.seq;
}

Seq EventEngine::schedule(Tick time, UnitId unit, EventKind kind, std::string label) {
    Event e;
    e.time  = time;
    e.unit  = unit;
    e.kind  = kind;
    e.label = std::move(label);
    return schedule(std::move(e));
}

std::size_t EventEngine::run() {
    std::size_t n = 0;
    while (!queue_.empty()) {
        Event e = queue_.top();
        queue_.pop();
        now_ = e.time;
        if (handler_) handler_(e);
        ++n;
    }
    return n;
}

std::size_t EventEngine::run_until(Tick until) {
    std::size_t n = 0;
    while (!queue_.empty() && queue_.top().time <= until) {
        Event e = queue_.top();
        queue_.pop();
        now_ = e.time;
        if (handler_) handler_(e);
        ++n;
    }
    return n;
}

} // namespace tpu
