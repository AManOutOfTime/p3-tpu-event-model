#pragma once

#include "engine/event.h"

#include <functional>
#include <queue>
#include <vector>

namespace tpu {

// Callback invoked for every event as it is dispatched. Phase 0 uses this
// for the "print to terminal" handler; later phases will register per-kind
// handlers that update unit state, push follow-on events, etc.
using EventHandler = std::function<void(const Event&)>;

class EventEngine {
public:
    EventEngine() = default;

    // Schedule an event. The engine stamps it with a fresh sequence number
    // so that the caller does not have to reason about tiebreaking.
    // Returns the assigned Seq, mostly useful for tests / tracing.
    Seq schedule(Event e);

    // Convenience overload for the common case.
    Seq schedule(Tick time, UnitId unit, EventKind kind, std::string label = {});

    // Register the handler invoked for every dispatched event. Overwrites
    // any previously registered handler. A single handler is enough for
    // Phase 0 — we are not yet routing by kind.
    void set_handler(EventHandler h) { handler_ = std::move(h); }

    // Drain the queue, dispatching events in (time, unit, seq) order.
    // `now_` advances monotonically to each event's time as it is popped,
    // so handlers can call `now()` to get the current simulated time.
    // Returns the number of events processed.
    std::size_t run();

    // Like run() but stops (without dispatching) the first event whose
    // time is strictly greater than `until`. Useful for stepping.
    std::size_t run_until(Tick until);

    Tick        now()   const noexcept { return now_; }
    std::size_t pending() const noexcept { return queue_.size(); }

private:
    using Queue = std::priority_queue<Event, std::vector<Event>, EventLater>;

    Queue         queue_{};
    EventHandler  handler_{};
    Tick          now_{0};
    Seq           next_seq_{0};
};

} // namespace tpu
