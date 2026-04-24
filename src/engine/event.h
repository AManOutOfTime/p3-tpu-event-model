#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace tpu {

// Simulator time is an integer count of picoseconds. Using a signed 64-bit
// integer gives ~292 years of headroom, which is more than enough for any
// workload we will simulate and avoids any floating-point drift in the
// priority queue ordering.
using Tick = std::int64_t;

// Stable, self-describing identifier for a hardware unit (systolic array,
// vector unit, sparse core, DMA engine, HBM channel, etc). Phase 0 only
// cares that these are unique and comparable; later phases will map them
// to real resource models.
using UnitId = std::uint32_t;

// A monotonically increasing sequence number assigned at schedule() time.
// It is the tiebreaker when two events share the same Tick, and together
// with UnitId it guarantees a total, deterministic order in the priority
// queue. Without it, std::priority_queue would give nondeterministic
// schedules whenever two events collide on time.
using Seq = std::uint64_t;

enum class EventKind : std::uint16_t {
    // Phase 0 only uses these two for the dummy schedule. Later phases
    // will extend this enum with UnitStart/UnitFinish, BufferPushDone,
    // BufferPopDone, DMADone, BufferSwap, etc.
    Tick,        // generic "something happened at t" placeholder
    PrintMessage // dummy event that just prints payload.label to stdout
};

struct Event {
    Tick       time   {0};    // when the event fires (picoseconds)
    UnitId     unit   {0};    // which unit produced / owns this event
    EventKind  kind   {EventKind::Tick};
    Seq        seq    {0};    // assigned by the engine, do not set manually
    // Minimal payload for Phase 0. Later we'll replace this with a tagged
    // union or std::variant carrying kind-specific data (tile ids, bytes,
    // src/dst buffers, ...). A std::string is fine for a "print this"
    // dummy event and keeps the scaffolding obvious.
    std::string label;
};

// Ordering used by the engine's priority queue. Earliest time wins;
// ties are broken by UnitId then Seq so the schedule is fully deterministic.
// std::priority_queue is a max-heap, so "less urgent" must compare greater.
struct EventLater {
    bool operator()(const Event& a, const Event& b) const noexcept {
        if (a.time != b.time) return a.time > b.time;
        if (a.unit != b.unit) return a.unit > b.unit;
        return a.seq > b.seq;
    }
};

std::string_view to_string(EventKind k) noexcept;

} // namespace tpu
