#pragma once
#include "core/types.h"
#include <any>
#include <string>

namespace sim {

// Event types. Add to this enum when a new built-in event class is needed.
// For module-specific events, prefer payload-typed CUSTOM events to avoid
// constantly editing this header.
enum class EventType {
    OP_START,        // begin executing an operation on a unit
    OP_DONE,         // operation finished
    DMA_DONE,        // a DMA transfer completed
    BUFFER_SWAP,     // double-buffer flip
    BARRIER,         // multi-unit synchronization point
    CUSTOM,          // user-defined; meaning is carried in payload
};

inline const char* to_string(EventType t) {
    switch (t) {
        case EventType::OP_START:    return "OP_START";
        case EventType::OP_DONE:     return "OP_DONE";
        case EventType::DMA_DONE:    return "DMA_DONE";
        case EventType::BUFFER_SWAP: return "BUFFER_SWAP";
        case EventType::BARRIER:     return "BARRIER";
        case EventType::CUSTOM:      return "CUSTOM";
    }
    return "UNKNOWN";
}

// Single event in the simulation queue. Sorted in the engine by (cycle, seq)
// so events at the same cycle still fire in insertion order -- this gives
// deterministic, reproducible runs across hosts.
struct Event {
    Cycle           cycle  = 0;
    EventId         seq    = 0;            // engine-assigned tie-breaker
    EventType       type   = EventType::CUSTOM;
    UnitId          target = INVALID_UNIT;
    InstructionId   instr  = 0;            // schedule instruction id (0 = none)
    std::string     label;                 // human-readable, optional
    std::any        payload;               // arbitrary user data

    // Min-heap ordering: smaller (cycle, seq) is "less", which a std::greater
    // priority_queue treats as higher priority -> earliest first.
    bool operator>(const Event& o) const {
        if (cycle != o.cycle) return cycle > o.cycle;
        return seq > o.seq;
    }
};

}  // namespace sim
