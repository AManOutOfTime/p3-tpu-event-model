#pragma once
#include "schedule/instruction.h"
#include <functional>
#include <string>
#include <unordered_map>

namespace sim {

class EventEngine;
class Scheduler;

// Passed to every op handler when an instruction is issued.
struct IssueCtx {
    EventEngine&       engine;
    Scheduler&         scheduler;
    const Instruction& inst;
};

// An op handler must:
//   1. Schedule one or more events on the engine (across any units).
//   2. Eventually call scheduler.notify_done(inst.id) once the logical
//      operation is complete (usually from a unit's OP_DONE handler).
//      For zero-cycle ops, call notify_done() immediately inside the handler.
//
// This is what enables flexible granularity:
//   - Fine op  -> fires 1 event on 1 unit, notifies when that event fires.
//   - Coarse op -> fires N events across M units, notifies after the last.
using OpHandler = std::function<void(const IssueCtx&)>;

// Global registry: op-name string -> handler.
// Register once at startup; the Scheduler looks ops up at issue time.
// To add a new op: call registry.register_op("my_op", my_handler);
class OpRegistry {
public:
    void             register_op(const std::string& name, OpHandler handler);
    bool             has(const std::string& name) const;
    const OpHandler& get(const std::string& name) const;

private:
    std::unordered_map<std::string, OpHandler> ops_;
};

}  // namespace sim
