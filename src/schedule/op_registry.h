#pragma once
#include "schedule/instruction.h"
#include <functional>
#include <string>
#include <unordered_map>

namespace sim {

class EventEngine;
class Scheduler;

struct IssueCtx {
    EventEngine&       engine;
    Scheduler&         scheduler;
    const Instruction& inst;
};

using OpHandler = std::function<void(const IssueCtx&)>;

class OpRegistry {
public:
    void             register_op(const std::string& name, OpHandler handler);
    bool             has(const std::string& name) const;
    const OpHandler& get(const std::string& name) const;
    // Fast path: integer key, no string hash.  Used by the Scheduler hot loop.
    const OpHandler& get(uint8_t code) const;

private:
    std::unordered_map<std::string, OpHandler> ops_;
    std::unordered_map<uint8_t,    OpHandler>  fast_ops_;
};

}  // namespace sim