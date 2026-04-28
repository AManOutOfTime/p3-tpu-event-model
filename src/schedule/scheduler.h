#pragma once
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "core/event_engine.h"
#include <unordered_map>
#include <unordered_set>

namespace sim {

// Drives a Schedule on an EventEngine via an OpRegistry.
// Tracks which instructions are done and issues newly-unblocked ones.
//
// Typical usage:
//   Scheduler sched(engine, registry, schedule);
//   sched.launch();        // issues all initially-ready instructions
//   engine.run();          // drains event queue; units call notify_done()
//   assert(sched.all_done());
class Scheduler {
public:
    Scheduler(EventEngine& engine, OpRegistry& registry, Schedule schedule);

    // Issue all instructions with no unsatisfied dependencies.
    void launch();

    // Called by op handlers or units when instruction `id` is finished.
    // Automatically issues any dependents that are now unblocked.
    void notify_done(InstructionId id);

    bool   all_done()    const { return done_count_ == sched_.instructions.size(); }
    size_t outstanding() const { return sched_.instructions.size() - done_count_; }

private:
    void try_issue(InstructionId id);

    EventEngine& engine_;
    OpRegistry&  registry_;
    Schedule     sched_;

    std::unordered_map<InstructionId, int>                        remaining_deps_;
    std::unordered_map<InstructionId, std::vector<InstructionId>> successors_;
    std::unordered_set<InstructionId>                             issued_;
    size_t done_count_ = 0;
};

}  // namespace sim
