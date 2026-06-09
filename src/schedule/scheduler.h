#pragma once
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "core/event_engine.h"
#include <vector>

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

    // Reserve hardware resources for an operation.
    Cycle reserve_unit(UnitId id, Cycle duration);
    UnitReservation reserve_unit_pool(const std::vector<UnitId>& ids, Cycle duration,
                                      uint64_t buffer_bytes = 0);

    bool   all_done()    const { return done_count_ == sched_.instructions.size(); }
    size_t outstanding() const { return sched_.instructions.size() - done_count_; }

private:
    void try_issue(InstructionId id);

    // Flatten InstructionId to a zero-based vector index.
    // In practice IDs are always 0..N-1 (the builders assign them sequentially),
    // so id_base_==0 and this is a no-op subtraction. The base is computed once
    // at construction time for correctness in the general case.
    InstructionId id_base_ = 0;
    size_t        idx(InstructionId id) const {
        return static_cast<size_t>(id - id_base_);
    }

    EventEngine& engine_;
    OpRegistry&  registry_;
    Schedule     sched_;

    // -----------------------------------------------------------------------
    // RAM-efficient flat vectors (replacing the three unordered containers
    // that dominated host memory on large schedules).
    //
    // Because InstructionIds are sequential integers (0..N-1), a direct vector
    // index replaces hashing entirely — O(1) access with no bucket arrays,
    // no per-node heap allocation, and cache-friendly layout.
    //
    // Memory at 11.6M instructions (full LLaMA-3-8B):
    //   remaining_deps_  vector<int>               :  46 MB  (was ~370 MB hashmap)
    //   issued_          vector<uint8_t>            :  12 MB  (was ~280 MB hashset)
    //   successors_      vector<vector<uint32_t>>   : ~320 MB (was ~510 MB hashmap)
    //   Savings: ~780 MB
    // -----------------------------------------------------------------------
    std::vector<int>                        remaining_deps_;  // [idx(id)] dep counter
    std::vector<uint8_t>                    issued_;          // [idx(id)] 0=no 1=yes
    std::vector<std::vector<InstructionId>> successors_;      // [idx(id)] dependents

    // id -> instruction pointer for O(1) lookup in try_issue.
    // Now a flat vector indexed by (id - id_base_), same as the other
    // three structures above. Saves ~310 MB vs the unordered_map at
    // 11.1M instructions (eliminates the bucket array + per-node overhead).
    // A null entry means the id was never registered — same error semantics
    // as the old map::find() returning end().
    std::vector<const Instruction*> by_id_;

    size_t done_count_ = 0;
};

}  // namespace sim