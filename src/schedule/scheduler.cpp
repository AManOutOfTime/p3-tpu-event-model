#include "schedule/scheduler.h"
#include <stdexcept>
#include <algorithm>

namespace sim {

Scheduler::Scheduler(EventEngine& engine, OpRegistry& registry, Schedule schedule)
    : engine_(engine), registry_(registry), sched_(std::move(schedule)) {

    const size_t N = sched_.instructions.size();
    if (N == 0) return;

    // Compute the ID range. In practice the builder assigns IDs sequentially
    // from 0, so min_id==0 and id_range==N. We compute it explicitly so the
    // code is correct even if an edge case shifts the base (e.g. a hand-written
    // YAML that starts numbering at 1).
    InstructionId min_id = sched_.instructions[0].id;
    InstructionId max_id = sched_.instructions[0].id;
    for (const auto& inst : sched_.instructions) {
        if (inst.id < min_id) min_id = inst.id;
        if (inst.id > max_id) max_id = inst.id;
    }
    id_base_             = min_id;
    const size_t id_range = static_cast<size_t>(max_id - min_id) + 1;

    // Allocate all three flat vectors in one pass — no rehashing, no per-node
    // heap allocation. Default-initialise to 0 / empty so the second pass
    // (which fills successor lists) can push_back unconditionally.
    remaining_deps_.assign(id_range, 0);
    issued_.assign(id_range, 0u);
    successors_.resize(id_range);
    by_id_.assign(id_range, nullptr);

    for (const auto& inst : sched_.instructions) {
        by_id_[idx(inst.id)] = &inst;
        remaining_deps_[idx(inst.id)] = static_cast<int>(inst.depends_on.size());
        for (auto d : inst.depends_on)
            successors_[idx(d)].push_back(inst.id);
    }

    // Release per-instruction dep vectors: the Scheduler has extracted
    // everything it needs into remaining_deps_ and successors_. Freeing here
    // eliminates ~11M small heap allocations and ~88 MB of dep-ID storage,
    // reducing both peak RSS and minor page faults during simulation.
    for (auto& inst : sched_.instructions)
        std::vector<InstructionId>{}.swap(inst.depends_on);
}

void Scheduler::launch() {
    for (const auto& inst : sched_.instructions)
        if (remaining_deps_[idx(inst.id)] == 0)
            try_issue(inst.id);
}

void Scheduler::notify_done(InstructionId id) {
    done_count_++;
    // Walk the flat successor list — no hash lookup, no iterator indirection.
    for (auto s : successors_[idx(id)])
        if (--remaining_deps_[idx(s)] == 0)
            try_issue(s);
}

Cycle Scheduler::reserve_unit(UnitId id, Cycle duration) {
    return reserve_unit_pool(std::vector<UnitId>{id}, duration).start;
}

UnitReservation Scheduler::reserve_unit_pool(const std::vector<UnitId>& ids,
                                             Cycle duration,
                                             uint64_t buffer_bytes) {
    return engine_.reserve_unit_pool(ids, duration, buffer_bytes);
}

void Scheduler::try_issue(InstructionId id) {
    uint8_t& flag = issued_[idx(id)];
    if (flag) return;   // already issued — guard against double-fire
    flag = 1;

    const Instruction* inst = by_id_[idx(id)];
    if (!inst)
        throw std::runtime_error("Scheduler: instruction id not found: " + std::to_string(id));
    // Use the uint8_t fast path — integer key lookup, no string hash.
    registry_.get(inst->op.code())(IssueCtx{engine_, *this, *inst});
}

}  // namespace sim