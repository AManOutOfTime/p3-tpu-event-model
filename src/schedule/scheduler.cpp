#include "schedule/scheduler.h"
#include <stdexcept>

namespace sim {

Scheduler::Scheduler(EventEngine& engine, OpRegistry& registry, Schedule schedule)
    : engine_(engine), registry_(registry), sched_(std::move(schedule)) {

    by_id_.reserve(sched_.instructions.size() * 2);
    remaining_deps_.reserve(sched_.instructions.size() * 2);
    for (const auto& inst : sched_.instructions) {
        by_id_[inst.id] = &inst;
        remaining_deps_[inst.id] = static_cast<int>(inst.depends_on.size());
        for (auto d : inst.depends_on)
            successors_[d].push_back(inst.id);
    }
}

void Scheduler::launch() {
    for (const auto& inst : sched_.instructions)
        if (remaining_deps_[inst.id] == 0)
            try_issue(inst.id);
}

void Scheduler::notify_done(InstructionId id) {
    done_count_++;
    auto it = successors_.find(id);
    if (it == successors_.end()) return;
    for (auto s : it->second)
        if (--remaining_deps_[s] == 0)
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
    if (!issued_.insert(id).second) return;  // already issued

    auto it = by_id_.find(id);
    if (it == by_id_.end())
        throw std::runtime_error("Scheduler: instruction id not found: " + std::to_string(id));
    const Instruction* inst = it->second;

    registry_.get(inst->op)(IssueCtx{engine_, *this, *inst});
}

}  // namespace sim
