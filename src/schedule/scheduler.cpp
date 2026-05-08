#include "schedule/scheduler.h"
#include <stdexcept>

namespace sim {

Scheduler::Scheduler(EventEngine& engine, OpRegistry& registry, Schedule schedule)
    : engine_(engine), registry_(registry), sched_(std::move(schedule)) {

    for (const auto& inst : sched_.instructions) {
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

void Scheduler::try_issue(InstructionId id) {
    if (!issued_.insert(id).second) return;  // already issued

    const Instruction* inst = nullptr;
    for (const auto& i : sched_.instructions)
        if (i.id == id) { inst = &i; break; }
    if (!inst)
        throw std::runtime_error("Scheduler: instruction id not found: " + std::to_string(id));

    registry_.get(inst->op)(IssueCtx{engine_, *this, *inst});
}

}  // namespace sim
