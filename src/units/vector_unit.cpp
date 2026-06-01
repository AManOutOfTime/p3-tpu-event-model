#include "units/vector_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

VectorUnit::VectorUnit(std::string name, const VectorCoreConfig& cfg,
                       Scheduler* sched, TensorStore*, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

Cycle VectorUnit::compute_latency(const VectorOp& op) const {
    if (!op.elements || !cfg_.simd_width) return 0;
    Cycle groups = static_cast<Cycle>(std::ceil(
        static_cast<double>(op.elements) / cfg_.simd_width));
    return static_cast<Cycle>(op.passes) * groups
         + static_cast<Cycle>(op.exp_ops) * cfg_.exp_latency * groups;
}

void VectorUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat = 0;
        if (const auto* op = std::any_cast<VectorOp>(&e.payload)) {
            lat = compute_latency(*op);
            os_ << "  [" << name() << "]  VEC_START"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << "  kind=" << op->kind << "  elems=" << op->elements
                << "  lat=" << lat;
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);
            os_ << "  [" << name() << "]  VEC_START"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << "  lat=" << lat;
        }
        os_ << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        Event done = e;
        done.type  = EventType::OP_DONE;
        done.cycle = e.cycle + lat;
        done.seq   = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        os_ << "  [" << name() << "]  VEC_DONE"
            << "  instr=" << e.instr << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
