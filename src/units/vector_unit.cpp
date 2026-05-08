#include "units/vector_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

VectorUnit::VectorUnit(std::string name, const VectorCoreConfig& cfg,
                       Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

// ---------------------------------------------------------------------------
// compute_latency
//
//   simd_groups = ceil(elements / simd_width)
//   total = passes × simd_groups          ← linear-pass cycles
//         + exp_ops × exp_latency × simd_groups  ← transcendental overhead
//
//   Worked examples (simd_width=64, exp_latency=4):
//
//   scale(128×128=16384 elems):
//     groups=256, passes=1, exp_ops=0  → 256 cycles
//
//   exp(128×128=16384 elems):
//     groups=256, passes=1, exp_ops=1  → 256 + 4×256 = 1280 cycles
//
//   softmax(Br=128, Bc=128 → 16384 elems):
//     groups=256, passes=3, exp_ops=1  → 3×256 + 4×256 = 1792 cycles
//
//   rowmax(16384 elems):
//     groups=256, passes=1, exp_ops=0  → 256 cycles
//
//   logsumexp(128 elems — length Br):
//     groups=2, passes=1, exp_ops=1    → 2 + 4×2 = 10 cycles
// ---------------------------------------------------------------------------
Cycle VectorUnit::compute_latency(const VectorOp& op) const {
    if (op.elements == 0 || cfg_.simd_width == 0) return 0;
    const Cycle groups = static_cast<Cycle>(
        std::ceil(static_cast<double>(op.elements) /
                  static_cast<double>(cfg_.simd_width)));
    const Cycle linear = static_cast<Cycle>(op.passes) * groups;
    const Cycle trans  = static_cast<Cycle>(op.exp_ops)
                        * static_cast<Cycle>(cfg_.exp_latency) * groups;
    return linear + trans;
}

void VectorUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat      = 0;
        bool  from_cfg = false;

        if (const auto* op = std::any_cast<VectorOp>(&e.payload)) {
            lat      = compute_latency(*op);

            os_ << "  [" << name() << "]  VEC_START"
                << "  instr="  << e.instr
                << "  @cycle=" << e.cycle
                << "  kind="   << (op->kind.empty() ? "?" : op->kind)
                << "  elems="  << op->elements
                << "  passes=" << op->passes
                << "  exp_ops="<< op->exp_ops
                << "  lat="    << lat
                << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);
            os_ << "  [" << name() << "]  VEC_START"
                << "  instr="  << e.instr
                << "  @cycle=" << e.cycle
                << "  lat="    << lat
                << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";
        }

        Event done = e;
        done.type  = EventType::OP_DONE;
        done.cycle = e.cycle + lat;
        done.seq   = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        os_ << "  [" << name() << "]  VEC_DONE"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";
        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
