#include "units/access_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace sim {

AccessUnit::AccessUnit(std::string name, const AccessCoreConfig& cfg,
                       Scheduler* sched, TensorStore* ts, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), ts_(ts), os_(os) {}

Cycle AccessUnit::compute_latency(uint64_t elements) const {
    if (!elements || !cfg_.bandwidth) return 0;
    return static_cast<Cycle>(std::ceil(
        static_cast<double>(elements) / cfg_.bandwidth));
}

// ---------------------------------------------------------------------------
// do_init_fill
// Fills op.dst with op.fill_value for op.elements elements.
// ---------------------------------------------------------------------------
void AccessUnit::do_init_fill(const AccessOp& op) {
    if (op.dst.empty()) return;
    if (!ts_->has(op.dst)) {
        ts_->init_value(op.dst, op.elements, op.fill_value);
    } else {
        auto& buf = ts_->get_mutable(op.dst);
        std::fill(buf.begin(), buf.end(), op.fill_value);
    }
    os_ << "  [" << name() << "]  INIT_FILL  \""
        << op.dst << "\"  elems=" << op.elements
        << "  value=" << op.fill_value << "\n";
}

// ---------------------------------------------------------------------------
// do_transpose
// Reads src [input_rows × input_cols] row-major.
// Writes dst [input_cols × input_rows] row-major (transposed).
// ---------------------------------------------------------------------------
void AccessUnit::do_transpose(const AccessOp& op) {
    if (op.src.empty() || op.dst.empty()) return;
    if (!ts_->has(op.src)) {
        os_ << "  [" << name() << "]  TRANSPOSE SKIPPED (src '"
            << op.src << "' not found)\n";
        return;
    }
    const auto& src = ts_->get(op.src);
    const uint32_t R = op.input_rows;
    const uint32_t C = op.input_cols;
    std::vector<float> dst(static_cast<size_t>(R) * C);
    // dst[c][r] = src[r][c]
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t c = 0; c < C; c++)
            dst[c * R + r] = src[r * C + c];
    ts_->set(op.dst, std::move(dst));
    os_ << "  [" << name() << "]  TRANSPOSE  \"" << op.src
        << "\" [" << R << "x" << C << "] → \"" << op.dst
        << "\" [" << C << "x" << R << "]\n";
}

void AccessUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat = 0;
        if (const auto* op = std::any_cast<AccessOp>(&e.payload)) {
            lat = compute_latency(op->elements);
            os_ << "  [" << name() << "]  ACCESS_START"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << "  kind=" << op->kind << "  elems=" << op->elements
                << "  lat=" << lat;
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            lat = static_cast<Cycle>(*p);
            os_ << "  [" << name() << "]  ACCESS_START"
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
        // ── Compute at OP_DONE ─────────────────────────────────────────
        if (ts_) {
            if (const auto* op = std::any_cast<AccessOp>(&e.payload)) {
                if      (op->kind == "init_fill") do_init_fill(*op);
                else if (op->kind == "transpose") do_transpose(*op);
            }
        }

        os_ << "  [" << name() << "]  ACCESS_DONE"
            << "  instr=" << e.instr << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
