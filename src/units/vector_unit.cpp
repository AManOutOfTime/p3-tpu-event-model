#include "units/vector_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace sim {

VectorUnit::VectorUnit(std::string name, const VectorCoreConfig& cfg,
                       Scheduler* sched, TensorStore* ts, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), ts_(ts), os_(os) {}

Cycle VectorUnit::compute_latency(const VectorOp& op) const {
    if (!op.elements || !cfg_.simd_width) return 0;
    Cycle g = static_cast<Cycle>(std::ceil(
        static_cast<double>(op.elements) / cfg_.simd_width));
    return static_cast<Cycle>(op.passes) * g
         + static_cast<Cycle>(op.exp_ops) * cfg_.exp_latency * g;
}

// ---------------------------------------------------------------------------
// do_scale
// If src_scale is set: dst[i] *= row_vector[i % rows]  (row-wise broadcast)
// Otherwise:           dst[i] *= 1.0  (no-op; scalar stored in scalar field)
// In FA2 id=9:  S_tile /= sqrt(d_k)  — here we just pass through;
//               the actual scale factor is symbolic in the schedule.
//               We mark S_tile as "scaled" without changing values so the
//               downstream reads still see valid attention logits.
// In FA2 id=14: O_acc[r,*] *= correction[r]  — row broadcast.
// ---------------------------------------------------------------------------
void VectorUnit::do_scale(const VectorOp& op) {
    const std::string& in  = op.src.empty()  ? op.src_a : op.src;
    const std::string& out = op.dst;
    if (in.empty() || out.empty() || !ts_->has(in)) return;

    // Copy src → dst if different buffers
    if (in != out) ts_->copy(in, out);

    // Apply row-vector scale if provided (e.g. O_acc *= correction row-wise)
    if (!op.src_scale.empty() && ts_->has(op.src_scale)) {
        auto& dst_buf      = ts_->get_mutable(out);
        const auto& scale  = ts_->get(op.src_scale);
        const uint32_t rows = op.rows ? op.rows
                            : static_cast<uint32_t>(scale.size());
        const uint32_t cols = op.cols ? op.cols
                            : static_cast<uint32_t>(dst_buf.size() / rows);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < cols; c++)
                dst_buf[r * cols + c] *= scale[r];
        os_ << "  [" << name() << "]  SCALE  \"" << in
            << "\" *= \"" << op.src_scale << "\" (row-broadcast) → \""
            << out << "\"\n";
    }
    // else: symbolic scalar (1/sqrt(d_k)) — timing only, values unchanged
}

// ---------------------------------------------------------------------------
// do_rowmax
// dst[r] = max over cols of src[r, :]
// FA2 id=10: rowmax_tmp[Br] = rowmax(S_tile[Br×Bc])
// ---------------------------------------------------------------------------
void VectorUnit::do_rowmax(const VectorOp& op) {
    if (!ts_->has(op.src) || op.dst.empty()) return;
    const auto& src = ts_->get(op.src);
    const uint32_t rows = op.rows;
    const uint32_t cols = op.cols;
    std::vector<float> out(rows, -std::numeric_limits<float>::infinity());
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++)
            out[r] = std::max(out[r], src[r * cols + c]);
    ts_->set(op.dst, std::move(out));
    os_ << "  [" << name() << "]  ROWMAX  \"" << op.src
        << "\" [" << rows << "x" << cols << "] → \"" << op.dst << "\"\n";
}

// ---------------------------------------------------------------------------
// do_update_rowmax
// m_new[r] = max(m_old[r], rowmax[r])
// correction[r] = exp(m_old[r] - m_new[r])
// FA2 id=11
// ---------------------------------------------------------------------------
void VectorUnit::do_update_rowmax(const VectorOp& op) {
    if (!ts_->has(op.src_m) || !ts_->has(op.src_rowmax)) return;

    const auto& m_old   = ts_->get(op.src_m);
    const auto& rowmax  = ts_->get(op.src_rowmax);
    const size_t Br     = m_old.size();

    std::vector<float> m_new(Br);
    std::vector<float> correction(Br);

    for (size_t r = 0; r < Br; r++) {
        m_new[r]       = std::max(m_old[r], rowmax[r]);
        correction[r]  = std::exp(m_old[r] - m_new[r]);
    }

    ts_->set(op.dst_m,           std::move(m_new));
    ts_->set(op.dst_correction,  std::move(correction));

    os_ << "  [" << name() << "]  UPDATE_ROWMAX  m=max(m_old,rowmax)"
        << "  correction=exp(m_old-m_new) → \""
        << op.dst_m << "\", \"" << op.dst_correction << "\"\n";
}

// ---------------------------------------------------------------------------
// do_exp_shift
// dst[r,c] = exp(src_matrix[r,c] - src_shift[r])   [broadcast shift over cols]
// FA2 id=12: P_tile = exp(S_tile - m_new)
// ---------------------------------------------------------------------------
void VectorUnit::do_exp_shift(const VectorOp& op) {
    if (!ts_->has(op.src_matrix) || !ts_->has(op.src_shift)) return;

    const auto& S     = ts_->get(op.src_matrix);
    const auto& m     = ts_->get(op.src_shift);
    const uint32_t Br = op.rows;
    const uint32_t Bc = op.cols;

    std::vector<float> P(static_cast<size_t>(Br) * Bc);
    for (uint32_t r = 0; r < Br; r++)
        for (uint32_t c = 0; c < Bc; c++)
            P[r * Bc + c] = std::exp(S[r * Bc + c] - m[r]);

    ts_->set(op.dst, std::move(P));
    os_ << "  [" << name() << "]  EXP_SHIFT  exp(\""
        << op.src_matrix << "\" - \"" << op.src_shift
        << "\") → \"" << op.dst << "\"\n";
}

// ---------------------------------------------------------------------------
// do_update_rowsum
// l_new[r] = correction[r] * l_old[r] + sum over cols of P[r,:]
// FA2 id=13
// ---------------------------------------------------------------------------
void VectorUnit::do_update_rowsum(const VectorOp& op) {
    if (!ts_->has(op.src_p) || !ts_->has(op.src_correction)
        || !ts_->has(op.src_l)) return;

    const auto& P           = ts_->get(op.src_p);
    const auto& correction  = ts_->get(op.src_correction);
    const auto& l_old       = ts_->get(op.src_l);
    const uint32_t Br       = op.rows;
    const uint32_t Bc       = op.cols;

    std::vector<float> l_new(Br);
    for (uint32_t r = 0; r < Br; r++) {
        float rowsum = 0.f;
        for (uint32_t c = 0; c < Bc; c++)
            rowsum += P[r * Bc + c];
        l_new[r] = correction[r] * l_old[r] + rowsum;
    }

    const std::string& dst = op.dst_l.empty() ? op.dst : op.dst_l;
    ts_->set(dst, std::move(l_new));
    os_ << "  [" << name() << "]  UPDATE_ROWSUM  l = correction*l_old + rowsum(P)"
        << " → \"" << dst << "\"\n";
}

// ---------------------------------------------------------------------------
// do_accumulate
// dst[i] = src_a[i] + src_b[i]   element-wise
// FA2 id=17: O_acc += Temp
// ---------------------------------------------------------------------------
void VectorUnit::do_accumulate(const VectorOp& op) {
    if (!ts_->has(op.src_a) || !ts_->has(op.src_b)) return;
    const auto& a = ts_->get(op.src_a);
    const auto& b = ts_->get(op.src_b);
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); i++)
        out[i] = a[i] + b[i];
    ts_->set(op.dst, std::move(out));
    os_ << "  [" << name() << "]  ACCUMULATE  \""
        << op.src_a << "\" += \"" << op.src_b
        << "\" → \"" << op.dst << "\"\n";
}

// ---------------------------------------------------------------------------
// do_normalize
// dst[r,c] = src_matrix[r,c] / src_denom[r]   row-wise divide
// FA2 id=18: O_tile = O_acc / l
// ---------------------------------------------------------------------------
void VectorUnit::do_normalize(const VectorOp& op) {
    if (!ts_->has(op.src_matrix) || !ts_->has(op.src_denom)) return;
    const auto& mat   = ts_->get(op.src_matrix);
    const auto& denom = ts_->get(op.src_denom);
    const uint32_t Br = op.rows;
    const uint32_t cols = op.cols;
    std::vector<float> out(mat.size());
    for (uint32_t r = 0; r < Br; r++) {
        float d = (std::abs(denom[r]) < 1e-9f) ? 1e-9f : denom[r];
        for (uint32_t c = 0; c < cols; c++)
            out[r * cols + c] = mat[r * cols + c] / d;
    }
    ts_->set(op.dst, std::move(out));
    os_ << "  [" << name() << "]  NORMALIZE  \""
        << op.src_matrix << "\" / \"" << op.src_denom
        << "\" → \"" << op.dst << "\"\n";
}

// ---------------------------------------------------------------------------
// do_logsumexp
// dst[r] = src_m[r] + log(src_l[r])
// FA2 id=19: L_tile = m + log(l)
// ---------------------------------------------------------------------------
void VectorUnit::do_logsumexp(const VectorOp& op) {
    if (!ts_->has(op.src_m) || !ts_->has(op.src_l)) return;
    const auto& m = ts_->get(op.src_m);
    const auto& l = ts_->get(op.src_l);
    std::vector<float> L(m.size());
    for (size_t r = 0; r < m.size(); r++)
        L[r] = m[r] + std::log(l[r] < 1e-9f ? 1e-9f : l[r]);
    ts_->set(op.dst, std::move(L));
    os_ << "  [" << name() << "]  LOGSUMEXP  m + log(l) → \""
        << op.dst << "\"\n";
}

// ---------------------------------------------------------------------------
// handle
// ---------------------------------------------------------------------------
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
        // ── Compute at OP_DONE ─────────────────────────────────────────
        if (ts_) {
            if (const auto* op = std::any_cast<VectorOp>(&e.payload)) {
                if      (op->kind == "scale")          do_scale(*op);
                else if (op->kind == "rowmax")         do_rowmax(*op);
                else if (op->kind == "update_rowmax")  do_update_rowmax(*op);
                else if (op->kind == "exp_shift")      do_exp_shift(*op);
                else if (op->kind == "update_rowsum")  do_update_rowsum(*op);
                else if (op->kind == "accumulate")     do_accumulate(*op);
                else if (op->kind == "normalize")      do_normalize(*op);
                else if (op->kind == "logsumexp")      do_logsumexp(*op);
            }
        }

        os_ << "  [" << name() << "]  VEC_DONE"
            << "  instr=" << e.instr << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
