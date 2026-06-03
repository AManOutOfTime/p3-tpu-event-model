#include "units/systolic_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <algorithm>
#include <cmath>

namespace sim {

SystolicUnit::SystolicUnit(std::string name, const SystolicConfig& cfg,
                           Scheduler* sched, TensorStore* ts, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), ts_(ts), os_(os) {}

// ---------------------------------------------------------------------------
// fill_latency
//
//   Unidirectional: wavefront traverses the full array from one edge.
//     fill = (rows-1) + (cols-1)
//
//   Bidirectional: two wavefronts meet in the middle, halving the fill.
//     fill = ceil((rows-1)/2) + ceil((cols-1)/2)
// ---------------------------------------------------------------------------
Cycle SystolicUnit::fill_latency() const {
    if (cfg_.bidirectional) {
        return static_cast<Cycle>((cfg_.rows - 1 + 1) / 2)
             + static_cast<Cycle>((cfg_.cols - 1 + 1) / 2);
    }
    return static_cast<Cycle>(cfg_.rows - 1)
         + static_cast<Cycle>(cfg_.cols - 1);
}

// ---------------------------------------------------------------------------
// compute_latency
//
//   Tiles the logical GEMM over the physical array:
//     tiles_m = ceil(M / SA_rows),  tiles_n = ceil(N / SA_cols)
//     per_tile = K + fill_latency
//     total    = tiles_m * tiles_n * per_tile
//
//   In normal use every gemm instruction arrives pre-tiled by
//   Tiler::expand_gemm_subtiles(), so tiles_m = tiles_n = 1 and the
//   formula reduces to K + fill.  Residual-size tiles work correctly.
// ---------------------------------------------------------------------------
Cycle SystolicUnit::compute_latency(uint32_t M, uint32_t K, uint32_t N) const {
    if (M == 0 || K == 0 || N == 0) return 0;
    const Cycle tiles_m  = (M + cfg_.rows - 1) / cfg_.rows;
    const Cycle tiles_n  = (N + cfg_.cols - 1) / cfg_.cols;
    const Cycle per_tile = static_cast<Cycle>(K) + fill_latency();
    return tiles_m * tiles_n * per_tile;
}

// ---------------------------------------------------------------------------
// do_gemm  —  tiled float GEMM matching the physical array layout.
//
//   C[M×N] = A[M×K] × B[K×N]    (row-major float32)
//
//   The outer tile loops match hardware routing: each (ti, tj) block is
//   one physical array execution.  Inside each tile K values are streamed
//   in, mirroring the K-cycle accumulation phase.
//
//   Buffer retrieval cost is already captured in compute_latency():
//     src_a: zero (pre-staged by the preceding dma_stage instruction)
//     src_b: the K term (one IBUF row per cycle)
//     dst_c: pipelined write during accumulation (folded into latency)
// ---------------------------------------------------------------------------
void SystolicUnit::do_gemm(const GemmShape& shape) {
    const uint32_t M = shape.M, K = shape.K, N = shape.N;

    if (!ts_->has(shape.src_a) || !ts_->has(shape.src_b)) {
        os_ << "  [" << name() << "]  GEMM_COMPUTE  SKIPPED"
            << "  (buffers \"" << shape.src_a << "\" or \""
            << shape.src_b << "\" not found in TensorStore)\n";
        return;
    }

    const auto& A = ts_->get(shape.src_a);  // M×K, row-major
    const auto& B = ts_->get(shape.src_b);  // K×N, row-major

    if (A.size() < static_cast<size_t>(M) * K ||
        B.size() < static_cast<size_t>(K) * N) {
        os_ << "  [" << name() << "]  GEMM_COMPUTE  ERROR"
            << "  buffer size mismatch (A=" << A.size()
            << " need " << M * K
            << ", B=" << B.size() << " need " << K * N << ")\n";
        return;
    }

    std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);

    const uint32_t TM = cfg_.rows;
    const uint32_t TN = cfg_.cols;

    for (uint32_t ti = 0; ti < M; ti += TM) {
        const uint32_t ib = std::min(TM, M - ti);
        for (uint32_t tj = 0; tj < N; tj += TN) {
            const uint32_t jb = std::min(TN, N - tj);
            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t i = 0; i < ib; i++) {
                    const float a = A[(ti + i) * K + k];
                    for (uint32_t j = 0; j < jb; j++) {
                        C[(ti + i) * N + (tj + j)] += a * B[k * N + (tj + j)];
                    }
                }
            }
        }
    }

    ts_->set(shape.dst_c, std::move(C));

    os_ << "  [" << name() << "]  GEMM_COMPUTE"
        << "  \"" << shape.src_a << "\" [" << M << "x" << K << "]"
        << " x \"" << shape.src_b << "\" [" << K << "x" << N << "]"
        << " → \"" << shape.dst_c << "\" [" << M << "x" << N << "]\n";
}

void SystolicUnit::handle(const Event& e, EventEngine& engine) {

    if (e.type == EventType::OP_START) {
        uint32_t M = cfg_.rows, K = cfg_.rows, N = cfg_.cols;
        if (const auto* s = std::any_cast<GemmShape>(&e.payload)) {
            M = s->M; K = s->K; N = s->N;
        }

        const Cycle tiles_m  = (M + cfg_.rows - 1) / cfg_.rows;
        const Cycle tiles_n  = (N + cfg_.cols - 1) / cfg_.cols;
        const Cycle fill     = fill_latency();
        const Cycle per_tile = static_cast<Cycle>(K) + fill;
        const Cycle lat      = tiles_m * tiles_n * per_tile;

        os_ << "  [" << name() << "]  GEMM_START"
            << "  instr="     << e.instr
            << "  @cycle="    << e.cycle
            << "  shape=["    << M << "x" << K << "x" << N << "]"
            << "  array=["    << cfg_.rows << "x" << cfg_.cols << "]"
            << "  mode="      << (cfg_.bidirectional ? "bidir" : "unidir")
            << "  tiles=["    << tiles_m << "x" << tiles_n << "]"
            << "  fill="      << fill
            << "  per_tile="  << per_tile
            << "  total_lat=" << lat
            << (e.label.empty() ? "" : "  \"" + e.label + "\"")
            << "\n";

        Event done  = e;
        done.type   = EventType::OP_DONE;
        done.cycle  = e.cycle + lat;
        done.seq    = engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {

        if (ts_) {
            if (const auto* s = std::any_cast<GemmShape>(&e.payload)) {
                if (!s->src_a.empty() && !s->src_b.empty() && !s->dst_c.empty()) {
                    do_gemm(*s);
                }
            }
        }

        os_ << "  [" << name() << "]  GEMM_DONE"
            << "  instr="  << e.instr
            << "  @cycle=" << e.cycle
            << (e.label.empty() ? "" : "  \"" + e.label + "\"")
            << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
