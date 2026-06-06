#include "units/systolic_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

SystolicUnit::SystolicUnit(std::string name, const SystolicConfig& cfg,
                           Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

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

        if (verbose_)
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

        if (verbose_)
            os_ << "  [" << name() << "]  GEMM_DONE"
                << "  instr="  << e.instr
                << "  @cycle=" << e.cycle
                << (e.label.empty() ? "" : "  \"" + e.label + "\"")
                << "\n";

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
