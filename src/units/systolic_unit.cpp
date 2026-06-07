#include "units/systolic_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <algorithm>
#include <cmath>

namespace sim {

SystolicUnit::SystolicUnit(std::string name, const SystolicConfig& cfg,
                           Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

// ---------------------------------------------------------------------------
// systolic_fill_latency  (free function — shared with the gemm op handler)
//
//   Unidirectional: wavefront traverses the full array from one edge.
//     fill = (rows-1) + (cols-1)
//   Bidirectional: two wavefronts meet in the middle, halving the fill.
//     fill = ceil((rows-1)/2) + ceil((cols-1)/2)
// ---------------------------------------------------------------------------
Cycle systolic_fill_latency(const SystolicConfig& cfg) {
    if (cfg.bidirectional) {
        return static_cast<Cycle>((cfg.rows - 1 + 1) / 2)
             + static_cast<Cycle>((cfg.cols - 1 + 1) / 2);
    }
    return static_cast<Cycle>(cfg.rows - 1)
         + static_cast<Cycle>(cfg.cols - 1);
}

// ---------------------------------------------------------------------------
// systolic_gemm_latency  (free function — single source of truth, P0.1/P1.4)
//
// WEIGHT-STATIONARY (default):
//   Weights (a K×N block) sit in the PE array; M rows of activations stream
//   through. Fragmentation is over (K, N) — M streams and is never tiled here.
//
//     tiles_k = ceil(K / rows)    K-blocks; each needs a weight load
//     tiles_n = ceil(N / cols)    output-column tiles
//     weight_load = cycles to load one K-block of weights (default = rows)
//
//   No prefetch (weight_double_buffer=false):
//     per_n = tiles_k*(weight_load + M) + fill           (load+stream serially)
//   Double-buffered weight FIFO (P1.4, default):
//     per_n = weight_load + tiles_k*max(weight_load, M) + fill
//       → weight-load(i+1) hides behind stream(i); only the 1st load is exposed.
//       → decode (M=1) and small-M prefill are weight-load-bound (correct);
//         large-M prefill (M ≥ weight_load) becomes streaming/compute-bound.
//   lat = tiles_n * per_n.
//
//   Fragmentation falls out naturally: a 512² array gives tiles_k=ceil(4096/512)
//   =8, not a free 4×, reproducing the TPU-v1 "no free speedup" finding the
//   array-size sweep needs.
//
// OUTPUT-STATIONARY (legacy, selectable): M-insensitive K+fill per output tile.
// ---------------------------------------------------------------------------
Cycle systolic_gemm_latency(const SystolicConfig& cfg,
                            uint32_t M, uint32_t K, uint32_t N) {
    if (M == 0 || K == 0 || N == 0) return 0;
    const Cycle fill = systolic_fill_latency(cfg);

    if (cfg.dataflow == "output_stationary") {
        const Cycle tiles_m = (M + cfg.rows - 1) / cfg.rows;
        const Cycle tiles_n = (N + cfg.cols - 1) / cfg.cols;
        return tiles_m * tiles_n * (static_cast<Cycle>(K) + fill);
    }

    // weight_stationary (default)
    const Cycle tiles_k = (K + cfg.rows - 1) / cfg.rows;
    const Cycle tiles_n = (N + cfg.cols - 1) / cfg.cols;
    const Cycle wload   = cfg.weight_load_cycles
        ? static_cast<Cycle>(cfg.weight_load_cycles)
        : static_cast<Cycle>(cfg.rows);
    const Cycle Mc = static_cast<Cycle>(M);

    Cycle per_n;
    if (cfg.weight_double_buffer)
        per_n = wload + tiles_k * std::max(wload, Mc) + fill;
    else
        per_n = tiles_k * (wload + Mc) + fill;

    return tiles_n * per_n;
}

Cycle SystolicUnit::fill_latency() const { return systolic_fill_latency(cfg_); }

Cycle SystolicUnit::compute_latency(uint32_t M, uint32_t K, uint32_t N) const {
    return systolic_gemm_latency(cfg_, M, K, N);
}


void SystolicUnit::handle(const Event& e, EventEngine& engine) {

    if (e.type == EventType::OP_START) {
        uint32_t M = cfg_.rows, K = cfg_.rows, N = cfg_.cols;
        uint64_t buffer_bytes = 0;
        uint32_t spill_penalty = 0;
        if (const auto* s = std::any_cast<GemmShape>(&e.payload)) {
            M = s->M; K = s->K; N = s->N;
            buffer_bytes = s->buffer_bytes; spill_penalty = s->spill_penalty;
        }

        const Cycle tiles_k = (K + cfg_.rows - 1) / cfg_.rows;
        const Cycle tiles_n = (N + cfg_.cols - 1) / cfg_.cols;
        const Cycle fill    = fill_latency();
        Cycle lat           = systolic_gemm_latency(cfg_, M, K, N);

        // P1.2: acquire the SRAM working set for this GEMM's actual execution
        // window. Overflow charges a spill penalty (operands streamed from HBM).
        if (buffer_bytes && !engine.sram_acquire(buffer_bytes))
            lat += static_cast<Cycle>(spill_penalty);

        if (verbose_)
            os_ << "  [" << name() << "]  GEMM_START"
                << "  instr="     << e.instr
                << "  @cycle="    << e.cycle
                << "  shape=["    << M << "x" << K << "x" << N << "]"
                << "  array=["    << cfg_.rows << "x" << cfg_.cols << "]"
                << "  mode="      << (cfg_.bidirectional ? "bidir" : "unidir")
                << "  df="        << cfg_.dataflow
                << "  ktiles="    << tiles_k
                << "  ntiles="    << tiles_n
                << "  fill="      << fill
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

        // P1.2: release the SRAM working set this GEMM held.
        if (const auto* s = std::any_cast<GemmShape>(&e.payload))
            if (s->buffer_bytes) engine.sram_release(s->buffer_bytes);

        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
