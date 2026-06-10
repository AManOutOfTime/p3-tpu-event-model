#include "core/event_engine.h"
#include "core/logger.h"
#include "config/arch_config.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/op_handlers.h"
#include "schedule/scheduler.h"
#include "schedule/tiler.h"
#include "schedule/llama_schedule.h"
#include "units/delay_unit.h"
#include <malloc.h>
#include "units/systolic_unit.h"
#include "units/dma_unit.h"
#include "units/buffer_unit.h"
#include "units/vector_unit.h"
#include "units/access_unit.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace sim;
using Clock = std::chrono::steady_clock;

static uint32_t precision_bytes(const std::string& precision) {
    if (precision == "FP8") return 1;
    if (precision == "FP32") return 4;
    return 2;
}

// ---------------------------------------------------------------------------
// Wire scheduler into every unit
// ---------------------------------------------------------------------------
static void wire_units(EventEngine& engine, Scheduler& sched) {
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<DelayUnit*>   (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<SystolicUnit*>(u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<DmaUnit*>     (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<VectorUnit*>  (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<AccessUnit*>  (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<BufferUnit*>  (u)) { x->set_scheduler(&sched); continue; }
    }
}

// ---------------------------------------------------------------------------
// Pre-processing phase: build + tile schedule
//
// This is intentionally separated from the simulation phase so that:
//   (a) pre-processing time is reported independently of simulation time, and
//   (b) callers can cache / serialise the returned Schedule and reuse it
//       across hardware-config sweeps without re-running the builder/tiler.
//
// The tiler call is a no-op for LLaMA paths (the builder already emits
// hardware-sized tiles), but having it here makes the phase boundary explicit.
// ---------------------------------------------------------------------------
struct PreprocessResult {
    Schedule            schedule;
    TileDecomposition   tile_decomp;  // populated only for --workload
    LlamaScheduleConfig llama_cfg;
    bool                used_tiler = false;
    bool                used_llama = false;
};

static PreprocessResult preprocess_schedule(
        const std::string& sched_path,
        const std::string& workload_path,
        const std::string& llama_path,
        const ArchConfig&  arch,
        bool               trace) {

    PreprocessResult r;

    if (!llama_path.empty()) {
        r.llama_cfg = llama_config_from_yaml_file(llama_path);
        r.llama_cfg.dtype_bytes = precision_bytes(arch.systolic.precision);
        if (r.llama_cfg.sram_kv_capacity_kb == 0)
            r.llama_cfg.sram_kv_capacity_kb = arch.sram.ibuf_kb + arch.sram.obuf_kb;

        // Build in "minimal" metadata mode when tracing is off — drops
        // trace-only labels/buffer-name strings to cut host RAM on large
        // schedules (timing-neutral; the timing model reads only numeric params).
        Schedule raw = build_llama_schedule(r.llama_cfg, /*minimal=*/!trace);

        // Tiler fast-path: LLaMA builder emits hardware-sized tiles, so this
        // is a move-through no-op (no copy). Kept here so the pre-processing
        // phase is the single place that handles tiling regardless of input mode.
        r.schedule   = Tiler::expand_gemm_subtiles(std::move(raw), arch);
        r.used_llama = true;

        std::cout << "  llama_mode=" << r.llama_cfg.mode
                  << "  q_heads=" << r.llama_cfg.num_q_heads
                  << "  kv_heads=" << r.llama_cfg.num_kv_heads
                  << "  gqa_group=" << (r.llama_cfg.num_q_heads / r.llama_cfg.num_kv_heads)
                  << "  kv_cache=" << (r.llama_cfg.kv_cache_enabled ? "on" : "off")
                  << ":" << r.llama_cfg.kv_cache_location << "\n";

    } else if (!workload_path.empty()) {
        // --workload: tiler decomposes a logical GEMM into STAGE+GEMM tiles.
        WorkloadGemm wl  = Tiler::from_yaml_file(workload_path);
        r.tile_decomp    = Tiler::decompose(wl, arch);
        Tiler::print_decomposition(r.tile_decomp);
        r.schedule.instructions = r.tile_decomp.instructions;
        r.used_tiler = true;

    } else {
        // --schedule: hand-written YAML; tiler expands any over-sized GEMMs.
        r.schedule = Schedule::from_yaml_file(sched_path);
        r.schedule = Tiler::expand_gemm_subtiles(std::move(r.schedule), arch);
    }

    return r;
}

// ---------------------------------------------------------------------------
// Metrics: per-unit utilization with per-pool breakdown
// ---------------------------------------------------------------------------
static void print_pool_utilization(const EventEngine& engine, Cycle final_cycle) {
    // Group physical units by their logical pool prefix (strip the trailing
    // _N suffix). Preserve insertion order per pool so units print as
    // systolic_0, systolic_1, systolic_2 rather than in hash order.
    struct UnitStat { UnitId id; std::string name; };
    std::map<std::string, std::vector<UnitStat>> pools;

    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
        const Unit* u = engine.get_unit(uid);
        if (!u) continue;
        const std::string& n = u->name();
        auto pos = n.rfind('_');
        std::string prefix = (pos == std::string::npos) ? n : n.substr(0, pos);
        pools[prefix].push_back({uid, n});
    }

    for (const auto& [prefix, units] : pools) {
        const size_t cnt = units.size();

        // Compute pool totals.
        uint64_t pool_busy = 0;
        for (const auto& us : units)
            pool_busy += engine.unit_busy_cycles(us.id);

        const double pool_util = (final_cycle > 0 && cnt > 0)
            ? 100.0 * static_cast<double>(pool_busy)
              / (static_cast<double>(final_cycle) * static_cast<double>(cnt))
            : 0.0;

        if (cnt == 1) {
            // Single-unit pool: compact one-liner (same format as before).
            std::cout << "  " << prefix
                      << "   busy=" << pool_busy
                      << "   util=" << pool_util << "%\n";
        } else {
            // Multi-unit pool: pool summary header, then one line per unit.
            std::cout << "  " << prefix << " x" << cnt
                      << "   avg_util=" << pool_util << "%\n";
            for (const auto& us : units) {
                const uint64_t busy = engine.unit_busy_cycles(us.id);
                const double util = (final_cycle > 0)
                    ? 100.0 * static_cast<double>(busy)
                      / static_cast<double>(final_cycle)
                    : 0.0;
                std::cout << "    " << us.name
                          << "   busy=" << busy
                          << "   util=" << util << "%\n";
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string config_path   = "configs/default.yaml";
    std::string sched_path    = "";
    std::string workload_path = "";
    std::string llama_path    = "";
    bool        trace         = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--config"   && i+1 < argc) config_path   = argv[++i];
        else if (a == "--schedule" && i+1 < argc) sched_path    = argv[++i];
        else if (a == "--workload" && i+1 < argc) workload_path = argv[++i];
        else if (a == "--llama-workload" && i+1 < argc) llama_path = argv[++i];
        else if (a == "--no-trace")               trace = false;
        else {
            std::cerr << "Usage: sim_main [--config FILE]"
                         " [--schedule FILE | --workload FILE | --llama-workload FILE]"
                         " [--no-trace]\n";
            return 1;
        }
    }
    if (sched_path.empty() && workload_path.empty() && llama_path.empty())
        sched_path = "schedules/dummy_example.yaml";

    ArchConfig arch = ArchConfig::from_yaml_file(config_path);

    std::cout << "clock=" << arch.clock_ghz << " GHz"
              << "  systolic=" << arch.systolic.rows << "x" << arch.systolic.cols
              << "x" << arch.systolic_units
              << " " << (arch.systolic.bidirectional ? "bidir" : "unidir")
              << "  d_head=" << arch.systolic.d_head << "\n"
              << "hbm_bw=" << arch.hbm.bandwidth_tb_s << " TB/s"
              << "  hbm_bpc=" << arch.hbm_bytes_per_cycle()
              << "  hbm_lat=" << arch.hbm.latency_cycles << "\n"
              << "vec_simd=" << arch.vector_core.simd_width
              << "  exp_lat=" << arch.vector_core.exp_latency
              << "  access_bw=" << arch.access_core.bandwidth << "\n\n";

    // ── Pre-processing phase (schedule build + tile decomposition) ──────────
    // Tiling happens here, outside the simulation clock, so its wall-time is
    // reported separately. For sweeps over hardware configs, the pre-processed
    // schedule can be cached and reused without re-running the builder/tiler.
    std::cout << "== pre-processing start ==\n";
    const auto t_preproc_start = Clock::now();

    PreprocessResult pp = preprocess_schedule(
        sched_path, workload_path, llama_path, arch, trace);

    const auto t_preproc_end = Clock::now();
    const double preproc_ms  = std::chrono::duration<double, std::milli>(
        t_preproc_end - t_preproc_start).count();
    const size_t n_instructions = pp.schedule.instructions.size();

    std::cout << "== pre-processing done"
              << "  instructions=" << n_instructions
              << "  wall=" << preproc_ms << " ms ==\n\n";

    malloc_trim(0);

    // ── Build engine ──────────────────────────────────────────────────────
    EventEngine engine(arch.clock_ghz);

    // P1.2: bind shared IBUF+OBUF capacity so over-capacity working sets spill.
    if (arch.model_sram)
        engine.set_sram_capacity(
            static_cast<uint64_t>(arch.sram.ibuf_kb + arch.sram.obuf_kb) * 1024);

    // Systolic/MXU pool
    for (uint32_t i = 0; i < arch.systolic_units; i++)
        engine.register_unit(std::make_unique<SystolicUnit>(
            "systolic_" + std::to_string(i), arch.systolic));

    // DMA channel pool
    for (uint32_t i = 0; i < arch.dma.channels; i++)
        engine.register_unit(std::make_unique<DmaUnit>(
            "dma_" + std::to_string(i), arch));

    // Vector core pool
    for (uint32_t i = 0; i < arch.vector_cores; i++)
        engine.register_unit(std::make_unique<VectorUnit>(
            "vector_core_" + std::to_string(i), arch.vector_core));

    // Access core pool
    for (uint32_t i = 0; i < arch.access_cores; i++)
        engine.register_unit(std::make_unique<AccessUnit>(
            "access_core_" + std::to_string(i), arch.access_core));

    // ── Ops + scheduler ──────────────────────────────────────────────────
    OpRegistry reg;
    register_builtin_ops(reg, arch);

    Scheduler scheduler(engine, reg, std::move(pp.schedule));
    wire_units(engine, scheduler);

    malloc_trim(0);

    // ── Trace ─────────────────────────────────────────────────────────────
    // --no-trace must silence BOTH the engine-level event log AND the
    // per-unit OP_START/OP_DONE prints. The latter dominate wall-clock time
    // on large schedules (formatting millions of lines), so gate them off.
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++)
        if (Unit* u = engine.get_unit(uid)) u->set_verbose(trace);

    ConsoleLogger logger(engine);
    if (trace) engine.set_trace([&](const Event& e) { logger(e); });

    // ── Run ───────────────────────────────────────────────────────────────
    std::cout << "== simulation start  instructions=" << n_instructions << " ==\n";
    const auto t_sim_start = Clock::now();
    scheduler.launch();
    Cycle final_cycle = engine.run();
    const auto t_sim_end = Clock::now();
    const double sim_ms  = std::chrono::duration<double, std::milli>(
        t_sim_end - t_sim_start).count();

    std::cout << "== simulation done"
              << "  cycle=" << final_cycle
              << "  (" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns)"
              << "  wall=" << sim_ms << " ms"
              << "  outstanding=" << scheduler.outstanding() << " ==\n";

    // ── Metrics (P0.2) ────────────────────────────────────────────────────
    {
        const double clk_hz = arch.clock_ghz * 1e9;
        const double sec    = (final_cycle > 0) ? final_cycle / clk_hz : 0.0;

        std::cout << "\n== metrics ==\n";

        // Per-unit utilization with per-pool breakdown.
        // Single-unit pools print one line (same format as before).
        // Multi-unit pools (e.g. systolic_units=3) print a pool header line
        // followed by one indented line per physical unit, so you can see
        // how evenly work is distributed across the array.
        print_pool_utilization(engine, final_cycle);

        // Roofline.
        const uint64_t macs      = engine.total_macs();
        const uint64_t hbm_bytes = engine.total_hbm_bytes();

        // FIX Bug 3: bidirectional doubles MACs/cell/cycle — include the
        // factor in the compute ceiling so sweep 1c's bidir vs unidir
        // comparison reflects the correct roofline for each config.
        // Without this, bidir compute ceiling is 2× too low, causing
        // compute-bound bidir configs to be misclassified as memory-bound
        // and roofline_efficiency to read ~50% of its true value.
        const double bidir_factor  = arch.systolic.bidirectional ? 2.0 : 1.0;
        const double peak_macs_cyc =
            static_cast<double>(arch.systolic.rows) * arch.systolic.cols
            * arch.systolic_units * bidir_factor;

        const double hbm_bpc = arch.hbm_bytes_per_cycle() * arch.dma.channels;
        const double compute_cyc = peak_macs_cyc > 0 ? std::ceil(macs / peak_macs_cyc) : 0.0;
        const double mem_cyc     = hbm_bpc > 0 ? std::ceil(hbm_bytes / hbm_bpc) : 0.0;
        const double bound       = std::max(compute_cyc, mem_cyc);
        std::cout << "  MACs=" << macs << "   HBM_bytes=" << hbm_bytes << "\n";
        std::cout << "  roofline: compute=" << compute_cyc << "cyc"
                  << "  memory=" << mem_cyc << "cyc"
                  << "  bound=" << bound
                  << " (" << (compute_cyc >= mem_cyc ? "compute" : "memory") << "-bound)\n";
        if (final_cycle > 0)
            std::cout << "  roofline_efficiency="
                      << (100.0 * bound / static_cast<double>(final_cycle)) << "%\n";

        // SRAM pressure (P1.2).
        if (arch.model_sram)
            std::cout << "  sram: capacity=" << engine.sram_capacity()
                      << "B  peak=" << engine.sram_peak()
                      << "B  spills=" << engine.sram_spills() << "\n";

        // Throughput (LLaMA workloads only — we know the token count there).
        if (pp.used_llama) {
            uint64_t tokens = 0;
            if      (pp.llama_cfg.mode == "decode")  tokens = pp.llama_cfg.generation_steps;
            else if (pp.llama_cfg.mode == "prefill") tokens = pp.llama_cfg.prompt_len;
            else                                      tokens = pp.llama_cfg.seq_len;
            std::cout << "  TTFT=" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns\n";
            if (tokens > 0 && sec > 0)
                std::cout << "  throughput=" << (tokens / sec) << " tok/s"
                          << "  (" << tokens << " tokens)\n";
        }
    }

    // ── Output ────────────────────────────────────────────────────────────
    std::cout << "\n";
    if (pp.used_tiler) {
        const auto& wl = pp.tile_decomp.workload;
        std::cout << "Generated tiled GEMM schedule for \"" << wl.dst_c
                  << "\" [" << wl.M << "x" << wl.N << "] with "
                  << pp.tile_decomp.tiles.size() << " array execution(s)\n";
    } else if (pp.used_llama) {
        std::cout << "Generated LLaMA schedule instructions=" << n_instructions << "\n";
    } else {
        std::cout << "Schedule instructions=" << n_instructions << "\n";
    }

    return scheduler.all_done() ? 0 : 1;
}