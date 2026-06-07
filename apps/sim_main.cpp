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
#include "units/systolic_unit.h"
#include "units/dma_unit.h"
#include "units/buffer_unit.h"
#include "units/vector_unit.h"
#include "units/access_unit.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace sim;

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

    // ── Schedule ───────────────────────────────────────────────────────
    Schedule            schedule;
    TileDecomposition   tile_decomp;
    LlamaScheduleConfig llama_cfg;
    bool                used_tiler = false;
    bool                used_llama = false;

    if (!llama_path.empty()) {
        llama_cfg = llama_config_from_yaml_file(llama_path);
        llama_cfg.dtype_bytes = precision_bytes(arch.systolic.precision);
        if (llama_cfg.sram_kv_capacity_kb == 0) {
            llama_cfg.sram_kv_capacity_kb = arch.sram.ibuf_kb + arch.sram.obuf_kb;
        }
        // Build in "minimal" metadata mode when tracing is off — drops
        // trace-only labels/buffer-name strings to cut host RAM on large
        // schedules (timing-neutral; the timing model reads only numeric params).
        schedule = Tiler::expand_gemm_subtiles(
            build_llama_schedule(llama_cfg, /*minimal=*/!trace), arch);
        used_llama = true;
        std::cout << "llama_mode=" << llama_cfg.mode
                  << "  q_heads=" << llama_cfg.num_q_heads
                  << "  kv_heads=" << llama_cfg.num_kv_heads
                  << "  gqa_group=" << (llama_cfg.num_q_heads / llama_cfg.num_kv_heads)
                  << "  kv_cache=" << (llama_cfg.kv_cache_enabled ? "on" : "off")
                  << ":" << llama_cfg.kv_cache_location << "\n\n";
    } else if (!workload_path.empty()) {
        // --workload: tiler generates STAGE+GEMM instructions automatically
        WorkloadGemm wl = Tiler::from_yaml_file(workload_path);
        tile_decomp     = Tiler::decompose(wl, arch);
        Tiler::print_decomposition(tile_decomp);
        schedule.instructions = tile_decomp.instructions;
        used_tiler = true;
    } else {
        // --schedule: hand-written YAML
        schedule = Schedule::from_yaml_file(sched_path);
        schedule = Tiler::expand_gemm_subtiles(std::move(schedule), arch);
    }

    // ── Build engine ─────────────────────────────────────────────────────
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

    // ── Ops + scheduler ───────────────────────────────────────────────────
    OpRegistry reg;
    register_builtin_ops(reg, arch);

    // Move the (potentially multi-million-instruction) schedule into the
    // Scheduler instead of copying it — the local `schedule` is not needed
    // afterward except for its size, which we capture first.
    const size_t n_instructions = schedule.instructions.size();
    Scheduler scheduler(engine, reg, std::move(schedule));
    wire_units(engine, scheduler);

    // ── Trace ─────────────────────────────────────────────────────────────
    // --no-trace must silence BOTH the engine-level event log AND the
    // per-unit OP_START/OP_DONE prints. The latter dominate wall-clock time
    // on large schedules (formatting millions of lines), so gate them off.
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++)
        if (Unit* u = engine.get_unit(uid)) u->set_verbose(trace);

    ConsoleLogger logger(engine);
    if (trace) engine.set_trace([&](const Event& e) { logger(e); });

    // ── Run ───────────────────────────────────────────────────────────────
    std::cout << "== simulation start  instructions="
              << n_instructions << " ==\n";
    scheduler.launch();
    Cycle final_cycle = engine.run();
    std::cout << "== simulation done"
              << "  cycle=" << final_cycle
              << "  (" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns)"
              << "  outstanding=" << scheduler.outstanding() << " ==\n";

    // ── Metrics (P0.2) ─────────────────────────────────────────────────────
    {
        const double clk_hz = arch.clock_ghz * 1e9;
        const double sec    = (final_cycle > 0) ? final_cycle / clk_hz : 0.0;

        std::cout << "\n== metrics ==\n";

        // Per-pool utilization (group physical units by logical name prefix).
        std::map<std::string, std::pair<uint64_t, uint64_t>> pool;  // -> (busy, count)
        for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
            Unit* u = engine.get_unit(uid);
            if (!u) continue;
            const std::string& n = u->name();
            auto pos = n.rfind('_');
            std::string pre = (pos == std::string::npos) ? n : n.substr(0, pos);
            pool[pre].first  += engine.unit_busy_cycles(uid);
            pool[pre].second += 1;
        }
        for (const auto& kv : pool) {
            const uint64_t busy = kv.second.first;
            const uint64_t cnt  = kv.second.second;
            const double util = (final_cycle > 0 && cnt > 0)
                ? 100.0 * static_cast<double>(busy) / (static_cast<double>(final_cycle) * cnt)
                : 0.0;
            std::cout << "  " << kv.first << " x" << cnt
                      << "   busy=" << busy
                      << "   util=" << util << "%\n";
        }

        // Roofline.
        const uint64_t macs      = engine.total_macs();
        const uint64_t hbm_bytes = engine.total_hbm_bytes();
        const double peak_macs_cyc =
            static_cast<double>(arch.systolic.rows) * arch.systolic.cols * arch.systolic_units;
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
        if (used_llama) {
            uint64_t tokens = 0;
            if      (llama_cfg.mode == "decode")  tokens = llama_cfg.generation_steps;
            else if (llama_cfg.mode == "prefill") tokens = llama_cfg.prompt_len;
            else                                  tokens = llama_cfg.seq_len;
            std::cout << "  TTFT=" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns\n";
            if (tokens > 0 && sec > 0)
                std::cout << "  throughput=" << (tokens / sec) << " tok/s"
                          << "  (" << tokens << " tokens)\n";
        }
    }

    // ── Output ────────────────────────────────────────────────────────
    if (used_tiler) {
        const auto& wl = tile_decomp.workload;
        std::cout << "\nGenerated tiled GEMM schedule for \"" << wl.dst_c
                  << "\" [" << wl.M << "x" << wl.N << "] with "
                  << tile_decomp.tiles.size() << " array execution(s)\n";
    } else if (used_llama) {
        std::cout << "\nGenerated LLaMA schedule instructions="
                  << n_instructions << "\n";
    } else {
        std::cout << "\nSchedule instructions=" << n_instructions << "\n";
    }

    return scheduler.all_done() ? 0 : 1;
}
