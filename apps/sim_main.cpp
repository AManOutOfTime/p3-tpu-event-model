#include "core/event_engine.h"
#include "core/logger.h"
#include "core/tensor_store.h"
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
#include <iostream>
#include <string>
#include <chrono>

using namespace sim;

static double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Discards everything written to it (used to silence per-unit trace output
// when --no-trace is set, without touching every unit's print statements).
struct NullBuffer : std::streambuf {
    int overflow(int c) override { return c; }
};

static uint32_t precision_bytes(const std::string& precision) {
    if (precision == "FP8") return 1;
    if (precision == "FP32") return 4;
    return 2;
}

// ---------------------------------------------------------------------------
// Wire scheduler + tensor store into every unit
// ---------------------------------------------------------------------------
static void wire_units(EventEngine& engine, Scheduler& sched, TensorStore& ts) {
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<DelayUnit*>   (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<SystolicUnit*>(u)) { x->set_scheduler(&sched);
                                                         x->set_tensor_store(&ts); continue; }
        if (auto* x = dynamic_cast<DmaUnit*>     (u)) { x->set_scheduler(&sched);
                                                         x->set_tensor_store(&ts); continue; }
        if (auto* x = dynamic_cast<VectorUnit*>  (u)) { x->set_scheduler(&sched);
                                                         x->set_tensor_store(&ts); continue; }
        if (auto* x = dynamic_cast<AccessUnit*>  (u)) { x->set_scheduler(&sched);
                                                         x->set_tensor_store(&ts); continue; }
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
    TensorStore       ts;
    Schedule          schedule;
    TileDecomposition tile_decomp;
    bool              used_tiler = false;
    bool              used_llama = false;

    if (!llama_path.empty()) {
        LlamaScheduleConfig llama_cfg = llama_config_from_yaml_file(llama_path);
        llama_cfg.dtype_bytes = precision_bytes(arch.systolic.precision);
        if (llama_cfg.sram_kv_capacity_kb == 0) {
            llama_cfg.sram_kv_capacity_kb = arch.sram.ibuf_kb + arch.sram.obuf_kb;
        }
        auto _t0 = std::chrono::steady_clock::now();
        Schedule _raw = build_llama_schedule(llama_cfg);
        std::cerr << "[timing] build_llama_schedule: " << secs_since(_t0)
                  << "s  raw_instrs=" << _raw.instructions.size() << "\n";
        auto _t1 = std::chrono::steady_clock::now();
        schedule = Tiler::expand_gemm_subtiles(_raw, arch);
        std::cerr << "[timing] expand_gemm_subtiles: " << secs_since(_t1)
                  << "s  expanded_instrs=" << schedule.instructions.size() << "\n";
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
        schedule = Tiler::expand_gemm_subtiles(schedule, arch);
    }

    // ── Build engine ─────────────────────────────────────────────────────
    EventEngine engine(arch.clock_ghz);

    // Per-unit trace output goes to std::cout only when tracing is enabled;
    // otherwise it is discarded.  (--no-trace previously only silenced the
    // engine event hook, leaving millions of per-unit prints on the hot path.)
    NullBuffer    null_buf;
    std::ostream  null_os(&null_buf);
    std::ostream& unit_os = trace ? std::cout : null_os;

    // Systolic/MXU pool
    for (uint32_t i = 0; i < arch.systolic_units; i++)
        engine.register_unit(std::make_unique<SystolicUnit>(
            "systolic_" + std::to_string(i), arch.systolic, nullptr, &ts, unit_os));

    // DMA channel pool
    for (uint32_t i = 0; i < arch.dma.channels; i++)
        engine.register_unit(std::make_unique<DmaUnit>(
            "dma_" + std::to_string(i), arch, &ts, nullptr, unit_os));

    // Vector core pool
    for (uint32_t i = 0; i < arch.vector_cores; i++)
        engine.register_unit(std::make_unique<VectorUnit>(
            "vector_core_" + std::to_string(i), arch.vector_core, nullptr, &ts, unit_os));

    // Access core pool
    for (uint32_t i = 0; i < arch.access_cores; i++)
        engine.register_unit(std::make_unique<AccessUnit>(
            "access_core_" + std::to_string(i), arch.access_core, nullptr, &ts, unit_os));

    // ── Ops + scheduler ───────────────────────────────────────────────────
    OpRegistry reg;
    register_builtin_ops(reg, arch);

    Scheduler scheduler(engine, reg, schedule);
    wire_units(engine, scheduler, ts);

    // ── Trace ─────────────────────────────────────────────────────────────
    ConsoleLogger logger(engine);
    if (trace) engine.set_trace([&](const Event& e) { logger(e); });

    // ── Run ───────────────────────────────────────────────────────────────
    std::cout << "== simulation start  instructions="
              << schedule.instructions.size() << " ==\n";
    auto _trun = std::chrono::steady_clock::now();
    scheduler.launch();
    std::cerr << "[timing] scheduler.launch: " << secs_since(_trun) << "s\n";
    auto _teng = std::chrono::steady_clock::now();
    Cycle final_cycle = engine.run();
    std::cerr << "[timing] engine.run: " << secs_since(_teng) << "s\n";
    std::cout << "== simulation done"
              << "  cycle=" << final_cycle
              << "  (" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns)"
              << "  outstanding=" << scheduler.outstanding() << " ==\n";

    // ── Output ────────────────────────────────────────────────────────
    if (used_tiler) {
        const auto& wl = tile_decomp.workload;
        std::cout << "\nGenerated tiled GEMM schedule for \"" << wl.dst_c
                  << "\" [" << wl.M << "x" << wl.N << "] with "
                  << tile_decomp.tiles.size() << " array execution(s)\n";
    } else if (used_llama) {
        std::cout << "\nGenerated LLaMA schedule instructions="
                  << schedule.instructions.size() << "\n";
    } else {
        std::cout << "\nSchedule instructions=" << schedule.instructions.size() << "\n";
    }

    return scheduler.all_done() ? 0 : 1;
}
