#include "core/event_engine.h"
#include "core/logger.h"
#include "core/tensor_store.h"
#include "config/arch_config.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/delay_unit.h"
#include "units/systolic_unit.h"
#include "units/dma_unit.h"
#include "units/buffer_unit.h"
#include "units/vector_unit.h"
#include "units/access_unit.h"
#include <iostream>
#include <string>
#include <unordered_map>

using namespace sim;

static uint32_t dtype_bytes(const std::string& precision) {
    if (precision == "FP8")  return 1;
    if (precision == "FP32") return 4;
    return 2;  // FP16 / BF16
}

static void register_all_ops(OpRegistry& reg, const ArchConfig& arch) {

    // delay — backward compat
    reg.register_op("delay", [](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        if (t == INVALID_UNIT)
            throw std::runtime_error("delay: unknown unit '" + ctx.inst.unit + "'");
        Event e;
        e.type    = EventType::OP_START;
        e.target  = t;
        e.cycle   = ctx.engine.current_cycle();
        e.instr   = ctx.inst.id;
        e.label   = ctx.inst.label;
        e.payload = static_cast<int64_t>(pget_int(ctx.inst.params, "latency_cycles", 10));
        ctx.engine.schedule(std::move(e));
    });

    // gemm — reads M/K/N plus optional buffer names for actual computation:
    //   source_a    : TensorStore key for A [M×K]  (e.g. "systolic_array.Q_operand")
    //   source_b    : TensorStore key for B [K×N]  (e.g. "shared_ibuf.K_tile_T")
    //   destination : TensorStore key for C [M×N]  (e.g. "shared_obuf.S_tile")
    // If any of these are absent the unit is timing-only (backward compatible).
    reg.register_op("gemm", [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        if (t == INVALID_UNIT) throw std::runtime_error("gemm: unknown unit");
        GemmShape s;
        s.M     = (uint32_t)pget_int(ctx.inst.params, "M", arch.systolic.rows);
        s.K     = (uint32_t)pget_int(ctx.inst.params, "K", arch.systolic.rows);
        s.N     = (uint32_t)pget_int(ctx.inst.params, "N", arch.systolic.cols);
        s.src_a = pget_str(ctx.inst.params, "source_a");
        s.src_b = pget_str(ctx.inst.params, "source_b");
        s.dst_c = pget_str(ctx.inst.params, "destination");
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=s; ctx.engine.schedule(std::move(e));
    });

    // dma_load / dma_store
    auto dma_op = [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        if (t == INVALID_UNIT) throw std::runtime_error("dma: unknown unit");
        DmaTransfer xfer;
        xfer.bytes = (uint64_t)pget_int(ctx.inst.params, "bytes", 0);
        if (xfer.bytes == 0) {
            uint64_t r = (uint64_t)pget_int(ctx.inst.params, "rows", 0);
            uint64_t c = (uint64_t)pget_int(ctx.inst.params, "cols", 0);
            xfer.bytes = r * c * dtype_bytes(arch.systolic.precision);
        }
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=xfer; ctx.engine.schedule(std::move(e));
    };
    reg.register_op("dma_load",  dma_op);
    reg.register_op("dma_store", dma_op);

    // sram_read / sram_write
    auto sram_op = [&arch](bool is_write) {
        return [&arch, is_write](const IssueCtx& ctx) {
            UnitId t = ctx.engine.find_unit(ctx.inst.unit);
            if (t == INVALID_UNIT) throw std::runtime_error("sram: unknown unit");
            SramAccess a; a.is_write = is_write;
            a.bytes = (uint64_t)pget_int(ctx.inst.params, "bytes", 0);
            if (a.bytes == 0) {
                uint64_t r = (uint64_t)pget_int(ctx.inst.params, "rows", 0);
                uint64_t c = (uint64_t)pget_int(ctx.inst.params, "cols", 0);
                a.bytes = r * c * dtype_bytes(arch.systolic.precision);
            }
            Event e; e.type=EventType::OP_START; e.target=t;
            e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
            e.payload=a; ctx.engine.schedule(std::move(e));
        };
    };
    reg.register_op("sram_read",  sram_op(false));
    reg.register_op("sram_write", sram_op(true));

    // Vector ops
    struct VD { uint32_t passes; uint32_t exp_ops; };
    const std::unordered_map<std::string, VD> vec_defs = {
        {"scale",{1,0}},{"add",{1,0}},{"accumulate",{1,0}},{"normalize",{1,0}},
        {"rowmax",{1,0}},{"rowsum",{1,0}},{"exp",{1,1}},{"rope",{1,0}},
        {"softmax",{3,1}},{"layer_norm",{2,0}},{"logsumexp",{1,1}},
    };
    for (auto& [name, def] : vec_defs) {
        reg.register_op(name, [name, def](const IssueCtx& ctx) {
            UnitId t = ctx.engine.find_unit(ctx.inst.unit);
            if (t == INVALID_UNIT) throw std::runtime_error(name + ": unknown unit");
            VectorOp op; op.kind = name;
            op.elements = (uint64_t)pget_int(ctx.inst.params, "elements", 0);
            if (op.elements == 0) {
                uint64_t r = (uint64_t)pget_int(ctx.inst.params, "rows",   0);
                uint64_t c = (uint64_t)pget_int(ctx.inst.params, "cols",   0);
                uint64_t l = (uint64_t)pget_int(ctx.inst.params, "length", 0);
                op.elements = (r && c) ? r*c : l;
            }
            op.passes  = (uint32_t)pget_int(ctx.inst.params, "passes",  def.passes);
            op.exp_ops = (uint32_t)pget_int(ctx.inst.params, "exp_ops", def.exp_ops);
            Event e; e.type=EventType::OP_START; e.target=t;
            e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
            e.payload=op; ctx.engine.schedule(std::move(e));
        });
    }

    // Access-core ops
    for (auto* n : {"transpose","scatter","gather","init_fill"}) {
        std::string name = n;
        reg.register_op(name, [name](const IssueCtx& ctx) {
            UnitId t = ctx.engine.find_unit(ctx.inst.unit);
            if (t == INVALID_UNIT) throw std::runtime_error(name + ": unknown unit");
            AccessOp op; op.kind = name;
            op.elements = (uint64_t)pget_int(ctx.inst.params, "elements", 0);
            if (op.elements == 0) {
                uint64_t r = (uint64_t)pget_int(ctx.inst.params, "rows",   0);
                uint64_t c = (uint64_t)pget_int(ctx.inst.params, "cols",   0);
                uint64_t l = (uint64_t)pget_int(ctx.inst.params, "length", 0);
                op.elements = (r && c) ? r*c : l;
            }
            Event e; e.type=EventType::OP_START; e.target=t;
            e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
            e.payload=op; ctx.engine.schedule(std::move(e));
        });
    }
}

static void wire_scheduler(EventEngine& engine, Scheduler& sched) {
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<DelayUnit*>   (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<SystolicUnit*>(u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<DmaUnit*>     (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<BufferUnit*>  (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<VectorUnit*>  (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<AccessUnit*>  (u)) { x->set_scheduler(&sched); continue; }
    }
}

int main(int argc, char** argv) {
    std::string config_path = "configs/default.yaml";
    std::string sched_path  = "schedules/dummy_example.yaml";
    bool        trace       = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--config"   && i+1 < argc) config_path = argv[++i];
        else if (a == "--schedule" && i+1 < argc) sched_path  = argv[++i];
        else if (a == "--no-trace")               trace = false;
        else { std::cerr << "Usage: sim_main [--config FILE] [--schedule FILE] [--no-trace]\n"; return 1; }
    }

    ArchConfig arch     = ArchConfig::from_yaml_file(config_path);
    Schedule   schedule = Schedule::from_yaml_file(sched_path);

    std::cout << "clock=" << arch.clock_ghz << " GHz"
              << "  systolic=" << arch.systolic.rows << "x" << arch.systolic.cols
              << " " << (arch.systolic.bidirectional ? "bidir" : "unidir")
              << "  precision=" << arch.systolic.precision << "\n"
              << "hbm_bw=" << arch.hbm.bandwidth_tb_s << " TB/s"
              << "  hbm_bpc=" << arch.hbm_bytes_per_cycle() << " B/cyc"
              << "  hbm_lat=" << arch.hbm.latency_cycles << " cyc\n"
              << "dma_ch=" << arch.dma.channels
              << "  ibuf=" << arch.sram.ibuf_kb << " KB"
              << "  obuf=" << arch.sram.obuf_kb << " KB"
              << "  bank_factor=" << arch.sram.banking_factor << "\n"
              << "vec_simd=" << arch.vector_core.simd_width
              << "  exp_lat=" << arch.vector_core.exp_latency
              << "  access_bw=" << arch.access_core.bandwidth << " elem/cyc\n\n";

    // ── TensorStore ─────────────────────────────────────────────────────
    // Pre-populate dummy input matrices for any gemm ops that name buffers.
    // Real workloads would DMA-load actual weights/activations here.
    // Buffer names mirror the source_a / source_b params in schedule YAMLs.
    TensorStore ts;

    // Pre-stage buffers that schedules reference as source_a.
    // "systolic_array.Q_operand" is the weight/activation pre-loaded into
    // the array's input register by the preceding DMA issue instruction.
    const uint32_t SA = arch.systolic.rows;   // square array dimension
    ts.init_random("systolic_array.Q_operand", SA * SA, -1.0f, 1.0f, /*seed=*/1);

    // IBUF buffers that schedules reference as source_b.
    ts.init_random("shared_ibuf.K_tile_T",  SA * SA, -1.0f, 1.0f, /*seed=*/2);
    ts.init_random("shared_ibuf.V_tile",    SA * SA, -1.0f, 1.0f, /*seed=*/3);
    ts.init_random("shared_ibuf.P_tile",    SA * SA, -1.0f, 1.0f, /*seed=*/4);

    // Generic "A" / "B" for simple GEMM schedules.
    ts.init_random("A", SA * SA, -1.0f, 1.0f, /*seed=*/5);
    ts.init_random("B", SA * SA, -1.0f, 1.0f, /*seed=*/6);

    // ── Build engine and register hardware units ─────────────────────────
    EventEngine engine(arch.clock_ghz);
    engine.register_unit(std::make_unique<SystolicUnit>("systolic",     arch.systolic,
                                                         nullptr, &ts));
    engine.register_unit(std::make_unique<BufferUnit>  ("shared_ibuf",  arch.sram));
    engine.register_unit(std::make_unique<BufferUnit>  ("shared_obuf",  arch.sram));
    engine.register_unit(std::make_unique<VectorUnit>  ("tandem_1",     arch.vector_core));
    engine.register_unit(std::make_unique<VectorUnit>  ("tandem_2",     arch.vector_core));
    engine.register_unit(std::make_unique<VectorUnit>  ("tandem_3",     arch.vector_core));
    engine.register_unit(std::make_unique<VectorUnit>  ("vector_core",  arch.vector_core));
    engine.register_unit(std::make_unique<AccessUnit>  ("access_core_1",arch.access_core));
    engine.register_unit(std::make_unique<AccessUnit>  ("access_core_2",arch.access_core));
    engine.register_unit(std::make_unique<DmaUnit>     ("dma",          arch));

    OpRegistry reg;
    register_all_ops(reg, arch);

    Scheduler scheduler(engine, reg, schedule);
    wire_scheduler(engine, scheduler);

    ConsoleLogger logger(engine);
    if (trace) engine.set_trace([&](const Event& e) { logger(e); });

    std::cout << "== simulation start  instructions=" << schedule.instructions.size() << " ==\n";
    scheduler.launch();
    Cycle final_cycle = engine.run();
    std::cout << "== simulation done"
              << "  cycle=" << final_cycle
              << "  (" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns)"
              << "  outstanding=" << scheduler.outstanding() << " ==\n";

    // ── Print any OBUF / output buffers that were written ────────────────
    for (const char* name : {"shared_obuf.S_tile", "shared_obuf.Temp",
                              "shared_obuf.O_tile", "C"}) {
        if (ts.has(name)) {
            // Infer rows from schedule params if possible; use SA as fallback.
            ts.print(name, SA, SA, /*max_rows=*/4);
        }
    }

    return scheduler.all_done() ? 0 : 1;
}
