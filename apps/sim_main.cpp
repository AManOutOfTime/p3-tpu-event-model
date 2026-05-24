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
#include "units/vector_unit.h"
#include "units/access_unit.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>

using namespace sim;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Resolve a symbolic dimension (Br, Bc, d_k, d_head) to a concrete value.
static uint32_t resolve(const ParamMap& p, const std::string& key,
                        const ArchConfig& arch, uint32_t def = 0) {
    int64_t ival = pget_int(p, key, -1);
    if (ival >= 0) return static_cast<uint32_t>(ival);
    std::string s = pget_str(p, key);
    if (s == "Br" || s == "Bc")           return arch.systolic.rows;
    if (s == "d_k" || s == "d_head")      return arch.systolic.d_head;
    return def;
}

static uint32_t dtype_bytes(const std::string& p) {
    if (p == "FP8") return 1; if (p == "FP32") return 4; return 2;
}

// ---------------------------------------------------------------------------
// Op registration — one handler per named op.
// Each handler builds a typed payload and schedules OP_START on the right unit.
// ---------------------------------------------------------------------------
static void register_all_ops(OpRegistry& reg, const ArchConfig arch) {

    // ── delay (legacy backward-compat) ───────────────────────────────────
    reg.register_op("delay", [](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty())
            throw std::runtime_error("delay: unknown unit '"+ctx.inst.unit+"'");
        Cycle lat = (Cycle)pget_int(ctx.inst.params,"latency_cycles",10);
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=static_cast<int64_t>(lat);
        ctx.engine.schedule(std::move(e));
    });

    // ─────────────────────────────────────────────────────────────────────
    // DMA ops
    // ─────────────────────────────────────────────────────────────────────

    // dma_load  HBM → IBUF
    // dma_store OBUF → HBM
    // Both use HBM bandwidth model: hbm_latency + ceil(bytes / bw)
    auto hbm_op = [&arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error(ctx.inst.op+": unknown unit");
        const auto& p = ctx.inst.params;
        DmaTransfer xfer;
        xfer.on_chip = false;
        xfer.src_buf = pget_str(p, "source");
        xfer.dst_buf = pget_str(p, "destination");
        uint32_t rows = resolve(p, "rows",   arch);
        uint32_t cols = resolve(p, "cols",   arch);
        uint32_t len  = resolve(p, "length", arch);
        xfer.bytes = (rows && cols)
            ? (uint64_t)rows * cols * dtype_bytes(arch.systolic.precision)
            : (uint64_t)len  * dtype_bytes(arch.systolic.precision);
        // Latency
        double bw = arch.hbm_bytes_per_cycle() * arch.dma.channels;
        Cycle lat = (Cycle)arch.hbm.latency_cycles +
                    (Cycle)std::ceil((double)xfer.bytes / bw);
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=xfer;
        ctx.engine.schedule(std::move(e));
    };
    reg.register_op("dma_load",  hbm_op);
    reg.register_op("dma_store", hbm_op);

    // dma_stage  IBUF → systolic_array register  (on-chip SRAM read)
    // Latency = ceil(bytes / banking_factor)   — no HBM penalty
    reg.register_op("dma_stage", [&arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("dma_stage: unknown unit");
        const auto& p = ctx.inst.params;
        DmaTransfer xfer;
        xfer.on_chip = true;
        xfer.src_buf = pget_str(p, "source");
        xfer.dst_buf = pget_str(p, "destination");
        uint32_t rows = resolve(p, "rows",   arch);
        uint32_t cols = resolve(p, "cols",   arch);
        xfer.bytes = (uint64_t)rows * cols * dtype_bytes(arch.systolic.precision);
        Cycle lat = (Cycle)std::ceil((double)xfer.bytes / arch.sram.banking_factor);
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=xfer;
        ctx.engine.schedule(std::move(e));
    });

    // ─────────────────────────────────────────────────────────────────────
    // Access core ops
    // ─────────────────────────────────────────────────────────────────────

    // init_fill  Fill buffer with constant value
    // Latency = ceil(elements / access_core.bandwidth)
    reg.register_op("init_fill", [&arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("init_fill: unknown unit");
        const auto& p = ctx.inst.params;
        uint32_t rows = resolve(p, "rows",   arch);
        uint32_t cols = resolve(p, "cols",   arch);
        uint32_t len  = resolve(p, "length", arch);
        AccessOp op;
        op.kind     = "init_fill";
        op.elements = (rows && cols) ? (uint64_t)rows*cols : (uint64_t)len;
        Cycle lat = (Cycle)std::ceil((double)op.elements / arch.access_core.bandwidth);
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=op;
        ctx.engine.schedule(std::move(e));
    });

    // transpose  In-place transpose of a matrix tile
    // Latency = ceil(input_rows × input_cols / access_core.bandwidth)
    reg.register_op("transpose", [&arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("transpose: unknown unit");
        const auto& p = ctx.inst.params;
        uint32_t rows = resolve(p, "input_rows", arch);
        uint32_t cols = resolve(p, "input_cols", arch);
        AccessOp op;
        op.kind     = "transpose";
        op.elements = (uint64_t)rows * cols;
        Cycle lat = (Cycle)std::ceil((double)op.elements / arch.access_core.bandwidth);
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=op;
        ctx.engine.schedule(std::move(e));
    });

    // ─────────────────────────────────────────────────────────────────────
    // Systolic array op
    // ─────────────────────────────────────────────────────────────────────

    // gemm  C[M×N] = A[M×K] × B[K×N]
    // Latency = tiles_m × tiles_n × (K + fill_latency)
    reg.register_op("gemm", [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit("systolic");
        if (t == INVALID_UNIT) throw std::runtime_error("gemm: systolic not found");
        const auto& p = ctx.inst.params;
        GemmShape s;
        s.M     = resolve(p, "M", arch, arch.systolic.rows);
        s.K     = resolve(p, "K", arch, arch.systolic.d_head);
        s.N     = resolve(p, "N", arch, arch.systolic.cols);
        s.src_a = pget_str(p, "source_a");
        s.src_b = pget_str(p, "source_b");
        s.dst_c = pget_str(p, "destination");
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=s; ctx.engine.schedule(std::move(e));
    });

    // ─────────────────────────────────────────────────────────────────────
    // Vector core ops
    // All use: latency = passes × groups + exp_ops × exp_latency × groups
    //          where groups = ceil(elements / simd_width)
    // ─────────────────────────────────────────────────────────────────────

    // Helper: build and schedule a VectorOp event
    auto vec_op = [&arch](const std::string& kind,
                          uint32_t passes, uint32_t exp_ops,
                          const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error(kind+": unknown unit");
        const auto& p = ctx.inst.params;

        VectorOp op;
        op.kind    = kind;
        op.passes  = passes;
        op.exp_ops = exp_ops;

        // Element count from rows×cols or length param
        uint32_t rows = resolve(p, "rows",   arch);
        uint32_t cols = resolve(p, "cols",   arch);
        uint32_t len  = resolve(p, "length", arch);
        op.elements = (rows && cols) ? (uint64_t)rows*cols : (uint64_t)len;
        if (!op.elements) op.elements = arch.systolic.rows; // safe fallback

        // Compute latency
        double groups = std::ceil((double)op.elements / arch.vector_core.simd_width);
        Cycle lat = (Cycle)(passes * groups)
                  + (Cycle)(exp_ops * arch.vector_core.exp_latency * groups);

        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=op;
        ctx.engine.schedule(std::move(e));
    };

    // scale — element-wise multiply by scalar or row vector
    // Br×d_head matrix: passes=1, exp_ops=0
    reg.register_op("scale",         [vec_op](const IssueCtx& ctx){ vec_op("scale",         1, 0, ctx); });
    reg.register_op("rowmax",        [vec_op](const IssueCtx& ctx){ vec_op("rowmax",        1, 0, ctx); });
    reg.register_op("update_rowmax", [vec_op](const IssueCtx& ctx){ vec_op("update_rowmax", 1, 1, ctx); });
    reg.register_op("exp_shift",     [vec_op](const IssueCtx& ctx){ vec_op("exp_shift",     1, 1, ctx); });
    reg.register_op("update_rowsum", [vec_op](const IssueCtx& ctx){ vec_op("update_rowsum", 1, 0, ctx); });
    reg.register_op("accumulate",    [vec_op](const IssueCtx& ctx){ vec_op("accumulate",    1, 0, ctx); });
    reg.register_op("normalize",     [vec_op](const IssueCtx& ctx){ vec_op("normalize",     1, 0, ctx); });
    reg.register_op("logsumexp",     [vec_op](const IssueCtx& ctx){ vec_op("logsumexp",     1, 1, ctx); });
}

// ---------------------------------------------------------------------------
// Wire scheduler + tensor store into every unit
// ---------------------------------------------------------------------------
static void wire_units(EventEngine& engine, Scheduler& sched, TensorStore& ts) {
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<DelayUnit*>   (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<SystolicUnit*>(u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<DmaUnit*>     (u)) { x->set_scheduler(&sched);
                                                         x->set_tensor_store(&ts); continue; }
        if (auto* x = dynamic_cast<VectorUnit*>  (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<AccessUnit*>  (u)) { x->set_scheduler(&sched); continue; }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string config_path = "configs/default.yaml";
    std::string sched_path  = "schedules/dummy_example.yaml";
    bool        trace       = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--config"   && i+1 < argc) config_path = argv[++i];
        else if (a == "--schedule" && i+1 < argc) sched_path  = argv[++i];
        else if (a == "--no-trace")               trace = false;
        else {
            std::cerr << "Usage: sim_main [--config FILE] [--schedule FILE] [--no-trace]\n";
            return 1;
        }
    }

    ArchConfig arch     = ArchConfig::from_yaml_file(config_path);
    Schedule   schedule = Schedule::from_yaml_file(sched_path);

    std::cout << "clock=" << arch.clock_ghz << " GHz"
              << "  systolic=" << arch.systolic.rows << "x" << arch.systolic.cols
              << " " << (arch.systolic.bidirectional ? "bidir" : "unidir")
              << "  d_head=" << arch.systolic.d_head << "\n"
              << "hbm_bw=" << arch.hbm.bandwidth_tb_s << " TB/s"
              << "  hbm_bpc=" << arch.hbm_bytes_per_cycle()
              << "  hbm_lat=" << arch.hbm.latency_cycles << "\n"
              << "vec_simd=" << arch.vector_core.simd_width
              << "  exp_lat=" << arch.vector_core.exp_latency
              << "  access_bw=" << arch.access_core.bandwidth << "\n\n";

    // ── TensorStore — pre-seed FA2 buffers ─────────────────────────────
    TensorStore ts;
    const uint32_t Br = arch.systolic.rows;
    const uint32_t Bc = arch.systolic.cols;
    const uint32_t DH = arch.systolic.d_head;

    // IBUF: Q/K/V/K^T tiles (represent post-DMA state)
    ts.init_random("shared_ibuf.Q_tile",   (size_t)Br*DH, -1.f, 1.f, 1);
    ts.init_random("shared_ibuf.K_tile",   (size_t)Bc*DH, -1.f, 1.f, 2);
    ts.init_random("shared_ibuf.K_tile_T", (size_t)DH*Bc, -1.f, 1.f, 3);
    ts.init_random("shared_ibuf.V_tile",   (size_t)Bc*DH, -1.f, 1.f, 4);
    ts.init_zeros ("shared_ibuf.P_tile",   (size_t)Br*Bc);

    // OBUF: accumulators and running statistics
    ts.init_zeros  ("shared_obuf.O_acc",     (size_t)Br*DH);
    ts.init_neg_inf("shared_obuf.m",         (size_t)Br);
    ts.init_zeros  ("shared_obuf.l",         (size_t)Br);
    ts.init_zeros  ("shared_obuf.correction",(size_t)Br);

    // Systolic array registers — Q_operand pre-seeded from Q_tile so
    // the GEMM at id=8 has real values. (id=4 dma_stage will update it
    // at OP_DONE when the DMA fires, but GEMM depends_on [4,7] correctly.)
    ts.init_random("systolic_array.Q_operand", (size_t)Br*DH, -1.f, 1.f, 1);
    ts.init_zeros ("systolic_array.P_operand", (size_t)Br*Bc);

    // Vector scratch space
    ts.init_zeros("vector_scratch.rowmax_tmp", (size_t)Br);

    // ── Build engine ─────────────────────────────────────────────────────
    EventEngine engine(arch.clock_ghz);

    // Single systolic array
    engine.register_unit(std::make_unique<SystolicUnit>("systolic",
                          arch.systolic, nullptr, &ts));

    // DMA channel pool
    for (uint32_t i = 0; i < arch.dma.channels; i++)
        engine.register_unit(std::make_unique<DmaUnit>(
            "dma_" + std::to_string(i), arch, &ts));

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
    register_all_ops(reg, arch);

    Scheduler scheduler(engine, reg, schedule);
    wire_units(engine, scheduler, ts);

    // ── Trace ─────────────────────────────────────────────────────────────
    ConsoleLogger logger(engine);
    if (trace) engine.set_trace([&](const Event& e) { logger(e); });

    // ── Run ───────────────────────────────────────────────────────────────
    std::cout << "== simulation start  instructions="
              << schedule.instructions.size() << " ==\n";
    scheduler.launch();
    Cycle final_cycle = engine.run();
    std::cout << "== simulation done"
              << "  cycle=" << final_cycle
              << "  (" << cycles_to_ns(final_cycle, arch.clock_ghz) << " ns)"
              << "  outstanding=" << scheduler.outstanding() << " ==\n";

    // Print key output buffers
    if (ts.has("shared_obuf.S_tile"))
        ts.print("shared_obuf.S_tile", Br, Bc, 4);
    if (ts.has("shared_obuf.O_tile"))
        ts.print("shared_obuf.O_tile", Br, DH, 4);

    return scheduler.all_done() ? 0 : 1;
}
