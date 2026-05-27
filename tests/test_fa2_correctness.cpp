#include <doctest/doctest.h>
#include "core/event_engine.h"
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
#include "units/buffer_unit.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>

using namespace sim;

// ---------------------------------------------------------------------------
// CPU reference: exact attention for a single tile
//   O[i,:] = softmax(Q[i,:] @ K^T / sqrt(d_k)) @ V
//   L[i]   = m[i] + log(l[i])
// ---------------------------------------------------------------------------
static void cpu_attention(
        const std::vector<float>& Q,   // [Br × dk]
        const std::vector<float>& KT,  // [dk × Bc]   (K already transposed)
        const std::vector<float>& V,   // [Bc × dh]
        uint32_t Br, uint32_t Bc, uint32_t dk, uint32_t dh,
        std::vector<float>& ref_O,     // out [Br × dh]
        std::vector<float>& ref_L)     // out [Br]
{
    const float scale = 1.f / std::sqrt(static_cast<float>(dk));

    // S = Q @ KT * scale  [Br × Bc]
    std::vector<float> S(Br * Bc, 0.f);
    for (uint32_t i=0;i<Br;i++)
        for (uint32_t k=0;k<dk;k++) {
            float qi = Q[i*dk+k] * scale;
            for (uint32_t j=0;j<Bc;j++)
                S[i*Bc+j] += qi * KT[k*Bc+j];
        }

    // rowmax, exp, rowsum
    ref_O.assign(Br*dh, 0.f);
    ref_L.assign(Br, 0.f);
    std::vector<float> m(Br, -std::numeric_limits<float>::infinity());
    std::vector<float> l(Br, 0.f);
    std::vector<float> P(Br*Bc);

    for (uint32_t i=0;i<Br;i++)
        for (uint32_t j=0;j<Bc;j++)
            m[i] = std::max(m[i], S[i*Bc+j]);

    for (uint32_t i=0;i<Br;i++) {
        for (uint32_t j=0;j<Bc;j++) {
            P[i*Bc+j] = std::exp(S[i*Bc+j] - m[i]);
            l[i] += P[i*Bc+j];
        }
    }

    // O = P @ V / l
    for (uint32_t i=0;i<Br;i++)
        for (uint32_t j=0;j<Bc;j++)
            for (uint32_t d=0;d<dh;d++)
                ref_O[i*dh+d] += P[i*Bc+j] * V[j*dh+d];
    for (uint32_t i=0;i<Br;i++)
        for (uint32_t d=0;d<dh;d++)
            ref_O[i*dh+d] /= l[i];

    // L = m + log(l)
    for (uint32_t i=0;i<Br;i++)
        ref_L[i] = m[i] + std::log(l[i]);
}

static float max_abs_diff(const std::vector<float>& a,
                           const std::vector<float>& b) {
    float mx = 0.f;
    for (size_t i=0;i<a.size();i++) mx=std::max(mx,std::abs(a[i]-b[i]));
    return mx;
}

// ---------------------------------------------------------------------------
// Minimal sim runner — replicates sim_main setup for a given schedule file.
// Returns true if all instructions completed.
// ---------------------------------------------------------------------------
static bool run_fa2_schedule(const std::string& sched_path,
                              TensorStore& ts,
                              const ArchConfig& arch) {
    Schedule schedule = Schedule::from_yaml_file(sched_path);
    std::ostringstream log;

    EventEngine engine(arch.clock_ghz);
    engine.register_unit(std::make_unique<SystolicUnit>("systolic",
                          arch.systolic, nullptr, &ts));
    engine.register_unit(std::make_unique<BufferUnit>("shared_ibuf",
                          arch.sram, nullptr, &ts));
    engine.register_unit(std::make_unique<BufferUnit>("shared_obuf",
                          arch.sram, nullptr, &ts));
    engine.register_unit(std::make_unique<DmaUnit>("dma_0", arch, &ts));
    for (uint32_t i=0;i<arch.vector_cores;i++)
        engine.register_unit(std::make_unique<VectorUnit>(
            "vector_core_"+std::to_string(i), arch.vector_core, nullptr, nullptr, log));
    for (uint32_t i=0;i<arch.access_cores;i++)
        engine.register_unit(std::make_unique<AccessUnit>(
            "access_core_"+std::to_string(i), arch.access_core, nullptr, &ts));

    OpRegistry reg;

    // Register all ops as delay (for timing) except gemm and access ops
    // which need real compute for correctness verification
    auto delay_op = [](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) return;
        Cycle lat = (Cycle)pget_int(ctx.inst.params,"latency_cycles",10);
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=static_cast<int64_t>(lat); ctx.engine.schedule(std::move(e));
    };

    auto resolve_d = [&arch](const ParamMap& p, const std::string& key) -> uint32_t {
        int64_t v = pget_int(p, key, -1);
        if (v >= 0) return (uint32_t)v;
        std::string s = pget_str(p, key);
        if (s=="Br"||s=="Bc") return arch.systolic.rows;
        if (s=="d_k"||s=="d_head") return arch.systolic.d_head;
        return arch.systolic.rows;
    };

    // real gemm
    reg.register_op("gemm", [&arch, resolve_d](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit("systolic");
        if (t==INVALID_UNIT) return;
        const auto& p = ctx.inst.params;
        GemmShape s;
        s.M = resolve_d(p,"M"); s.K = resolve_d(p,"K"); s.N = resolve_d(p,"N");
        s.src_a = pget_str(p,"source_a"); s.src_b = pget_str(p,"source_b");
        s.dst_c = pget_str(p,"destination");
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=s; ctx.engine.schedule(std::move(e));
    });

    // real weight_load
    reg.register_op("weight_load", [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit("systolic");
        if (t==INVALID_UNIT) return;
        const auto& p = ctx.inst.params;
        WeightLoad wl;
        wl.sa_rows=arch.systolic.rows; wl.sa_cols=arch.systolic.cols;
        wl.dtype_bytes=2; wl.banking_factor=arch.sram.banking_factor;
        wl.src_buf=pget_str(p,"source"); wl.dst_buf=pget_str(p,"destination");
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=wl; ctx.engine.schedule(std::move(e));
    });

    reg.register_op("init_fill", [&arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) return;
        const auto& p = ctx.inst.params;
        // Only use rows×cols if BOTH are explicitly present in params
        bool has_rows = (p.count("rows") > 0);
        bool has_cols = (p.count("cols") > 0);
        bool has_len  = (p.count("length") > 0);
        auto resolve_sym = [&arch](const ParamMap& pm, const std::string& k) -> uint32_t {
            int64_t v = pget_int(pm, k, -1);
            if (v >= 0) return (uint32_t)v;
            std::string s = pget_str(pm, k);
            if (s=="Br"||s=="Bc") return arch.systolic.rows;
            if (s=="d_k"||s=="d_head") return arch.systolic.d_head;
            return 0;
        };
        uint32_t rows = has_rows ? resolve_sym(p,"rows") : 0;
        uint32_t cols = has_cols ? resolve_sym(p,"cols") : 0;
        uint32_t len  = has_len  ? resolve_sym(p,"length") : 0;
        AccessOp op; op.kind="init_fill";
        op.elements = (rows && cols) ? (uint64_t)rows*cols : (uint64_t)len;
        op.dst = pget_str(p,"destination");
        std::string iv = pget_str(p,"init_value");
        op.fill_value = (iv=="-inf") ? -std::numeric_limits<float>::infinity()
                                     : (float)pget_dbl(p,"init_value",0.0);
        Cycle lat=(Cycle)std::ceil((double)op.elements/arch.access_core.bandwidth);
        auto res=ctx.scheduler.reserve_unit_pool(targets,lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=op;
        ctx.engine.schedule(std::move(e));
    });
    reg.register_op("transpose", [&arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) return;
        const auto& p = ctx.inst.params;
        auto resolve_sym = [&arch](const ParamMap& pm, const std::string& k) -> uint32_t {
            int64_t v = pget_int(pm, k, -1);
            if (v >= 0) return (uint32_t)v;
            std::string s = pget_str(pm, k);
            if (s=="Br"||s=="Bc") return arch.systolic.rows;
            if (s=="d_k"||s=="d_head") return arch.systolic.d_head;
            return 0;
        };
        uint32_t rows=resolve_sym(p,"input_rows"), cols=resolve_sym(p,"input_cols");
        AccessOp op; op.kind="transpose"; op.elements=(uint64_t)rows*cols;
        op.src=pget_str(p,"source"); op.dst=pget_str(p,"destination");
        op.input_rows=rows; op.input_cols=cols;
        Cycle lat=(Cycle)std::ceil((double)op.elements/arch.access_core.bandwidth);
        auto res=ctx.scheduler.reserve_unit_pool(targets,lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=op;
        ctx.engine.schedule(std::move(e));
    });

    // real vector ops
    auto vec = [&arch, resolve_d](const std::string& kind,
                                   uint32_t passes, uint32_t exp_ops,
                                   const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) return;
        const auto& p = ctx.inst.params;
        VectorOp op; op.kind=kind; op.passes=passes; op.exp_ops=exp_ops;
        // Only use rows×cols if both keys actually present in params
        auto resolve_sym = [&arch](const ParamMap& pm, const std::string& k) -> uint32_t {
            int64_t v = pget_int(pm, k, -1); if (v>=0) return (uint32_t)v;
            std::string s = pget_str(pm, k);
            if (s=="Br"||s=="Bc") return arch.systolic.rows;
            if (s=="d_k"||s=="d_head") return arch.systolic.d_head;
            return 0;
        };
        bool has_r = p.count("rows")>0, has_c = p.count("cols")>0;
        bool has_l = p.count("length")>0;
        uint32_t rows = has_r ? resolve_sym(p,"rows")   : 0;
        uint32_t cols = has_c ? resolve_sym(p,"cols")   : 0;
        uint32_t len  = has_l ? resolve_sym(p,"length") : 0;
        op.elements=(rows&&cols)?(uint64_t)rows*cols:(uint64_t)len;
        if (!op.elements) op.elements=arch.systolic.rows;
        op.rows=rows; op.cols=cols;
        op.src=pget_str(p,"source"); op.dst=pget_str(p,"destination");
        op.src_a=pget_str(p,"source_a"); op.src_b=pget_str(p,"source_b");
        op.src_scale=pget_str(p,"source_scale");
        // parse scalar: "1/sqrt(d_k)" → numeric
        std::string sc = pget_str(p,"scalar");
        if (!sc.empty()) {
            if (sc.find("sqrt") != std::string::npos)
                op.scalar_val = 1.f / std::sqrt((float)arch.systolic.d_head);
            else { try { op.scalar_val=std::stof(sc); } catch(...){} }
        }
        op.src_m=pget_str(p,"source_m"); op.src_rowmax=pget_str(p,"source_rowmax");
        op.dst_m=pget_str(p,"destination_m");
        op.dst_correction=pget_str(p,"destination_correction");
        op.src_matrix=pget_str(p,"source_matrix");
        op.src_shift=pget_str(p,"source_shift");
        op.src_p=pget_str(p,"source_p");
        op.src_correction=pget_str(p,"source_correction");
        // accept both "source_l_old" (update_rowsum) and "source_l" (logsumexp)
        op.src_l=pget_str(p,"source_l_old");
        if (op.src_l.empty()) op.src_l=pget_str(p,"source_l");
        op.dst_l=pget_str(p,"destination");
        op.src_denom=pget_str(p,"source_denom");
        if (!op.elements) op.elements=arch.systolic.rows;
        double g=std::ceil((double)op.elements/arch.vector_core.simd_width);
        Cycle lat=(Cycle)(passes*g)+(Cycle)(exp_ops*arch.vector_core.exp_latency*g);
        auto res=ctx.scheduler.reserve_unit_pool(targets,lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=op;
        ctx.engine.schedule(std::move(e));
    };
    reg.register_op("scale",         [&](const IssueCtx& c){vec("scale",        1,0,c);});
    reg.register_op("rowmax",        [&](const IssueCtx& c){vec("rowmax",       1,0,c);});
    reg.register_op("update_rowmax", [&](const IssueCtx& c){vec("update_rowmax",1,1,c);});
    reg.register_op("exp_shift",     [&](const IssueCtx& c){vec("exp_shift",    1,1,c);});
    reg.register_op("update_rowsum", [&](const IssueCtx& c){vec("update_rowsum",1,0,c);});
    reg.register_op("accumulate",    [&](const IssueCtx& c){vec("accumulate",   1,0,c);});
    reg.register_op("normalize",     [&](const IssueCtx& c){vec("normalize",    1,0,c);});
    reg.register_op("logsumexp",     [&](const IssueCtx& c){vec("logsumexp",    1,1,c);});
    // dma_stage: actually copies src→dst so downstream GEMMs see real data
    reg.register_op("dma_stage", [&arch, &ts](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) return;
        const auto& p = ctx.inst.params;
        DmaTransfer xfer;
        xfer.on_chip = true;
        xfer.src_buf = pget_str(p, "source");
        xfer.dst_buf = pget_str(p, "destination");
        xfer.bytes   = (uint64_t)arch.systolic.rows * arch.systolic.d_head * 2;
        Cycle lat    = static_cast<Cycle>(arch.systolic.rows);
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e; e.type=EventType::OP_START; e.target=res.id; e.cycle=res.start;
        e.instr=ctx.inst.id; e.label=ctx.inst.label; e.payload=xfer;
        ctx.engine.schedule(std::move(e));
    });
    reg.register_op("dma_load",  delay_op);
    reg.register_op("dma_store", delay_op);
    reg.register_op("sram_read",  delay_op);
    reg.register_op("sram_write", delay_op);

    Scheduler scheduler(engine, reg, schedule);
    // Wire all units
    for (UnitId uid=0; uid<(UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x=dynamic_cast<DelayUnit*>   (u)){x->set_scheduler(&scheduler);continue;}
        if (auto* x=dynamic_cast<SystolicUnit*>(u)){x->set_scheduler(&scheduler);continue;}
        if (auto* x=dynamic_cast<DmaUnit*>     (u)){x->set_scheduler(&scheduler);x->set_tensor_store(&ts);continue;}
        if (auto* x=dynamic_cast<VectorUnit*>  (u)){x->set_scheduler(&scheduler);x->set_tensor_store(&ts);continue;}
        if (auto* x=dynamic_cast<AccessUnit*>  (u)){x->set_scheduler(&scheduler);x->set_tensor_store(&ts);continue;}
        if (auto* x=dynamic_cast<BufferUnit*>  (u)){x->set_scheduler(&scheduler);x->set_tensor_store(&ts);continue;}
    }

    scheduler.launch();
    engine.run();
    return scheduler.all_done();
}

// =============================================================================
// TEST: CPU reference correctness
// =============================================================================
TEST_CASE("FA2 correctness: O_tile and L_tile match CPU reference attention") {
    ArchConfig arch = ArchConfig::from_yaml_file("configs/default.yaml");
    const uint32_t Br = arch.systolic.rows;   // 128
    const uint32_t Bc = arch.systolic.cols;   // 128
    const uint32_t DH = arch.systolic.d_head; // 128

    TensorStore ts;
    // Use fixed random seeds — same as sim_main --schedule mode
    ts.init_random("shared_ibuf.Q_tile",   (size_t)Br*DH, -1.f, 1.f, 1);
    ts.init_random("shared_ibuf.K_tile",   (size_t)Bc*DH, -1.f, 1.f, 2);
    ts.init_random("shared_ibuf.K_tile_T", (size_t)DH*Bc, -1.f, 1.f, 3);
    ts.init_random("shared_ibuf.V_tile",   (size_t)Bc*DH, -1.f, 1.f, 4);
    ts.init_zeros ("shared_ibuf.P_tile",   (size_t)Br*Bc);
    ts.init_zeros  ("shared_obuf.O_acc",      (size_t)Br*DH);
    ts.init_neg_inf("shared_obuf.m",          (size_t)Br);
    ts.init_zeros  ("shared_obuf.l",          (size_t)Br);
    ts.init_zeros  ("shared_obuf.correction", (size_t)Br);
    ts.init_random ("systolic_array.Q_operand", (size_t)Br*DH, -1.f, 1.f, 1);
    ts.init_zeros  ("systolic_array.P_operand", (size_t)Br*Bc);
    ts.init_zeros  ("vector_scratch.rowmax_tmp", (size_t)Br);

    bool done = run_fa2_schedule("schedules/fa2_single_tile.yaml", ts, arch);
    REQUIRE(done);

    REQUIRE(ts.has("shared_obuf.O_tile"));
    REQUIRE(ts.has("shared_obuf.L_tile"));

    // CPU reference
    // K_tile_T was computed by the transpose op from K_tile (seed 2).
    // We rebuild it here using the same seed-2 K_tile, then transpose.
    std::vector<float> K_raw((size_t)Bc*DH);
    {   // generate K_tile with seed 2 (same as ts.init_random above)
        uint32_t s = 2;
        for (auto& v : K_raw) {
            s = s*1664525u + 1013904223u;
            v = -1.f + 2.f * static_cast<float>(s>>8)/static_cast<float>(1<<24);
        }
    }
    // Transpose K_raw [Bc×DH] → KT [DH×Bc]
    std::vector<float> KT((size_t)DH*Bc);
    for (uint32_t r=0;r<Bc;r++)
        for (uint32_t c=0;c<DH;c++)
            KT[c*Bc+r] = K_raw[r*DH+c];

    std::vector<float> ref_O, ref_L;
    cpu_attention(ts.get("shared_ibuf.Q_tile"), KT,
                  ts.get("shared_ibuf.V_tile"),
                  Br, Bc, DH, DH, ref_O, ref_L);

    const float tol = 1e-3f;
    float O_err = max_abs_diff(ts.get("shared_obuf.O_tile"), ref_O);
    float L_err = max_abs_diff(ts.get("shared_obuf.L_tile"), ref_L);

    CAPTURE(O_err);
    CAPTURE(L_err);
    CHECK(O_err < tol);
    CHECK(L_err < tol);
}
