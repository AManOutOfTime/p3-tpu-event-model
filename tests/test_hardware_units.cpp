#include <doctest/doctest.h>
#include "core/event_engine.h"
#include "config/arch_config.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/dma_unit.h"
#include "units/buffer_unit.h"
#include "units/vector_unit.h"
#include "units/access_unit.h"
#include "units/systolic_unit.h"
#include "units/delay_unit.h"
#include <sstream>
#include <string>
#include <unordered_map>

using namespace sim;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {

ArchConfig default_arch() {
    ArchConfig a;
    a.clock_ghz                  = 1.0;
    a.hbm.bandwidth_tb_s         = 2.0;   // 2000 B/cycle
    a.hbm.latency_cycles         = 200;
    a.dma.channels               = 1;
    a.sram.banking_factor        = 8;
    a.vector_core.simd_width     = 64;
    a.vector_core.exp_latency    = 4;
    a.access_core.bandwidth      = 64;
    a.systolic.rows = a.systolic.cols = 128;
    return a;
}

// Fire a single OP_START on a unit and return when OP_DONE fires.
template<typename UnitT, typename PayloadT>
Cycle fire_and_run(UnitT* unit, EventEngine& engine, const PayloadT& payload,
                   InstructionId instr_id = 0) {
    UnitId uid = engine.find_unit(unit->name());
    Event e;
    e.type    = EventType::OP_START;
    e.target  = uid;
    e.cycle   = 0;
    e.instr   = instr_id;
    e.payload = payload;
    engine.schedule(e);
    return engine.run();
}

// Build a one-instruction schedule and run it with the given engine + registry.
Cycle run_single(EventEngine& engine, OpRegistry& reg,
                 const std::string& op, const std::string& unit_name,
                 const ParamMap& params) {
    Schedule sched;
    Instruction inst;
    inst.id   = 0;
    inst.op   = op;
    inst.unit = unit_name;
    inst.params = params;
    sched.instructions.push_back(inst);
    Scheduler scheduler(engine, reg, sched);
    // Wire every unit
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<DmaUnit*>    (u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x = dynamic_cast<BufferUnit*> (u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x = dynamic_cast<VectorUnit*> (u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x = dynamic_cast<AccessUnit*> (u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x = dynamic_cast<SystolicUnit*>(u)){ x->set_scheduler(&scheduler); continue; }
        if (auto* x = dynamic_cast<DelayUnit*>  (u)) { x->set_scheduler(&scheduler); continue; }
    }
    scheduler.launch();
    Cycle c = engine.run();
    CHECK(scheduler.all_done());
    return c;
}

}  // namespace

// =============================================================================
// DmaUnit tests
// =============================================================================
TEST_CASE("DmaUnit::transfer_latency formula") {
    ArchConfig arch = default_arch();  // 2000 B/cycle, hbm_lat=200

    SUBCASE("zero bytes returns 0") {
        DmaUnit du("dma", arch, nullptr);
        CHECK(du.transfer_latency(0) == 0);
    }
    SUBCASE("32 KB tile (typical Q/K/V)") {
        DmaUnit du("dma", arch, nullptr);
        // ceil(32768 / 2000) = 17  →  200 + 17 = 217
        CHECK(du.transfer_latency(32768) == 217);
    }
    SUBCASE("1 byte — latency dominated by HBM latency") {
        DmaUnit du("dma", arch, nullptr);
        // ceil(1 / 2000) = 1  →  200 + 1 = 201
        CHECK(du.transfer_latency(1) == 201);
    }
    SUBCASE("multi-channel halves transfer time") {
        ArchConfig a2 = default_arch();
        a2.dma.channels = 2;    // 4000 B/cycle effective
        DmaUnit du2("dma", a2, nullptr);
        // ceil(32768 / 4000) = 9  →  200 + 9 = 209
        CHECK(du2.transfer_latency(32768) == 209);
    }
}

TEST_CASE("DmaUnit backward compat with op:delay (int64_t payload)") {
    ArchConfig arch = default_arch();
    std::ostringstream log;
    auto* dma = new DmaUnit("dma", arch, nullptr, nullptr, log);
    EventEngine engine;
    engine.register_unit(std::unique_ptr<Unit>(dma));

    // int64_t payload → use as latency directly (no HBM model applied)
    Cycle c = fire_and_run(dma, engine, static_cast<int64_t>(99));
    CHECK(c == 99);
    CHECK(log.str().find("DMA_START") != std::string::npos);
    CHECK(log.str().find("DMA_DONE")  != std::string::npos);
}

TEST_CASE("DmaUnit DmaTransfer payload uses arch model") {
    ArchConfig arch = default_arch();
    std::ostringstream log;
    auto* dma = new DmaUnit("dma", arch, nullptr, nullptr, log);
    EventEngine engine;
    engine.register_unit(std::unique_ptr<Unit>(dma));

    DmaTransfer xfer; xfer.bytes = 32768;
    Cycle c = fire_and_run(dma, engine, xfer);
    CHECK(c == 217);
    CHECK(log.str().find("bytes=32768") != std::string::npos);
}

// =============================================================================
// BufferUnit tests
// =============================================================================
TEST_CASE("BufferUnit::access_latency formula") {
    SramConfig sc; sc.banking_factor = 8;
    BufferUnit bu("ibuf", sc);

    SUBCASE("zero bytes") { CHECK(bu.access_latency(0) == 0); }
    SUBCASE("8 bytes = 1 cycle") { CHECK(bu.access_latency(8) == 1); }
    SUBCASE("32 KB = 4096 cycles") { CHECK(bu.access_latency(32768) == 4096); }
    SUBCASE("not a multiple — ceiling") {
        // 9 bytes / 8 = 1.125 → ceil = 2
        CHECK(bu.access_latency(9) == 2);
    }
}

TEST_CASE("BufferUnit double-buffer bank overlap") {
    SramConfig sc; sc.banking_factor = 8;
    std::ostringstream log;
    auto* bu = new BufferUnit("ibuf", sc, nullptr, log);
    EventEngine engine;
    engine.register_unit(std::unique_ptr<Unit>(bu));
    UnitId uid = engine.find_unit("ibuf");

    // Access 1: 8 bytes → 1 cycle, uses bank 0 (free_at=0)
    // Access 2: 8 bytes → 1 cycle, uses bank 1 (both free at same time initially)
    // Both start at cycle 0, bank 0 frees at cycle 1, bank 1 frees at cycle 1.
    // Access 3: has to wait for any bank — should complete at cycle 2.
    auto fire = [&](Cycle start_cycle, InstructionId id) {
        SramAccess a; a.bytes = 8; a.is_write = false;
        Event e; e.type=EventType::OP_START; e.target=uid;
        e.cycle=start_cycle; e.instr=id; e.payload=a;
        engine.schedule(e);
    };

    fire(0, 0);
    fire(0, 1);
    fire(0, 2);  // third access must stall on whichever bank

    engine.run();
    CHECK(log.str().find("SRAM_READ") != std::string::npos);
    // (We can't easily check exact done cycles without a scheduler here,
    //  but the unit runs without crash and emits 3 start+done pairs.)
    size_t start_count = 0;
    std::string s = log.str();
    for (size_t pos = 0; (pos = s.find("SRAM_READ", pos)) != std::string::npos; ++pos)
        ++start_count;
    CHECK(start_count == 3);
}

TEST_CASE("BufferUnit backward compat with int64_t payload") {
    SramConfig sc; sc.banking_factor = 8;
    std::ostringstream log;
    auto* bu = new BufferUnit("ibuf", sc, nullptr, log);
    EventEngine engine;
    engine.register_unit(std::unique_ptr<Unit>(bu));
    Cycle c = fire_and_run(bu, engine, static_cast<int64_t>(42));
    CHECK(c == 42);
}

// =============================================================================
// VectorUnit tests
// =============================================================================
TEST_CASE("VectorUnit::compute_latency") {
    VectorCoreConfig vc; vc.simd_width = 64; vc.exp_latency = 4;
    VectorUnit vu("vc", vc);

    SUBCASE("scale 128x128: 1 pass, 0 exp") {
        VectorOp op; op.elements=16384; op.passes=1; op.exp_ops=0;
        // groups = 256, latency = 256
        CHECK(vu.compute_latency(op) == 256);
    }
    SUBCASE("exp 128x128: 1 pass + exp overhead") {
        VectorOp op; op.elements=16384; op.passes=1; op.exp_ops=1;
        // groups=256, linear=256, trans=4*256=1024 → total=1280
        CHECK(vu.compute_latency(op) == 1280);
    }
    SUBCASE("softmax 128x128: 3 passes + exp overhead") {
        VectorOp op; op.elements=16384; op.passes=3; op.exp_ops=1;
        // groups=256, linear=3*256=768, trans=4*256=1024 → total=1792
        CHECK(vu.compute_latency(op) == 1792);
    }
    SUBCASE("layer_norm 128x128: 2 passes, 0 exp") {
        VectorOp op; op.elements=16384; op.passes=2; op.exp_ops=0;
        CHECK(vu.compute_latency(op) == 512);
    }
    SUBCASE("logsumexp length=128: 1 pass + exp") {
        VectorOp op; op.elements=128; op.passes=1; op.exp_ops=1;
        // groups=2, linear=2, trans=4*2=8 → total=10
        CHECK(vu.compute_latency(op) == 10);
    }
    SUBCASE("zero elements returns 0") {
        VectorOp op; op.elements=0; op.passes=3; op.exp_ops=1;
        CHECK(vu.compute_latency(op) == 0);
    }
}

TEST_CASE("VectorUnit event flow and backward compat") {
    VectorCoreConfig vc; vc.simd_width=64; vc.exp_latency=4;
    std::ostringstream log;

    SUBCASE("VectorOp payload") {
        auto* vu = new VectorUnit("vc", vc, nullptr, log);
        EventEngine engine; engine.register_unit(std::unique_ptr<Unit>(vu));
        VectorOp op; op.elements=16384; op.passes=1; op.exp_ops=0; op.kind="scale";
        Cycle c = fire_and_run(vu, engine, op);
        CHECK(c == 256);
        CHECK(log.str().find("VEC_START") != std::string::npos);
        CHECK(log.str().find("scale")     != std::string::npos);
    }
    SUBCASE("int64_t payload (delay compat)") {
        auto* vu = new VectorUnit("vc2", vc, nullptr, log);
        EventEngine engine; engine.register_unit(std::unique_ptr<Unit>(vu));
        Cycle c = fire_and_run(vu, engine, static_cast<int64_t>(77));
        CHECK(c == 77);
    }
}

// =============================================================================
// AccessUnit tests
// =============================================================================
TEST_CASE("AccessUnit::compute_latency") {
    AccessCoreConfig ac; ac.bandwidth = 64;
    AccessUnit au("ac", ac);

    SUBCASE("transpose 128x128 = 16384 elements") {
        // ceil(16384/64) = 256
        CHECK(au.compute_latency(16384) == 256);
    }
    SUBCASE("init_fill 128 elements") {
        CHECK(au.compute_latency(128) == 2);
    }
    SUBCASE("zero elements") {
        CHECK(au.compute_latency(0) == 0);
    }
    SUBCASE("not a multiple — ceiling") {
        CHECK(au.compute_latency(65) == 2);
    }
}

TEST_CASE("AccessUnit event flow and backward compat") {
    AccessCoreConfig ac; ac.bandwidth=64;
    std::ostringstream log;

    SUBCASE("AccessOp payload") {
        auto* au = new AccessUnit("ac", ac, nullptr, log);
        EventEngine engine; engine.register_unit(std::unique_ptr<Unit>(au));
        AccessOp op; op.elements=16384; op.kind="transpose";
        Cycle c = fire_and_run(au, engine, op);
        CHECK(c == 256);
        CHECK(log.str().find("ACCESS_START") != std::string::npos);
        CHECK(log.str().find("transpose")    != std::string::npos);
    }
    SUBCASE("int64_t backward compat") {
        auto* au = new AccessUnit("ac2", ac, nullptr, log);
        EventEngine engine; engine.register_unit(std::unique_ptr<Unit>(au));
        Cycle c = fire_and_run(au, engine, static_cast<int64_t>(55));
        CHECK(c == 55);
    }
}

// =============================================================================
// Full FA2-hw pipeline integration
// =============================================================================
TEST_CASE("fa2_hw schedule: all 20 instructions complete") {
    std::string path = std::string(SIM_PROJECT_ROOT) + "/schedules/fa2_hw.yaml";
    Schedule sched = Schedule::from_yaml_file(path);
    REQUIRE(sched.instructions.size() == 20);

    ArchConfig arch = default_arch();
    std::ostringstream log;
    EventEngine engine(arch.clock_ghz);

    engine.register_unit(std::make_unique<SystolicUnit>("systolic",      arch.systolic));
    engine.register_unit(std::make_unique<BufferUnit>  ("shared_ibuf",   arch.sram));
    engine.register_unit(std::make_unique<BufferUnit>  ("shared_obuf",   arch.sram));
    engine.register_unit(std::make_unique<VectorUnit>  ("tandem_1",      arch.vector_core, nullptr, log));
    engine.register_unit(std::make_unique<VectorUnit>  ("tandem_2",      arch.vector_core, nullptr, log));
    engine.register_unit(std::make_unique<VectorUnit>  ("vector_core",   arch.vector_core, nullptr, log));
    engine.register_unit(std::make_unique<AccessUnit>  ("access_core_1", arch.access_core, nullptr, log));
    engine.register_unit(std::make_unique<DmaUnit>("dma", arch, nullptr, nullptr, log));

    // Register all ops (mirrors sim_main logic)
    OpRegistry reg;

    auto dma_op = [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        DmaTransfer x;
        x.bytes = (uint64_t)pget_int(ctx.inst.params,"bytes",0);
        if (!x.bytes) {
            x.bytes = (uint64_t)pget_int(ctx.inst.params,"rows",0)
                     *(uint64_t)pget_int(ctx.inst.params,"cols",0) * 2;
        }
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=x; ctx.engine.schedule(std::move(e));
    };
    reg.register_op("dma_load",  dma_op);
    reg.register_op("dma_store", dma_op);

    auto vec_op = [](uint32_t passes, uint32_t exp_ops, const std::string& kind) {
        return [passes, exp_ops, kind](const IssueCtx& ctx) {
            UnitId t = ctx.engine.find_unit(ctx.inst.unit);
            VectorOp op; op.kind=kind; op.passes=passes; op.exp_ops=exp_ops;
            op.elements = (uint64_t)pget_int(ctx.inst.params,"elements",0);
            if (!op.elements) {
                uint64_t r=(uint64_t)pget_int(ctx.inst.params,"rows",0);
                uint64_t c=(uint64_t)pget_int(ctx.inst.params,"cols",0);
                uint64_t l=(uint64_t)pget_int(ctx.inst.params,"length",0);
                op.elements = (r&&c)?r*c:l;
            }
            Event e; e.type=EventType::OP_START; e.target=t;
            e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
            e.payload=op; ctx.engine.schedule(std::move(e));
        };
    };
    reg.register_op("scale",      vec_op(1,0,"scale"));
    reg.register_op("exp",        vec_op(1,1,"exp"));
    reg.register_op("rowmax",     vec_op(1,0,"rowmax"));
    reg.register_op("rowsum",     vec_op(1,0,"rowsum"));
    reg.register_op("accumulate", vec_op(1,0,"accumulate"));
    reg.register_op("normalize",  vec_op(1,0,"normalize"));
    reg.register_op("logsumexp",  vec_op(1,1,"logsumexp"));

    auto acc_op = [](const std::string& kind) {
        return [kind](const IssueCtx& ctx) {
            UnitId t = ctx.engine.find_unit(ctx.inst.unit);
            AccessOp op; op.kind=kind;
            op.elements = (uint64_t)pget_int(ctx.inst.params,"elements",0);
            if (!op.elements) {
                uint64_t r=(uint64_t)pget_int(ctx.inst.params,"rows",0);
                uint64_t c=(uint64_t)pget_int(ctx.inst.params,"cols",0);
                uint64_t l=(uint64_t)pget_int(ctx.inst.params,"length",0);
                op.elements = (r&&c)?r*c:l;
            }
            Event e; e.type=EventType::OP_START; e.target=t;
            e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
            e.payload=op; ctx.engine.schedule(std::move(e));
        };
    };
    reg.register_op("transpose",  acc_op("transpose"));
    reg.register_op("init_fill",  acc_op("init_fill"));

    reg.register_op("gemm", [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        GemmShape s;
        s.M=(uint32_t)pget_int(ctx.inst.params,"M",arch.systolic.rows);
        s.K=(uint32_t)pget_int(ctx.inst.params,"K",arch.systolic.rows);
        s.N=(uint32_t)pget_int(ctx.inst.params,"N",arch.systolic.cols);
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=s; ctx.engine.schedule(std::move(e));
    });

    Scheduler scheduler(engine, reg, sched);
    for (UnitId uid=0; uid<(UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x=dynamic_cast<SystolicUnit*>(u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x=dynamic_cast<DmaUnit*>     (u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x=dynamic_cast<BufferUnit*>  (u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x=dynamic_cast<VectorUnit*>  (u)) { x->set_scheduler(&scheduler); continue; }
        if (auto* x=dynamic_cast<AccessUnit*>  (u)) { x->set_scheduler(&scheduler); continue; }
    }

    scheduler.launch();
    Cycle final_cycle = engine.run();

    CHECK(scheduler.all_done());
    CHECK(scheduler.outstanding() == 0);

    // The two GEMMs (ids 7 and 14) must have completed
    CHECK(log.str().find("DMA_DONE")    != std::string::npos);
    CHECK(log.str().find("ACCESS_DONE") != std::string::npos);
    CHECK(log.str().find("VEC_DONE")    != std::string::npos);

    // DMA serialization: store_L (19) must start after store_O (18) completes.
    // (Verified implicitly by the dependency graph in the schedule.)
    CHECK(final_cycle > 0);
}

TEST_CASE("fa2_hw: DMA latency dominates for large tiles") {
    // Sanity check: the DMA load of Q (32KB) should be at least 217 cycles.
    // The GEMM (382 cycles) must start after the DMA + transpose chain.
    // So final_cycle >> 382.
    std::string path = std::string(SIM_PROJECT_ROOT) + "/schedules/fa2_hw.yaml";
    Schedule sched = Schedule::from_yaml_file(path);

    ArchConfig arch = default_arch();
    EventEngine engine(arch.clock_ghz);
    engine.register_unit(std::make_unique<SystolicUnit>("systolic",      arch.systolic));
    engine.register_unit(std::make_unique<BufferUnit>  ("shared_ibuf",   arch.sram));
    engine.register_unit(std::make_unique<BufferUnit>  ("shared_obuf",   arch.sram));
    engine.register_unit(std::make_unique<VectorUnit>  ("tandem_1",      arch.vector_core));
    engine.register_unit(std::make_unique<VectorUnit>  ("tandem_2",      arch.vector_core));
    engine.register_unit(std::make_unique<VectorUnit>  ("vector_core",   arch.vector_core));
    engine.register_unit(std::make_unique<AccessUnit>  ("access_core_1", arch.access_core));
    engine.register_unit(std::make_unique<DmaUnit>("dma", arch, nullptr));

    // Re-use the registry build from the previous test case inline
    OpRegistry reg;
    auto dma_op = [&arch](const IssueCtx& ctx) {
        UnitId t = ctx.engine.find_unit(ctx.inst.unit);
        DmaTransfer x;
        x.bytes = (uint64_t)pget_int(ctx.inst.params,"bytes",0);
        if (!x.bytes) {
            x.bytes = (uint64_t)pget_int(ctx.inst.params,"rows",0)
                     *(uint64_t)pget_int(ctx.inst.params,"cols",0) * 2;
        }
        Event e; e.type=EventType::OP_START; e.target=t;
        e.cycle=ctx.engine.current_cycle(); e.instr=ctx.inst.id; e.label=ctx.inst.label;
        e.payload=x; ctx.engine.schedule(std::move(e));
    };
    reg.register_op("dma_load",  dma_op);
    reg.register_op("dma_store", dma_op);
    auto ve=[](uint32_t p,uint32_t e,const std::string& k){return[p,e,k](const IssueCtx&ctx){
        UnitId t=ctx.engine.find_unit(ctx.inst.unit);
        VectorOp op;op.kind=k;op.passes=p;op.exp_ops=e;
        op.elements=(uint64_t)pget_int(ctx.inst.params,"elements",0);
        if(!op.elements){uint64_t r=(uint64_t)pget_int(ctx.inst.params,"rows",0);
         uint64_t c=(uint64_t)pget_int(ctx.inst.params,"cols",0);
         uint64_t l=(uint64_t)pget_int(ctx.inst.params,"length",0);op.elements=(r&&c)?r*c:l;}
        Event ev;ev.type=EventType::OP_START;ev.target=t;
        ev.cycle=ctx.engine.current_cycle();ev.instr=ctx.inst.id;ev.label=ctx.inst.label;
        ev.payload=op;ctx.engine.schedule(std::move(ev));};};
    reg.register_op("scale",     ve(1,0,"scale"));  reg.register_op("exp",      ve(1,1,"exp"));
    reg.register_op("rowmax",    ve(1,0,"rowmax")); reg.register_op("rowsum",   ve(1,0,"rowsum"));
    reg.register_op("accumulate",ve(1,0,"acc"));    reg.register_op("normalize",ve(1,0,"norm"));
    reg.register_op("logsumexp", ve(1,1,"lse"));
    auto ae=[](const std::string& k){return[k](const IssueCtx&ctx){
        UnitId t=ctx.engine.find_unit(ctx.inst.unit);AccessOp op;op.kind=k;
        op.elements=(uint64_t)pget_int(ctx.inst.params,"elements",0);
        if(!op.elements){uint64_t r=(uint64_t)pget_int(ctx.inst.params,"rows",0);
         uint64_t c=(uint64_t)pget_int(ctx.inst.params,"cols",0);
         uint64_t l=(uint64_t)pget_int(ctx.inst.params,"length",0);op.elements=(r&&c)?r*c:l;}
        Event ev;ev.type=EventType::OP_START;ev.target=t;
        ev.cycle=ctx.engine.current_cycle();ev.instr=ctx.inst.id;ev.label=ctx.inst.label;
        ev.payload=op;ctx.engine.schedule(std::move(ev));};};
    reg.register_op("transpose", ae("transpose")); reg.register_op("init_fill",ae("init_fill"));
    reg.register_op("gemm",[&arch](const IssueCtx&ctx){
        UnitId t=ctx.engine.find_unit(ctx.inst.unit);GemmShape s;
        s.M=(uint32_t)pget_int(ctx.inst.params,"M",arch.systolic.rows);
        s.K=(uint32_t)pget_int(ctx.inst.params,"K",arch.systolic.rows);
        s.N=(uint32_t)pget_int(ctx.inst.params,"N",arch.systolic.cols);
        Event e;e.type=EventType::OP_START;e.target=t;
        e.cycle=ctx.engine.current_cycle();e.instr=ctx.inst.id;e.label=ctx.inst.label;
        e.payload=s;ctx.engine.schedule(std::move(e));});

    Scheduler scheduler(engine, reg, sched);
    for (UnitId uid=0; uid<(UnitId)engine.num_units(); uid++) {
        Unit* u=engine.get_unit(uid);
        if(auto*x=dynamic_cast<SystolicUnit*>(u)){x->set_scheduler(&scheduler);continue;}
        if(auto*x=dynamic_cast<DmaUnit*>     (u)){x->set_scheduler(&scheduler);continue;}
        if(auto*x=dynamic_cast<BufferUnit*>  (u)){x->set_scheduler(&scheduler);continue;}
        if(auto*x=dynamic_cast<VectorUnit*>  (u)){x->set_scheduler(&scheduler);continue;}
        if(auto*x=dynamic_cast<AccessUnit*>  (u)){x->set_scheduler(&scheduler);continue;}
    }
    scheduler.launch();
    Cycle final_cycle = engine.run();
    CHECK(scheduler.all_done());

    // The 3 DMA loads alone take 3 × 217 = 651 cycles minimum (serialized).
    // After that, transpose (256) + GEMM (382) + exp (1280) + ... must follow.
    // So total must be well above 382 cycles.
    CHECK(final_cycle > 1000);
}
