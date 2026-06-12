#include "schedule/op_handlers.h"

#include "core/event.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include "units/access_unit.h"
#include "units/dma_unit.h"
#include "units/systolic_unit.h"
#include "units/vector_unit.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sim {

uint32_t resolve_dim(const ParamMap& p, const std::string& key,
                     const ArchConfig& arch, uint32_t def) {
    int64_t ival = pget_int(p, key, -1);
    if (ival >= 0) return static_cast<uint32_t>(ival);
    std::string s = pget_str(p, key);
    if (s == "Br" || s == "Bc")      return arch.systolic.rows;
    if (s == "d_k" || s == "d_head") return arch.systolic.d_head;
    if (s == "hidden_dim")           return arch.systolic.rows * arch.systolic.d_head;
    return def;
}

uint32_t dtype_bytes(const std::string& p) {
    if (p == "FP8")  return 1;
    if (p == "FP32") return 4;
    return 2;
}

namespace {

Cycle vector_latency(const VectorOp& op, const ArchConfig& arch) {
    if (!op.elements || !arch.vector_core.simd_width) return 0;
    double g = std::ceil(static_cast<double>(op.elements) / arch.vector_core.simd_width);
    return static_cast<Cycle>(op.passes) * static_cast<Cycle>(g)
         + static_cast<Cycle>(op.exp_ops) * arch.vector_core.exp_latency * static_cast<Cycle>(g);
}

void issue_vector(const IssueCtx& ctx, const VectorOp& op, Cycle lat) {
    auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
    if (targets.empty()) throw std::runtime_error(ctx.inst.op + ": unknown unit");
    auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
    Event e;
    e.type = EventType::OP_START;
    e.target = res.id;
    e.cycle = res.start;
    e.instr = ctx.inst.id;
    e.label = ctx.inst.label;
    e.payload = op;
    ctx.engine.schedule(std::move(e));
}

}  // namespace

void register_builtin_ops(OpRegistry& reg, const ArchConfig& arch) {
    reg.register_op("delay", [](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty())
            throw std::runtime_error("delay: unknown unit '" + ctx.inst.unit + "'");
        Cycle lat = static_cast<Cycle>(pget_int(ctx.inst.params, "latency_cycles", 10));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = static_cast<int64_t>(lat);
        ctx.engine.schedule(std::move(e));
    });

    auto hbm_op = [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error(ctx.inst.op + ": unknown unit");
        const auto& p = ctx.inst.params;
        DmaTransfer xfer;
        xfer.on_chip = false;
        xfer.src_buf = pget_str(p, "source");
        xfer.dst_buf = pget_str(p, "destination");
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        uint32_t len  = resolve_dim(p, "length", arch);
        xfer.bytes = (rows && cols)
            ? static_cast<uint64_t>(rows) * cols * dtype_bytes(arch.systolic.precision)
            : static_cast<uint64_t>(len) * dtype_bytes(arch.systolic.precision);
        ctx.engine.add_hbm_bytes(xfer.bytes);   // P0.2/P1.5: off-chip traffic
        double bw = arch.hbm_bytes_per_cycle() * arch.dma.channels;
        Cycle band = static_cast<Cycle>(std::ceil(static_cast<double>(xfer.bytes) / bw));
        // P1.5: channel occupancy is the bandwidth term only when pipelined; the
        // latency is fill that overlaps neighbouring transfers. When not
        // pipelined, the channel is held for the full latency+bandwidth.
        // (The DmaUnit always completes the data after latency+bandwidth.)
        Cycle occ = arch.hbm.pipelined
            ? band
            : static_cast<Cycle>(arch.hbm.latency_cycles) + band;
        auto res = ctx.scheduler.reserve_unit_pool(targets, occ);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = xfer;
        ctx.engine.schedule(std::move(e));
    };
    reg.register_op("dma_load", hbm_op);
    reg.register_op("dma_store", hbm_op);

    reg.register_op("embedding_lookup", hbm_op);

    reg.register_op("dma_stage", [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("dma_stage: unknown unit");
        const auto& p = ctx.inst.params;
        DmaTransfer xfer;
        xfer.on_chip = true;
        xfer.src_buf = pget_str(p, "source");
        xfer.dst_buf = pget_str(p, "destination");
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        uint32_t len  = resolve_dim(p, "length", arch);
        uint64_t elems = (rows && cols) ? static_cast<uint64_t>(rows) * cols
                                        : static_cast<uint64_t>(len);
        xfer.bytes = elems * dtype_bytes(arch.systolic.precision);
        // IBUF -> systolic array PE registers over the WIDE on-chip operand bus.
        // The array has `rows` input lanes (one per PE row) and ingests one
        // column of `rows` elements per cycle, so latency = ceil(elements/rows).
        // This is the array's intrinsic ingest bandwidth and matches the K-cycle
        // GEMM streaming model. It is deliberately NOT bounded by the narrow SRAM
        // banking_factor (that bounds OBUF/IBUF SRAM r/w, a different, narrower
        // path).
        const uint32_t ingest_lanes = arch.systolic.rows ? arch.systolic.rows : 1;
        Cycle lat = static_cast<Cycle>(std::ceil(
            static_cast<double>(elems) / static_cast<double>(ingest_lanes)));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = xfer;
        ctx.engine.schedule(std::move(e));
    });

    reg.register_op("init_fill", [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("init_fill: unknown unit");
        const auto& p = ctx.inst.params;
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        uint32_t len  = resolve_dim(p, "length", arch);
        AccessOp op;
        op.kind = "init_fill";
        op.elements = (rows && cols) ? static_cast<uint64_t>(rows) * cols
                                     : static_cast<uint64_t>(len);
        op.dst = pget_str(p, "destination");
        std::string iv = pget_str(p, "init_value");
        op.fill_value = (iv == "-inf")
            ? -std::numeric_limits<float>::infinity()
            : static_cast<float>(pget_dbl(p, "init_value", 0.0));
        Cycle lat = static_cast<Cycle>(std::ceil(
            static_cast<double>(op.elements) / arch.access_core.bandwidth));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = op;
        ctx.engine.schedule(std::move(e));
    });

    reg.register_op("transpose", [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("transpose: unknown unit");
        const auto& p = ctx.inst.params;
        uint32_t rows = resolve_dim(p, "input_rows", arch);
        uint32_t cols = resolve_dim(p, "input_cols", arch);
        AccessOp op;
        op.kind = "transpose";
        op.elements = static_cast<uint64_t>(rows) * cols;
        op.src = pget_str(p, "source");
        op.dst = pget_str(p, "destination");
        op.input_rows = rows;
        op.input_cols = cols;
        Cycle lat = static_cast<Cycle>(std::ceil(
            static_cast<double>(op.elements) / arch.access_core.bandwidth));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = op;
        ctx.engine.schedule(std::move(e));
    });

    reg.register_op("sram_copy", [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("sram_copy: unknown unit");
        const auto& p = ctx.inst.params;
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        uint32_t len  = resolve_dim(p, "length", arch);
        AccessOp op;
        op.kind = "copy";
        op.src = pget_str(p, "source");
        op.dst = pget_str(p, "destination");
        op.elements = (rows && cols) ? static_cast<uint64_t>(rows) * cols
                                     : static_cast<uint64_t>(len);
        Cycle lat = static_cast<Cycle>(std::ceil(
            static_cast<double>(op.elements) / arch.access_core.bandwidth));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = op;
        ctx.engine.schedule(std::move(e));
    });

    reg.register_op("kv_stage_release", [](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("kv_stage_release: unknown unit");
        auto res = ctx.scheduler.reserve_unit_pool(targets, 0);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = static_cast<int64_t>(0);
        ctx.engine.schedule(std::move(e));
    });

    reg.register_op("gemm", [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool("systolic");
        if (targets.empty()) throw std::runtime_error("gemm: systolic not found");
        const auto& p = ctx.inst.params;
        GemmShape s;
        s.M = resolve_dim(p, "M", arch, arch.systolic.rows);
        s.K = resolve_dim(p, "K", arch, arch.systolic.d_head);
        s.N = resolve_dim(p, "N", arch, arch.systolic.cols);
        s.src_a = pget_str(p, "source_a");
        s.src_b = pget_str(p, "source_b");
        s.dst_c = pget_str(p, "destination");

        // P0.2: exact MAC count (independent of how the GEMM is tiled).
        ctx.engine.add_macs(static_cast<uint64_t>(s.M) * s.K * s.N);

        // P0.1/P1.4: weight-stationary latency (single source of truth). This
        // fragments (K,N) and streams M analytically, so oversized dims are
        // valid — Tiler pre-tiling is optional and additive.
        Cycle lat = systolic_gemm_latency(arch.systolic, s.M, s.K, s.N);

        // P1.2: tag the GEMM with its SRAM working set (input + weight + output
        // tile) and a spill penalty. The set is actually acquired/released by
        // the SystolicUnit across [OP_START, OP_DONE] — i.e. when the GEMM truly
        // executes — so concurrency (and thus peak SRAM) reflects the hardware,
        // not the scheduler's issue-ahead. Acquiring here would over-count.
        if (arch.model_sram) {
            const uint64_t db = dtype_bytes(arch.systolic.precision);
            s.buffer_bytes =
                (static_cast<uint64_t>(s.M) * s.K +
                 static_cast<uint64_t>(s.K) * s.N +
                 static_cast<uint64_t>(s.M) * s.N) * db;
            s.spill_penalty = arch.hbm.latency_cycles;
        }

        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = s;
        ctx.engine.schedule(std::move(e));
    });

    auto vector_matrix = [arch](const IssueCtx& ctx, const std::string& kind,
                                uint32_t passes, uint32_t exp_ops) {
        const auto& p = ctx.inst.params;
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        uint32_t len  = resolve_dim(p, "length", arch);
        VectorOp op;
        op.kind = kind;
        op.passes = passes;
        op.exp_ops = exp_ops;
        op.rows = rows;
        op.cols = cols;
        op.elements = (rows && cols) ? static_cast<uint64_t>(rows) * cols
                                     : static_cast<uint64_t>(len);
        op.src = pget_str(p, "source");
        op.src_a = pget_str(p, "source_a");
        op.src_b = pget_str(p, "source_b");
        op.dst = pget_str(p, "destination");
        op.src_scale = pget_str(p, "source_scale");
        op.src_matrix = pget_str(p, "source_matrix");
        op.src_shift = pget_str(p, "source_shift");
        op.src_denom = pget_str(p, "source_denom");
        op.row_start = static_cast<uint32_t>(pget_int(p, "row_start", 0));
        op.col_start = static_cast<uint32_t>(pget_int(p, "col_start", 0));
        issue_vector(ctx, op, vector_latency(op, arch));
    };

    reg.register_op("scale", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "scale", 1, 0);
    });
    reg.register_op("rowmax", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "rowmax", 1, 0);
    });
    reg.register_op("row_reduce_sum", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "row_reduce_sum", 1, 0);
    });
    reg.register_op("square", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "square", 1, 0);
    });
    reg.register_op("add_epsilon", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "add_epsilon", 1, 0);
    });
    reg.register_op("rsqrt", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "rsqrt", 1, 1);
    });
    reg.register_op("norm_scale", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "norm_scale", 1, 0);
    });
    reg.register_op("exp_shift", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "exp_shift", 1, 1);
    });
    reg.register_op("accumulate", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "accumulate", 1, 0);
    });
    reg.register_op("normalize", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "normalize", 1, 0);
    });
    reg.register_op("causal_mask", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "causal_mask", 1, 0);
    });
    reg.register_op("rope", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "rope", 2, 0);
    });
    reg.register_op("rope_pair_split", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "rope_pair_split", 1, 0);
    });
    reg.register_op("rope_rotate", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "rope_rotate", 2, 0);
    });
    reg.register_op("rope_store", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "rope_store", 1, 0);
    });
    reg.register_op("rmsnorm", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "rmsnorm", 2, 0);
    });
    reg.register_op("silu", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "silu", 1, 1);
    });
    reg.register_op("elementwise_mul", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "elementwise_mul", 1, 0);
    });
    reg.register_op("silu_mul", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "silu_mul", 2, 1);
    });
    reg.register_op("residual_add", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "residual_add", 1, 0);
    });
    reg.register_op("attention_merge", [arch](const IssueCtx& ctx) {
        const auto& p = ctx.inst.params;
        uint64_t input_elements = static_cast<uint64_t>(pget_int(p, "input_elements", 0));
        uint64_t output_elements = static_cast<uint64_t>(pget_int(p, "output_elements", 0));
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        if (!input_elements) {
            const uint32_t kv_tiles = static_cast<uint32_t>(pget_int(p, "kv_tiles", 1));
            input_elements = static_cast<uint64_t>(rows) * cols * kv_tiles;
        }
        if (!output_elements) output_elements = static_cast<uint64_t>(rows) * cols;

        VectorOp op;
        op.kind = "attention_merge";
        op.passes = 1;
        op.exp_ops = 0;
        op.rows = rows;
        op.cols = cols;
        op.elements = input_elements + output_elements;
        op.src = pget_str(p, "source");
        op.dst = pget_str(p, "destination");
        issue_vector(ctx, op, vector_latency(op, arch));
    });
    reg.register_op("softmax", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "softmax", 3, 1);
    });
    reg.register_op("sample_token", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "sample_token", 1, 0);
    });
    reg.register_op("sample_top1", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "sample_top1", 1, 0);
    });
    reg.register_op("token_feedback", [vector_matrix](const IssueCtx& ctx) {
        vector_matrix(ctx, "token_feedback", 1, 0);
    });
    reg.register_op("select_last_token", [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("select_last_token: unknown unit");
        const auto& p = ctx.inst.params;
        AccessOp op;
        op.kind = "copy";
        op.src = pget_str(p, "source");
        op.dst = pget_str(p, "destination");
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        op.elements = static_cast<uint64_t>(rows) * cols;
        Cycle lat = static_cast<Cycle>(std::ceil(
            static_cast<double>(op.elements) / arch.access_core.bandwidth));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = op;
        ctx.engine.schedule(std::move(e));
    });
    reg.register_op("gather_select", [arch](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        if (targets.empty()) throw std::runtime_error("gather_select: unknown unit");
        const auto& p = ctx.inst.params;
        AccessOp op;
        op.kind = "copy";
        op.src = pget_str(p, "source");
        op.dst = pget_str(p, "destination");
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        op.elements = static_cast<uint64_t>(rows) * cols;
        Cycle lat = static_cast<Cycle>(std::ceil(
            static_cast<double>(op.elements) / arch.access_core.bandwidth));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START;
        e.target = res.id;
        e.cycle = res.start;
        e.instr = ctx.inst.id;
        e.label = ctx.inst.label;
        e.payload = op;
        ctx.engine.schedule(std::move(e));
    });

    reg.register_op("update_rowmax", [arch](const IssueCtx& ctx) {
        const auto& p = ctx.inst.params;
        uint32_t len = resolve_dim(p, "length", arch);
        VectorOp op;
        op.kind = "update_rowmax";
        op.passes = 1;
        op.exp_ops = 1;
        op.elements = len;
        op.src_m = pget_str(p, "source_m_old");
        op.src_rowmax = pget_str(p, "source_rowmax");
        op.dst_m = pget_str(p, "destination_m");
        op.dst_correction = pget_str(p, "destination_correction");
        issue_vector(ctx, op, vector_latency(op, arch));
    });

    reg.register_op("update_rowsum", [arch](const IssueCtx& ctx) {
        const auto& p = ctx.inst.params;
        uint32_t rows = resolve_dim(p, "rows", arch);
        uint32_t cols = resolve_dim(p, "cols", arch);
        VectorOp op;
        op.kind = "update_rowsum";
        op.passes = 1;
        op.exp_ops = 0;
        op.rows = rows;
        op.cols = cols;
        op.elements = static_cast<uint64_t>(rows) * cols;
        op.src_p = pget_str(p, "source_p");
        op.src_correction = pget_str(p, "source_correction");
        op.src_l = pget_str(p, "source_l_old");
        op.dst_l = pget_str(p, "destination");
        issue_vector(ctx, op, vector_latency(op, arch));
    });

    reg.register_op("logsumexp", [arch](const IssueCtx& ctx) {
        const auto& p = ctx.inst.params;
        uint32_t len = resolve_dim(p, "length", arch);
        VectorOp op;
        op.kind = "logsumexp";
        op.passes = 1;
        op.exp_ops = 1;
        op.elements = len;
        op.src_m = pget_str(p, "source_m");
        op.src_l = pget_str(p, "source_l");
        op.dst = pget_str(p, "destination");
        issue_vector(ctx, op, vector_latency(op, arch));
    });
}

}  // namespace sim
