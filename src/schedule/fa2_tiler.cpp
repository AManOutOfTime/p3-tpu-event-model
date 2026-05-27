#include "schedule/fa2_tiler.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

namespace sim {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

// Quick helper: make an Instruction with concrete integer params.
static Instruction make_instr(InstructionId id,
                               const std::string& op,
                               const std::string& unit,
                               const std::string& label,
                               ParamMap params,
                               std::vector<InstructionId> deps = {})
{
    Instruction ins;
    ins.id         = id;
    ins.op         = op;
    ins.unit       = unit;
    ins.label      = label;
    ins.params     = std::move(params);
    ins.depends_on = std::move(deps);
    return ins;
}

// Param builder helpers
static ParamMap P(std::initializer_list<std::pair<std::string,ParamVal>> items) {
    ParamMap m;
    for (auto& [k,v] : items) m[k] = v;
    return m;
}

// Generate all instructions for one attention head.
// Returns the vector; caller owns it.
std::vector<Instruction> build(const WorkloadAttention& wl, InstructionId id0)
{
    std::vector<Instruction> out;
    InstructionId nid = id0;

    const int Nq     = (int)wl.Nq();
    const int Nkv    = (int)wl.Nkv();
    const int Br     = (int)wl.Br;
    const int Bc     = (int)wl.Bc;
    const int d_head = (int)wl.d_head;

    auto add = [&](const std::string& op,
                   const std::string& unit,
                   const std::string& label,
                   ParamMap params,
                   std::vector<InstructionId> deps = {}) -> InstructionId
    {
        InstructionId id = nid++;
        out.push_back(make_instr(id, op, unit, label,
                                 std::move(params), std::move(deps)));
        return id;
    };

    bool          first_q_tile = true;
    InstructionId prev_store_L = id0;  // initial value unused; guarded by first_q_tile

    for (int qi = 0; qi < Nq; ++qi) {
        int q_row0 = qi * Br;
        std::vector<InstructionId> q_dep;
        if (!first_q_tile) q_dep = {prev_store_L};
        first_q_tile = false;

        // ── Prologue ──────────────────────────────────────────────────────
        auto init_O = add("init_fill", "access_core",
            "[Q" + std::to_string(qi) + "] Init O_acc = 0",
            P({{"destination", std::string("shared_obuf.O_acc")},
               {"rows",  (int64_t)Br},
               {"cols",  (int64_t)d_head},
               {"init_value", (int64_t)0}}),
            q_dep);

        auto init_m = add("init_fill", "access_core",
            "[Q" + std::to_string(qi) + "] Init m = -inf",
            P({{"destination", std::string("shared_obuf.m")},
               {"length",     (int64_t)Br},
               {"init_value", std::string("-inf")}}),
            q_dep);

        auto init_l = add("init_fill", "access_core",
            "[Q" + std::to_string(qi) + "] Init l = 0",
            P({{"destination", std::string("shared_obuf.l")},
               {"length",     (int64_t)Br},
               {"init_value", (int64_t)0}}),
            q_dep);

        auto load_Q = add("dma_load", "dma",
            "[Q" + std::to_string(qi) + "] Load Q[" +
                std::to_string(q_row0) + ":" +
                std::to_string(q_row0 + Br) + ",0:" +
                std::to_string(d_head) + "]",
            P({{"source",      "HBM.Q[" + std::to_string(q_row0) + ":" +
                               std::to_string(q_row0+Br) + ",0:" +
                               std::to_string(d_head) + "]"},
               {"destination", std::string("shared_ibuf.Q_tile")},
               {"rows", (int64_t)Br},
               {"cols", (int64_t)d_head}}),
            q_dep);

        auto stage_Q = add("dma_stage", "dma",
            "[Q" + std::to_string(qi) + "] Stage Q_tile → Q_operand",
            P({{"source",      std::string("shared_ibuf.Q_tile")},
               {"destination", std::string("systolic_array.Q_operand")},
               {"rows", (int64_t)Br},
               {"cols", (int64_t)d_head}}),
            {load_Q});

        // Running state
        InstructionId prev_update_rowmax = init_m;
        InstructionId prev_update_rowsum = init_l;
        InstructionId prev_accumulate    = init_O;
        InstructionId prev_dma           = stage_Q;

        // ── KV-tile loop ──────────────────────────────────────────────────
        for (int kv = 0; kv < Nkv; ++kv) {
            int buf  = kv % 2;
            int row0 = kv * Bc;
            auto b   = std::to_string(buf);
            auto qi_s = std::to_string(qi);
            auto kv_s = std::to_string(kv);
            std::string tag = "[Q" + qi_s + "/KV" + kv_s + "] ";

            // ─ DMA: Load K[kv] and V[kv] ─────────────────────────────────
            auto load_K = add("dma_load", "dma",
                tag + "Load K[" + std::to_string(row0) + ":" +
                    std::to_string(row0+Bc) + ",0:" + std::to_string(d_head) +
                    "] → K_buf" + b,
                P({{"source",      "HBM.K[" + std::to_string(row0) + ":" +
                                   std::to_string(row0+Bc) + ",0:" +
                                   std::to_string(d_head) + "]"},
                   {"destination", "shared_ibuf.K_buf" + b},
                   {"rows", (int64_t)Bc},
                   {"cols", (int64_t)d_head}}),
                {prev_dma});

            auto load_V = add("dma_load", "dma",
                tag + "Load V[" + std::to_string(row0) + ":" +
                    std::to_string(row0+Bc) + ",0:" + std::to_string(d_head) +
                    "] → V_buf" + b,
                P({{"source",      "HBM.V[" + std::to_string(row0) + ":" +
                                   std::to_string(row0+Bc) + ",0:" +
                                   std::to_string(d_head) + "]"},
                   {"destination", "shared_ibuf.V_buf" + b},
                   {"rows", (int64_t)Bc},
                   {"cols", (int64_t)d_head}}),
                {load_K});

            // ─ Access core: Transpose K → K_T ─────────────────────────────
            auto transpose = add("transpose", "access_core",
                tag + "Transpose K_buf" + b + " → KT_buf" + b,
                P({{"source",      "shared_ibuf.K_buf" + b},
                   {"destination", "shared_ibuf.KT_buf" + b},
                   {"input_rows",  (int64_t)Bc},
                   {"input_cols",  (int64_t)d_head},
                   {"output_rows", (int64_t)d_head},
                   {"output_cols", (int64_t)Bc}}),
                {load_K});

            // ─ Systolic: weight-load K_T, then GEMM S = Q × K_T ──────────
            auto wl_K = add("weight_load", "systolic",
                tag + "Weight-load KT_buf" + b,
                P({{"source",      "shared_ibuf.KT_buf" + b},
                   {"destination", std::string("systolic_array.weight_reg")}}),
                {transpose});

            auto gemm_S = add("gemm", "systolic",
                tag + "GEMM S = Q × KT_buf" + b + "  [" +
                    std::to_string(Br) + "x" + std::to_string(Bc) + "]",
                P({{"source_a",   std::string("systolic_array.Q_operand")},
                   {"source_b",   "shared_ibuf.KT_buf" + b},
                   {"destination",std::string("shared_obuf.S_tile")},
                   {"M", (int64_t)Br},
                   {"K", (int64_t)d_head},
                   {"N", (int64_t)Bc}}),
                {stage_Q, wl_K});

            // ─ Vector: scale → rowmax → update_rowmax → exp_shift ─────────
            auto scale_S = add("scale", "vector_core",
                tag + "Scale S /= sqrt(d_head)",
                P({{"source",      std::string("shared_obuf.S_tile")},
                   {"destination", std::string("shared_obuf.S_tile")},
                   {"rows", (int64_t)Br}, {"cols", (int64_t)Bc},
                   {"scalar", std::string("1/sqrt(d_k)")}}),
                {gemm_S});

            auto rowmax_op = add("rowmax", "vector_core",
                tag + "rowmax(S) → rowmax_tmp",
                P({{"source",      std::string("shared_obuf.S_tile")},
                   {"destination", std::string("vector_scratch.rowmax_tmp")},
                   {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                {scale_S});

            auto update_rowmax = add("update_rowmax", "vector_core",
                tag + "update_rowmax → m, correction",
                P({{"source_m_old",           std::string("shared_obuf.m")},
                   {"source_rowmax",           std::string("vector_scratch.rowmax_tmp")},
                   {"destination_m",           std::string("shared_obuf.m")},
                   {"destination_correction",  std::string("shared_obuf.correction")},
                   {"length", (int64_t)Br}}),
                {rowmax_op, prev_update_rowmax});

            auto exp_shift = add("exp_shift", "vector_core",
                tag + "exp_shift P = exp(S - m_new)",
                P({{"source_matrix", std::string("shared_obuf.S_tile")},
                   {"source_shift",  std::string("shared_obuf.m")},
                   {"destination",   std::string("shared_ibuf.P_tile")},
                   {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                {update_rowmax});

            auto update_rowsum = add("update_rowsum", "vector_core",
                tag + "update_rowsum → l",
                P({{"source_p",          std::string("shared_ibuf.P_tile")},
                   {"source_correction", std::string("shared_obuf.correction")},
                   {"source_l_old",      std::string("shared_obuf.l")},
                   {"destination",       std::string("shared_obuf.l")},
                   {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                {exp_shift, update_rowmax, prev_update_rowsum});

            // scale_O can run in parallel with exp_shift on a second vector core
            auto scale_O = add("scale", "vector_core",
                tag + "Scale O_acc *= correction",
                P({{"source",       std::string("shared_obuf.O_acc")},
                   {"source_scale", std::string("shared_obuf.correction")},
                   {"destination",  std::string("shared_obuf.O_acc")},
                   {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
                {update_rowmax, prev_accumulate});

            // ─ DMA: stage P → systolic input register ──────────────────────
            auto stage_P = add("dma_stage", "dma",
                tag + "Stage P_tile → P_operand",
                P({{"source",      std::string("shared_ibuf.P_tile")},
                   {"destination", std::string("systolic_array.P_operand")},
                   {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                {exp_shift, load_V});

            // ─ Systolic: weight-load V, then GEMM Temp = P × V ───────────
            auto wl_V = add("weight_load", "systolic",
                tag + "Weight-load V_buf" + b,
                P({{"source",      "shared_ibuf.V_buf" + b},
                   {"destination", std::string("systolic_array.weight_reg")}}),
                {gemm_S, load_V});

            auto gemm_T = add("gemm", "systolic",
                tag + "GEMM Temp = P × V_buf" + b + "  [" +
                    std::to_string(Br) + "x" + std::to_string(d_head) + "]",
                P({{"source_a",    std::string("systolic_array.P_operand")},
                   {"source_b",    "shared_ibuf.V_buf" + b},
                   {"destination", std::string("shared_obuf.Temp")},
                   {"M", (int64_t)Br}, {"K", (int64_t)Bc}, {"N", (int64_t)d_head}}),
                {stage_P, wl_V});

            auto accumulate = add("accumulate", "vector_core",
                tag + "Accumulate O_acc += Temp",
                P({{"source_a",   std::string("shared_obuf.O_acc")},
                   {"source_b",   std::string("shared_obuf.Temp")},
                   {"destination",std::string("shared_obuf.O_acc")},
                   {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
                {scale_O, gemm_T});

            prev_update_rowmax = update_rowmax;
            prev_update_rowsum = update_rowsum;
            prev_accumulate    = accumulate;
            prev_dma           = load_V;
        }

        // ── Epilogue ──────────────────────────────────────────────────────
        auto normalize = add("normalize", "vector_core",
            "[Q" + std::to_string(qi) + "] Normalize O = O_acc / l",
            P({{"source_matrix", std::string("shared_obuf.O_acc")},
               {"source_denom",  std::string("shared_obuf.l")},
               {"destination",   std::string("shared_obuf.O_tile")},
               {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
            {prev_accumulate, prev_update_rowsum});

        auto logsumexp = add("logsumexp", "vector_core",
            "[Q" + std::to_string(qi) + "] Logsumexp L = m + log(l)",
            P({{"source_m",    std::string("shared_obuf.m")},
               {"source_l",    std::string("shared_obuf.l")},
               {"destination", std::string("shared_obuf.L_tile")},
               {"length", (int64_t)Br}}),
            {prev_update_rowmax, prev_update_rowsum, normalize});

        auto store_O = add("dma_store", "dma",
            "[Q" + std::to_string(qi) + "] Store O[" +
                std::to_string(q_row0) + ":" +
                std::to_string(q_row0+Br) + "] → HBM",
            P({{"source",      std::string("shared_obuf.O_tile")},
               {"destination", "HBM.O[" + std::to_string(q_row0) + ":" +
                               std::to_string(q_row0+Br) + ",0:" +
                               std::to_string(d_head) + "]"},
               {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
            {normalize, prev_dma});

        auto store_L = add("dma_store", "dma",
            "[Q" + std::to_string(qi) + "] Store L[" +
                std::to_string(q_row0) + ":" +
                std::to_string(q_row0+Br) + "] → HBM",
            P({{"source",      std::string("shared_obuf.L_tile")},
               {"destination", "HBM.L[" + std::to_string(q_row0) + ":" +
                               std::to_string(q_row0+Br) + "]"},
               {"length", (int64_t)Br}}),
            {logsumexp, store_O});

        prev_store_L = store_L;
    }

    return out;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// FA2Tiler public API
// ---------------------------------------------------------------------------

std::vector<Instruction> FA2Tiler::decompose(const WorkloadAttention& wl,
                                              const ArchConfig& /*arch*/,
                                              TensorStore& ts,
                                              InstructionId id_start)
{
    seed_hbm(wl, ts);

    // Seed on-chip scratch buffers (accessed on first write, but init ensures
    // no missing-key errors if a unit tries to read before the first write).
    size_t sz_BrDH = (size_t)wl.Br * wl.d_head;
    size_t sz_BrBc = (size_t)wl.Br * wl.Bc;
    size_t sz_Br   = (size_t)wl.Br;
    size_t sz_BcDH = (size_t)wl.Bc * wl.d_head;

    ts.init_zeros("shared_ibuf.Q_tile",     sz_BrDH);
    ts.init_zeros("shared_ibuf.K_buf0",     sz_BcDH);
    ts.init_zeros("shared_ibuf.K_buf1",     sz_BcDH);
    ts.init_zeros("shared_ibuf.V_buf0",     sz_BcDH);
    ts.init_zeros("shared_ibuf.V_buf1",     sz_BcDH);
    ts.init_zeros("shared_ibuf.KT_buf0",    sz_BcDH);
    ts.init_zeros("shared_ibuf.KT_buf1",    sz_BcDH);
    ts.init_zeros("shared_ibuf.P_tile",     sz_BrBc);
    ts.init_zeros("shared_obuf.S_tile",     sz_BrBc);
    ts.init_zeros("shared_obuf.Temp",       sz_BrDH);
    ts.init_zeros("shared_obuf.O_acc",      sz_BrDH);
    ts.init_zeros("shared_obuf.O_tile",     sz_BrDH);
    ts.init_neg_inf("shared_obuf.m",        sz_Br);
    ts.init_zeros("shared_obuf.l",          sz_Br);
    ts.init_zeros("shared_obuf.correction", sz_Br);
    ts.init_zeros("shared_obuf.L_tile",     sz_Br);
    ts.init_zeros("vector_scratch.rowmax_tmp", sz_Br);
    ts.init_zeros("systolic_array.Q_operand",  sz_BrDH);
    ts.init_zeros("systolic_array.P_operand",  sz_BrBc);

    return build(wl, id_start);
}

void FA2Tiler::seed_hbm(const WorkloadAttention& wl, TensorStore& ts)
{
    const int Nq     = (int)wl.Nq();
    const int Nkv    = (int)wl.Nkv();
    const int Br     = (int)wl.Br;
    const int Bc     = (int)wl.Bc;
    const int d_head = (int)wl.d_head;

    uint32_t seed = 1;
    for (int i = 0; i < Nq; ++i) {
        std::string key = "HBM.Q[" + std::to_string(i*Br) + ":" +
                          std::to_string((i+1)*Br) + ",0:" +
                          std::to_string(d_head) + "]";
        ts.init_random(key, (size_t)Br*d_head, -0.1f, 0.1f, seed++);
    }
    for (int j = 0; j < Nkv; ++j) {
        std::string kkey = "HBM.K[" + std::to_string(j*Bc) + ":" +
                           std::to_string((j+1)*Bc) + ",0:" +
                           std::to_string(d_head) + "]";
        std::string vkey = "HBM.V[" + std::to_string(j*Bc) + ":" +
                           std::to_string((j+1)*Bc) + ",0:" +
                           std::to_string(d_head) + "]";
        ts.init_random(kkey, (size_t)Bc*d_head, -0.1f, 0.1f, seed++);
        ts.init_random(vkey, (size_t)Bc*d_head, -0.1f, 0.1f, seed++);
    }

    // Placeholder output buffers (written by the simulation)
    for (int i = 0; i < Nq; ++i) {
        ts.init_zeros("HBM.O[" + std::to_string(i*Br) + ":" +
                      std::to_string((i+1)*Br) + ",0:" +
                      std::to_string(d_head) + "]", (size_t)Br*d_head);
        ts.init_zeros("HBM.L[" + std::to_string(i*Br) + ":" +
                      std::to_string((i+1)*Br) + "]", (size_t)Br);
    }
}

WorkloadAttention FA2Tiler::from_yaml_file(const std::string& path) {
    return from_yaml_string(YAML::Dump(YAML::LoadFile(path)));
}

WorkloadAttention FA2Tiler::from_yaml_string(const std::string& yaml_str) {
    auto root = YAML::Load(yaml_str);
    auto wl_node = root["workload"];
    if (!wl_node)
        throw std::runtime_error("Attention workload YAML must have a 'workload' section");

    WorkloadAttention wl;
    if (wl_node["seq_len"]) wl.seq_len = wl_node["seq_len"].as<uint32_t>();
    if (wl_node["d_head"])  wl.d_head  = wl_node["d_head"].as<uint32_t>();
    if (wl_node["Br"])      wl.Br      = wl_node["Br"].as<uint32_t>();
    if (wl_node["Bc"])      wl.Bc      = wl_node["Bc"].as<uint32_t>();
    if (wl_node["fill"])    wl.fill    = wl_node["fill"].as<std::string>();
    return wl;
}

}  // namespace sim
