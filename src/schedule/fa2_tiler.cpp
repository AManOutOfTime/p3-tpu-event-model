#include "schedule/fa2_tiler.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

namespace sim {

namespace {

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

static ParamMap P(std::initializer_list<std::pair<std::string,ParamVal>> items) {
    ParamMap m;
    for (auto& [k,v] : items) m[k] = v;
    return m;
}

// ---------------------------------------------------------------------------
// build() — generates the full GQA FA2 instruction DAG.
//
// Loop order: group (G) → Q-tile (Nq) → KV-tile (Nkv) → Q-head-in-group (H)
//
// GQA reuse: K[g][kv] and V[g][kv] are loaded ONCE per (group, kv) and
// reused across all H Q-heads.  The H systolic GEMMs run serially.
// ---------------------------------------------------------------------------
std::vector<Instruction> build(const WorkloadAttention& wl, InstructionId id0)
{
    std::vector<Instruction> out;
    InstructionId nid = id0;

    const int G      = (int)wl.num_gqa_groups;
    const int H      = (int)wl.heads_per_group;
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

    // hs(h) — suffix string for per-head buffer names
    auto hs = [](int h) { return "_h" + std::to_string(h); };

    bool          first_outer = true;
    InstructionId prev_store_L_last_head = id0;  // guarded by first_outer

    // ═══════════════════════════════════════════════════════════════════════
    // Outer loop: GQA group
    // ═══════════════════════════════════════════════════════════════════════
    for (int g = 0; g < G; ++g) {
        std::string gs = std::to_string(g);

        // ═══════════════════════════════════════════════════════════════════
        // Q-tile loop
        // ═══════════════════════════════════════════════════════════════════
        for (int qi = 0; qi < Nq; ++qi) {
            int q_row0 = qi * Br;
            std::string qis = std::to_string(qi);
            std::string outer_tag = "[G" + gs + "/Q" + qis + "] ";

            // Gate the first op of this (g,qi) block on the last store of the
            // previous block (or nothing for the very first block).
            std::vector<InstructionId> block_dep;
            if (!first_outer) block_dep = {prev_store_L_last_head};
            first_outer = false;

            // ── Prologue: for each Q-head h ──────────────────────────────
            // All H init-fills and Q-loads can issue in parallel (access_core
            // pool and DMA pool serialise the channel automatically).
            std::vector<InstructionId> stage_Q(H);  // stage_Q[h]

            for (int h = 0; h < H; ++h) {
                int q_head_global = g * H + h;
                std::string hpfx = outer_tag + "H" + std::to_string(h) + " ";

                auto init_O = add("init_fill", "access_core",
                    hpfx + "Init O_acc" + hs(h) + " = 0",
                    P({{"destination", "shared_obuf.O_acc" + hs(h)},
                       {"rows",  (int64_t)Br},
                       {"cols",  (int64_t)d_head},
                       {"init_value", (int64_t)0}}),
                    block_dep);

                auto init_m = add("init_fill", "access_core",
                    hpfx + "Init m" + hs(h) + " = -inf",
                    P({{"destination", "shared_obuf.m" + hs(h)},
                       {"length",     (int64_t)Br},
                       {"init_value", std::string("-inf")}}),
                    block_dep);

                auto init_l = add("init_fill", "access_core",
                    hpfx + "Init l" + hs(h) + " = 0",
                    P({{"destination", "shared_obuf.l" + hs(h)},
                       {"length",     (int64_t)Br},
                       {"init_value", (int64_t)0}}),
                    block_dep);

                // Q tile for head (g*H+h), Q-block qi
                std::string q_hbm = "HBM.Q_g" + gs + "_h" + std::to_string(h) +
                                    "[" + std::to_string(q_row0) + ":" +
                                    std::to_string(q_row0+Br) + ",0:" +
                                    std::to_string(d_head) + "]";
                auto load_Q = add("dma_load", "dma",
                    hpfx + "Load Q_g" + gs + "_h" + std::to_string(h) +
                        "[" + std::to_string(q_row0) + ":" +
                        std::to_string(q_row0+Br) + "]",
                    P({{"source",      q_hbm},
                       {"destination", "shared_ibuf.Q_tile" + hs(h)},
                       {"rows", (int64_t)Br},
                       {"cols", (int64_t)d_head}}),
                    block_dep);

                stage_Q[h] = add("dma_stage", "dma",
                    hpfx + "Stage Q_tile" + hs(h) + " → Q_operand" + hs(h),
                    P({{"source",      "shared_ibuf.Q_tile" + hs(h)},
                       {"destination", "systolic_array.Q_operand" + hs(h)},
                       {"rows", (int64_t)Br},
                       {"cols", (int64_t)d_head}}),
                    {load_Q});
                (void)init_O; (void)init_m; (void)init_l;
            }

            // Per-head running state (indexed by h)
            std::vector<InstructionId> prev_update_rowmax(H);
            std::vector<InstructionId> prev_update_rowsum(H);
            std::vector<InstructionId> prev_accumulate(H);
            for (int h = 0; h < H; ++h) {
                // IDs of the corresponding init_fill ops:
                // init_O, init_m, init_l are 3 ops per head, starting at
                // prologue base. We key off stage_Q[h] as a safe common dep
                // since all inits happen in the same block_dep wave.
                prev_update_rowmax[h] = stage_Q[h];
                prev_update_rowsum[h] = stage_Q[h];
                prev_accumulate[h]    = stage_Q[h];
            }

            // ---------------------------------------------------------------------------
            // DMA prefetch sentinels (per ping-pong buffer slot).
            // load_K[kv] waits until gemm_S[kv-2, last_head] frees K_buf[kv%2].
            // load_V[kv] waits until gemm_T[kv-2, last_head] frees V_buf[kv%2].
            // For kv=0,1 these point to stage_Q[0] (no prior user of those slots).
            // ---------------------------------------------------------------------------
            InstructionId prev_gemm_S_per_buf[2] = {stage_Q[0], stage_Q[0]};
            InstructionId prev_gemm_T_per_buf[2] = {stage_Q[0], stage_Q[0]};

            // MXU chain: all H heads within a kv-tile run serially; tracks the
            // last MXU op so the next head's weight_load waits for it.
            InstructionId prev_mxu = stage_Q[0];  // any valid sentinel before kv=0

            // ═════════════════════════════════════════════════════════════
            // KV-tile loop
            // ═════════════════════════════════════════════════════════════
            for (int kv = 0; kv < Nkv; ++kv) {
                int buf  = kv % 2;
                int row0 = kv * Bc;
                auto b   = std::to_string(buf);
                std::string kv_tag = outer_tag + "KV" + std::to_string(kv) + " ";

                // ── Shared K/V load (ONCE per kv-tile, reused across H heads) ──
                std::string k_hbm = "HBM.K_g" + gs + "[" +
                                    std::to_string(row0) + ":" +
                                    std::to_string(row0+Bc) + ",0:" +
                                    std::to_string(d_head) + "]";
                std::string v_hbm = "HBM.V_g" + gs + "[" +
                                    std::to_string(row0) + ":" +
                                    std::to_string(row0+Bc) + ",0:" +
                                    std::to_string(d_head) + "]";

                auto load_K = add("dma_load", "dma",
                    kv_tag + "Load K_g" + gs + "[" + std::to_string(row0) + ":" +
                        std::to_string(row0+Bc) + "] → K_buf" + b,
                    P({{"source",      k_hbm},
                       {"destination", "shared_ibuf.K_buf" + b},
                       {"rows", (int64_t)Bc},
                       {"cols", (int64_t)d_head}}),
                    {prev_gemm_S_per_buf[buf]});

                auto load_V = add("dma_load", "dma",
                    kv_tag + "Load V_g" + gs + "[" + std::to_string(row0) + ":" +
                        std::to_string(row0+Bc) + "] → V_buf" + b,
                    P({{"source",      v_hbm},
                       {"destination", "shared_ibuf.V_buf" + b},
                       {"rows", (int64_t)Bc},
                       {"cols", (int64_t)d_head}}),
                    {load_K, prev_gemm_T_per_buf[buf]});

                // Transpose K ONCE — shared across all H heads in this kv tile
                auto transpose = add("transpose", "access_core",
                    kv_tag + "Transpose K_buf" + b + " → KT_buf" + b,
                    P({{"source",      "shared_ibuf.K_buf" + b},
                       {"destination", "shared_ibuf.KT_buf" + b},
                       {"input_rows",  (int64_t)Bc},
                       {"input_cols",  (int64_t)d_head},
                       {"output_rows", (int64_t)d_head},
                       {"output_cols", (int64_t)Bc}}),
                    {load_K});

                InstructionId last_gemm_S = prev_gemm_S_per_buf[buf];
                InstructionId last_gemm_T = prev_gemm_T_per_buf[buf];

                // ══════════════════════════════════════════════════════════
                // Inner head loop: H GEMMs run serially on the MXU
                // ══════════════════════════════════════════════════════════
                for (int h = 0; h < H; ++h) {
                    std::string htag = kv_tag + "H" + std::to_string(h) + " ";

                    // weight-load K_T; first head waits for transpose,
                    // subsequent heads wait for the previous head's gemm_T
                    // (MXU serial chain) plus transpose (shared, already done).
                    auto wl_K = add("weight_load", "systolic",
                        htag + "WL KT_buf" + b,
                        P({{"source",      "shared_ibuf.KT_buf" + b},
                           {"destination", std::string("systolic_array.weight_reg")}}),
                        {transpose, prev_mxu});

                    auto gemm_S = add("gemm", "systolic",
                        htag + "GEMM S = Q" + hs(h) + " × KT_buf" + b +
                            " [" + std::to_string(Br) + "x" + std::to_string(Bc) + "]",
                        P({{"source_a",    "systolic_array.Q_operand" + hs(h)},
                           {"source_b",    "shared_ibuf.KT_buf" + b},
                           {"destination", std::string("shared_obuf.S_tile")},
                           {"M", (int64_t)Br},
                           {"K", (int64_t)d_head},
                           {"N", (int64_t)Bc}}),
                        {stage_Q[h], wl_K});

                    // ── Vector: scale → rowmax → update_rowmax → exp_shift ─
                    auto scale_S = add("scale", "vector_core",
                        htag + "Scale S /= sqrt(d_head)",
                        P({{"source",      std::string("shared_obuf.S_tile")},
                           {"destination", std::string("shared_obuf.S_tile")},
                           {"rows", (int64_t)Br}, {"cols", (int64_t)Bc},
                           {"scalar", std::string("1/sqrt(d_k)")}}),
                        {gemm_S});

                    auto rowmax_op = add("rowmax", "vector_core",
                        htag + "rowmax(S) → rowmax_tmp",
                        P({{"source",      std::string("shared_obuf.S_tile")},
                           {"destination", std::string("vector_scratch.rowmax_tmp")},
                           {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                        {scale_S});

                    auto update_rowmax = add("update_rowmax", "vector_core",
                        htag + "update_rowmax → m" + hs(h) + ", correction" + hs(h),
                        P({{"source_m_old",          "shared_obuf.m" + hs(h)},
                           {"source_rowmax",          std::string("vector_scratch.rowmax_tmp")},
                           {"destination_m",          "shared_obuf.m" + hs(h)},
                           {"destination_correction", "shared_obuf.correction" + hs(h)},
                           {"length", (int64_t)Br}}),
                        {rowmax_op, prev_update_rowmax[h]});

                    auto exp_shift = add("exp_shift", "vector_core",
                        htag + "exp_shift P = exp(S - m_new)",
                        P({{"source_matrix", std::string("shared_obuf.S_tile")},
                           {"source_shift",  "shared_obuf.m" + hs(h)},
                           {"destination",   std::string("shared_ibuf.P_tile")},
                           {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                        {update_rowmax});

                    auto update_rowsum = add("update_rowsum", "vector_core",
                        htag + "update_rowsum → l" + hs(h),
                        P({{"source_p",          std::string("shared_ibuf.P_tile")},
                           {"source_correction", "shared_obuf.correction" + hs(h)},
                           {"source_l_old",      "shared_obuf.l" + hs(h)},
                           {"destination",       "shared_obuf.l" + hs(h)},
                           {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                        {exp_shift, update_rowmax, prev_update_rowsum[h]});

                    // scale_O runs in parallel with exp_shift on a second vector core
                    auto scale_O = add("scale", "vector_core",
                        htag + "Scale O_acc" + hs(h) + " *= correction",
                        P({{"source",       "shared_obuf.O_acc" + hs(h)},
                           {"source_scale", "shared_obuf.correction" + hs(h)},
                           {"destination",  "shared_obuf.O_acc" + hs(h)},
                           {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
                        {update_rowmax, prev_accumulate[h]});

                    auto stage_P = add("dma_stage", "dma",
                        htag + "Stage P_tile → P_operand",
                        P({{"source",      std::string("shared_ibuf.P_tile")},
                           {"destination", std::string("systolic_array.P_operand")},
                           {"rows", (int64_t)Br}, {"cols", (int64_t)Bc}}),
                        {exp_shift, load_V});

                    auto wl_V = add("weight_load", "systolic",
                        htag + "WL V_buf" + b,
                        P({{"source",      "shared_ibuf.V_buf" + b},
                           {"destination", std::string("systolic_array.weight_reg")}}),
                        {gemm_S, load_V});

                    auto gemm_T = add("gemm", "systolic",
                        htag + "GEMM Temp = P × V_buf" + b +
                            " [" + std::to_string(Br) + "x" + std::to_string(d_head) + "]",
                        P({{"source_a",    std::string("systolic_array.P_operand")},
                           {"source_b",    "shared_ibuf.V_buf" + b},
                           {"destination", std::string("shared_obuf.Temp")},
                           {"M", (int64_t)Br}, {"K", (int64_t)Bc},
                           {"N", (int64_t)d_head}}),
                        {stage_P, wl_V});

                    auto accumulate = add("accumulate", "vector_core",
                        htag + "Accumulate O_acc" + hs(h) + " += Temp",
                        P({{"source_a",    "shared_obuf.O_acc" + hs(h)},
                           {"source_b",    std::string("shared_obuf.Temp")},
                           {"destination", "shared_obuf.O_acc" + hs(h)},
                           {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
                        {scale_O, gemm_T});

                    prev_update_rowmax[h] = update_rowmax;
                    prev_update_rowsum[h] = update_rowsum;
                    prev_accumulate[h]    = accumulate;
                    prev_mxu              = gemm_T;   // next head's wl_K waits here
                    last_gemm_S           = gemm_S;
                    last_gemm_T           = gemm_T;
                }  // end head loop

                // Update ping-pong sentinels to the last head's GEMMs
                prev_gemm_S_per_buf[buf] = last_gemm_S;
                prev_gemm_T_per_buf[buf] = last_gemm_T;

            }  // end KV-tile loop

            // ── Epilogue: once per Q-head ─────────────────────────────────
            InstructionId last_store_L = prev_store_L_last_head;  // will update
            for (int h = 0; h < H; ++h) {
                std::string hpfx = outer_tag + "H" + std::to_string(h) + " ";

                auto normalize = add("normalize", "vector_core",
                    hpfx + "Normalize O = O_acc" + hs(h) + " / l",
                    P({{"source_matrix", "shared_obuf.O_acc" + hs(h)},
                       {"source_denom",  "shared_obuf.l" + hs(h)},
                       {"destination",   "shared_obuf.O_tile" + hs(h)},
                       {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
                    {prev_accumulate[h], prev_update_rowsum[h]});

                auto logsumexp = add("logsumexp", "vector_core",
                    hpfx + "Logsumexp L = m + log(l)",
                    P({{"source_m",    "shared_obuf.m" + hs(h)},
                       {"source_l",    "shared_obuf.l" + hs(h)},
                       {"destination", "shared_obuf.L_tile" + hs(h)},
                       {"length", (int64_t)Br}}),
                    {prev_update_rowmax[h], prev_update_rowsum[h], normalize});

                auto store_O = add("dma_store", "dma",
                    hpfx + "Store O_g" + gs + "_h" + std::to_string(h) +
                        "[" + std::to_string(q_row0) + ":" +
                        std::to_string(q_row0+Br) + "] → HBM",
                    P({{"source",      "shared_obuf.O_tile" + hs(h)},
                       {"destination", "HBM.O_g" + gs + "_h" + std::to_string(h) +
                                       "[" + std::to_string(q_row0) + ":" +
                                       std::to_string(q_row0+Br) + ",0:" +
                                       std::to_string(d_head) + "]"},
                       {"rows", (int64_t)Br}, {"cols", (int64_t)d_head}}),
                    {normalize, prev_mxu});

                auto store_L = add("dma_store", "dma",
                    hpfx + "Store L_g" + gs + "_h" + std::to_string(h) +
                        "[" + std::to_string(q_row0) + ":" +
                        std::to_string(q_row0+Br) + "] → HBM",
                    P({{"source",      "shared_obuf.L_tile" + hs(h)},
                       {"destination", "HBM.L_g" + gs + "_h" + std::to_string(h) +
                                       "[" + std::to_string(q_row0) + ":" +
                                       std::to_string(q_row0+Br) + "]"},
                       {"length", (int64_t)Br}}),
                    {logsumexp, store_O});

                last_store_L = store_L;
            }
            prev_store_L_last_head = last_store_L;

        }  // end Q-tile loop
    }  // end group loop

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

    const int H      = (int)wl.heads_per_group;
    const int Br     = (int)wl.Br;
    const int Bc     = (int)wl.Bc;
    const int d_head = (int)wl.d_head;

    size_t sz_BrDH = (size_t)Br * d_head;
    size_t sz_BrBc = (size_t)Br * Bc;
    size_t sz_Br   = (size_t)Br;
    size_t sz_BcDH = (size_t)Bc * d_head;

    // Shared on-chip K/V buffers (ping-pong, one KV head at a time)
    ts.init_zeros("shared_ibuf.K_buf0",     sz_BcDH);
    ts.init_zeros("shared_ibuf.K_buf1",     sz_BcDH);
    ts.init_zeros("shared_ibuf.V_buf0",     sz_BcDH);
    ts.init_zeros("shared_ibuf.V_buf1",     sz_BcDH);
    ts.init_zeros("shared_ibuf.KT_buf0",    sz_BcDH);
    ts.init_zeros("shared_ibuf.KT_buf1",    sz_BcDH);
    ts.init_zeros("shared_ibuf.P_tile",     sz_BrBc);
    ts.init_zeros("shared_obuf.S_tile",     sz_BrBc);
    ts.init_zeros("shared_obuf.Temp",       sz_BrDH);
    ts.init_zeros("vector_scratch.rowmax_tmp", sz_Br);
    ts.init_zeros("systolic_array.P_operand",  sz_BrBc);

    // Per-head on-chip buffers
    for (int h = 0; h < H; ++h) {
        auto sfx = "_h" + std::to_string(h);
        ts.init_zeros("shared_ibuf.Q_tile"    + sfx, sz_BrDH);
        ts.init_zeros("shared_obuf.O_acc"     + sfx, sz_BrDH);
        ts.init_zeros("shared_obuf.O_tile"    + sfx, sz_BrDH);
        ts.init_neg_inf("shared_obuf.m"       + sfx, sz_Br);
        ts.init_zeros("shared_obuf.l"         + sfx, sz_Br);
        ts.init_zeros("shared_obuf.correction"+ sfx, sz_Br);
        ts.init_zeros("shared_obuf.L_tile"    + sfx, sz_Br);
        ts.init_zeros("systolic_array.Q_operand" + sfx, sz_BrDH);
    }

    return build(wl, id_start);
}

void FA2Tiler::seed_hbm(const WorkloadAttention& wl, TensorStore& ts)
{
    const int G      = (int)wl.num_gqa_groups;
    const int H      = (int)wl.heads_per_group;
    const int Nq     = (int)wl.Nq();
    const int Nkv    = (int)wl.Nkv();
    const int Br     = (int)wl.Br;
    const int Bc     = (int)wl.Bc;
    const int d_head = (int)wl.d_head;

    uint32_t seed = 1;

    for (int g = 0; g < G; ++g) {
        std::string gs = std::to_string(g);

        // Q tiles: one per (group, head, Q-tile)
        for (int h = 0; h < H; ++h) {
            for (int qi = 0; qi < Nq; ++qi) {
                std::string key = "HBM.Q_g" + gs + "_h" + std::to_string(h) +
                                  "[" + std::to_string(qi*Br) + ":" +
                                  std::to_string((qi+1)*Br) + ",0:" +
                                  std::to_string(d_head) + "]";
                ts.init_random(key, (size_t)Br*d_head, -0.1f, 0.1f, seed++);
            }
        }

        // K and V tiles: one per (group, KV-tile) — shared across H heads
        for (int kv = 0; kv < Nkv; ++kv) {
            std::string kkey = "HBM.K_g" + gs + "[" +
                               std::to_string(kv*Bc) + ":" +
                               std::to_string((kv+1)*Bc) + ",0:" +
                               std::to_string(d_head) + "]";
            std::string vkey = "HBM.V_g" + gs + "[" +
                               std::to_string(kv*Bc) + ":" +
                               std::to_string((kv+1)*Bc) + ",0:" +
                               std::to_string(d_head) + "]";
            ts.init_random(kkey, (size_t)Bc*d_head, -0.1f, 0.1f, seed++);
            ts.init_random(vkey, (size_t)Bc*d_head, -0.1f, 0.1f, seed++);
        }

        // Output buffers (written by simulation)
        for (int h = 0; h < H; ++h) {
            for (int qi = 0; qi < Nq; ++qi) {
                ts.init_zeros(
                    "HBM.O_g" + gs + "_h" + std::to_string(h) +
                    "[" + std::to_string(qi*Br) + ":" +
                    std::to_string((qi+1)*Br) + ",0:" +
                    std::to_string(d_head) + "]",
                    (size_t)Br * d_head);
                ts.init_zeros(
                    "HBM.L_g" + gs + "_h" + std::to_string(h) +
                    "[" + std::to_string(qi*Br) + ":" +
                    std::to_string((qi+1)*Br) + "]",
                    (size_t)Br);
            }
        }
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
    if (wl_node["seq_len"])          wl.seq_len         = wl_node["seq_len"].as<uint32_t>();
    if (wl_node["d_head"])           wl.d_head          = wl_node["d_head"].as<uint32_t>();
    if (wl_node["Br"])               wl.Br              = wl_node["Br"].as<uint32_t>();
    if (wl_node["Bc"])               wl.Bc              = wl_node["Bc"].as<uint32_t>();
    if (wl_node["num_gqa_groups"])   wl.num_gqa_groups  = wl_node["num_gqa_groups"].as<uint32_t>();
    if (wl_node["heads_per_group"])  wl.heads_per_group = wl_node["heads_per_group"].as<uint32_t>();
    if (wl_node["fill"])             wl.fill            = wl_node["fill"].as<std::string>();
    return wl;
}

}  // namespace sim
