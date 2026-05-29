#pragma once
#include "config/arch_config.h"
#include "core/tensor_store.h"
#include "schedule/instruction.h"
#include <string>
#include <vector>

namespace sim {

// ---------------------------------------------------------------------------
// WorkloadAttention — one full GQA attention layer forward pass.
// Used by the --workload pathway when workload.type == "attention".
//
// Standard (MHA):  num_gqa_groups=32, heads_per_group=1   (32 Q, 32 KV)
// GQA 8g×4h:       num_gqa_groups=8,  heads_per_group=4   (32 Q,  8 KV)
// MQA:             num_gqa_groups=1,  heads_per_group=32  (32 Q,  1 KV)
//
// LLaMA-3-8B GQA (8g×4h, seq=4096, d=128, Br=Bc=128):
//   Nq=32 Q-tiles, Nkv=32 KV-tiles, 8 groups × 4 heads
//   → 8 × 32 × 4 × 32 = 32768 GEMM pairs total
//   → K/V loaded ONCE per (group, kv-tile), reused across 4 Q-heads in group
//   → 4× KV HBM bandwidth saving vs. naïve MHA scheduling
// ---------------------------------------------------------------------------
struct WorkloadAttention {
    uint32_t    seq_len          = 4096;  // total sequence length
    uint32_t    d_head           = 128;   // attention head dimension
    uint32_t    Br               = 128;   // Q-tile rows  (≤ SA rows)
    uint32_t    Bc               = 128;   // KV-tile cols (≤ SA cols)
    uint32_t    num_gqa_groups   = 1;     // G: number of distinct KV heads
    uint32_t    heads_per_group  = 1;     // H: Q heads sharing one KV head
    std::string fill             = "random"; // TensorStore init: random | zeros | ones

    // Derived
    uint32_t num_q_heads()  const { return num_gqa_groups * heads_per_group; }
    uint32_t num_kv_heads() const { return num_gqa_groups; }
    uint32_t Nq()           const { return (seq_len + Br - 1) / Br; }
    uint32_t Nkv()          const { return (seq_len + Bc - 1) / Bc; }
};

// ---------------------------------------------------------------------------
// FA2Tiler — generates the pipelined GQA Flash-Attention-2 instruction DAG.
//
// Loop order  (outermost → innermost):
//   group (G)  →  Q-tile (Nq)  →  KV-tile (Nkv)  →  Q-head-in-group (H)
//
// GQA KV reuse strategy:
//   K[g][kv] and V[g][kv] are loaded ONCE per (group, kv-tile) and reused
//   across all H Q-heads within the group. The H GEMMs per KV tile run
//   serially on the single MXU. This gives H× KV HBM bandwidth saving
//   relative to naïvely repeating single-head FA2 H times.
//
// Pipeline:  per (group, Q-tile):
//   Prologue: all H Q-tiles loaded to Q_tile_h{h} in parallel with init fills.
//   KV loop:  load_K → load_V → transpose → [for h: wl_K → gemm_S → softmax
//             → stage_P → wl_V → gemm_T → accumulate] → next kv prefetch.
//   DMA prefetch:  load_K[kv+1] starts once gemm_S[kv, last_head] frees
//                  K_buf[kv%2]; load_V[kv+1] starts after gemm_T[kv, last_head].
//   Double-buffering:  K_buf0/K_buf1, V_buf0/V_buf1, KT_buf0/KT_buf1.
// ---------------------------------------------------------------------------
class FA2Tiler {
public:
    // Generate the full instruction DAG and seed TensorStore with random data.
    // id_start lets you offset IDs if combining with other schedules.
    static std::vector<Instruction> decompose(
        const WorkloadAttention& wl,
        const ArchConfig&        arch,
        TensorStore&             ts,
        InstructionId            id_start = 0);

    // Parse a workload YAML file (expects 'workload.type == "attention"').
    static WorkloadAttention from_yaml_file(const std::string& path);
    static WorkloadAttention from_yaml_string(const std::string& yaml);

    // Seed all HBM tiles in the TensorStore so dma_load ops have real data.
    static void seed_hbm(const WorkloadAttention& wl, TensorStore& ts);
};

}  // namespace sim
