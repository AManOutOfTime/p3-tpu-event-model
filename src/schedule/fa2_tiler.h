#pragma once
#include "config/arch_config.h"
#include "core/tensor_store.h"
#include "schedule/instruction.h"
#include <string>
#include <vector>

namespace sim {

// ---------------------------------------------------------------------------
// WorkloadAttention — one full-head FA2 forward pass described at workload
// level.  Used by the --workload pathway when workload.type == "attention".
//
// LLaMA-3-8B single GQA head: seq_len=4096, d_head=128, Br=128, Bc=128
//   → Nq = seq_len / Br = 32 Q-tiles
//   → Nkv= seq_len / Bc = 32 KV-tiles
//   → 32 × 32 = 1024 GEMM pairs  (Q×K^T  and  P×V)
// ---------------------------------------------------------------------------
struct WorkloadAttention {
    uint32_t    seq_len = 4096;   // total sequence length
    uint32_t    d_head  = 128;    // attention head dimension
    uint32_t    Br      = 128;    // Q-tile rows  (≤ SA rows)
    uint32_t    Bc      = 128;    // KV-tile cols (≤ SA cols)
    std::string fill    = "random"; // TensorStore init: random | zeros | ones

    // Derived
    uint32_t Nq()  const { return (seq_len + Br - 1) / Br; }
    uint32_t Nkv() const { return (seq_len + Bc - 1) / Bc; }
};

// ---------------------------------------------------------------------------
// FA2Tiler — generates the pipelined Flash-Attention-2 instruction DAG.
//
// The generated schedule mirrors the pipelined reference from:
//   Norrie et al., "The Design Process for Google's Training Chips:
//   TPUv2 and TPUv3," ISSCC 2021 (Figure 6).
//
// Pipeline strategy (per Q-tile, inner KV loop):
//   - DMA prefetch: Load K[j+1]/V[j+1] while weight_load K_T[j] / GEMM[j]
//     runs on the systolic (DMA unit is free during systolic compute).
//   - Vector parallelism: exp_shift[j] and scale_O[j] both depend only on
//     update_rowmax[j] → dispatched to separate vector cores in parallel.
//   - scale_S[j+1] (depends on GEMM_S[j+1]) can overlap with
//     update_rowsum[j] on a third vector core.
//   - Transpose K[j+1] (access_core) runs concurrently with load_V[j].
//
// Double-buffering:
//   shared_ibuf.K_buf0 / K_buf1   ← K tiles ping-pong
//   shared_ibuf.V_buf0 / V_buf1   ← V tiles ping-pong
//   shared_ibuf.KT_buf0/ KT_buf1  ← K^T transposed ping-pong
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
