#pragma once
#include "config/arch_config.h"
#include "schedule/instruction.h"
#include "schedule/schedule.h"
#include <string>
#include <vector>
#include <iostream>

namespace sim {

// ---------------------------------------------------------------------------
// WorkloadGemm — one GEMM described at workload level.
//
//   M  (or Br)    : output rows  — may be tiled across SA rows
//   K  (or d_head): streaming axis — NEVER tiled (streams fully per execution)
//   N  (or Bc)    : output cols  — may be tiled across SA cols
//
//   src_a  : symbolic full A matrix location
//   src_b  : symbolic full B matrix location
//   dst_c  : symbolic output C location
// ---------------------------------------------------------------------------
struct WorkloadGemm {
    uint32_t    M    = 0;   // Br
    uint32_t    K    = 0;   // d_head — never tiled
    uint32_t    N    = 0;   // Bc
    std::string src_a = "shared_ibuf.Q_tile";
    std::string src_b = "shared_ibuf.KT_tile";
    std::string dst_c = "shared_obuf.S_tile";
    std::string fill  = "random";
};

// ---------------------------------------------------------------------------
// SubTileInfo — metadata for one physical array execution.
// ---------------------------------------------------------------------------
struct SubTileInfo {
    uint32_t    sub_row   = 0;   // tile index along M
    uint32_t    sub_col   = 0;   // tile index along N
    uint32_t    row_start = 0;   // first row of A / C covered
    uint32_t    col_start = 0;   // first col of B / C covered
    uint32_t    tm        = 0;   // actual rows  (≤ SA_rows, partial on last)
    uint32_t    tn        = 0;   // actual cols  (≤ SA_cols, partial on last)
    uint32_t    k         = 0;   // always full K (d_head)
    std::string name_a;          // "shared_ibuf.Q_sub_r{i}"
    std::string name_b;          // "shared_ibuf.KT_sub_c{j}"
    std::string name_c;          // "shared_obuf.S_sub_r{i}_c{j}"
};

// ---------------------------------------------------------------------------
// TileDecomposition — result of Tiler::decompose().
// ---------------------------------------------------------------------------
struct TileDecomposition {
    WorkloadGemm             workload;
    uint32_t                 sa_rows  = 0;
    uint32_t                 sa_cols  = 0;
    uint32_t                 tiles_m  = 0;   // ceil(M / SA_rows)
    uint32_t                 tiles_n  = 0;   // ceil(N / SA_cols)
    std::vector<SubTileInfo> tiles;
    std::vector<Instruction> instructions;   // STAGE + GEMM per tile
};

// ---------------------------------------------------------------------------
// Tiler — decomposes a WorkloadGemm into per-tile STAGE + GEMM instructions.
//
// LOOP ORDER  (Q-stationary)
// ──────────────────────────
//   outer i  → Q row sub-tiles   Q stays in IBUF, reused for all j
//     STAGE: shared_ibuf.Q_sub_r{i} → systolic_array.Q_operand
//
//     inner j  → KT col sub-tiles  KT cycles through
//       GEMM: Q_operand × KT_sub_c{j} → S_sub_r{i}_c{j}
//
// WHY K IS NEVER TILED
//   K (d_head) is the streaming/accumulation axis. One row of B enters the
//   array per cycle — that IS the K term in per_tile = K + fill_latency.
//   Tiling K would require partial-sum accumulation across multiple
//   executions (K-split), which is a future extension.
//
// DEPENDENCIES
//   STAGE_i   : depends on last GEMM of previous Q sub-tile (array must drain)
//   GEMM_i_j  : depends on STAGE_i  (Q_operand ready)
//             + previous GEMM on array (structural hazard — one array)
//
// SINGLE TILE
//   If M ≤ SA_rows AND N ≤ SA_cols: 1 STAGE + 1 GEMM, no sub-tiling needed.
// ---------------------------------------------------------------------------
class Tiler {
public:
    // Decompose workload given arch config.
    static TileDecomposition decompose(const WorkloadGemm& wl,
                                       const ArchConfig&   arch,
                                       InstructionId       id_start = 0);

    // Rewrite oversized GEMM instructions into the same STAGE+GEMM sub-events
    // produced by decompose(). Instructions that already fit the array are left
    // unchanged; dependencies on expanded instructions are rewired to the last
    // generated sub-event.
    static Schedule expand_gemm_subtiles(const Schedule& sched,
                                         const ArchConfig& arch);

    // Parse a workload YAML file / string.
    static WorkloadGemm from_yaml_file(const std::string& path);
    static WorkloadGemm from_yaml_string(const std::string& yaml);

    // Print decomposition summary to os.
    static void print_decomposition(const TileDecomposition& td,
                                    std::ostream& os = std::cout);

};

}  // namespace sim
