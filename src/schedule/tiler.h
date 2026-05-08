#pragma once
#include "config/arch_config.h"
#include "core/tensor_store.h"
#include "schedule/instruction.h"
#include <string>
#include <vector>
#include <iostream>

namespace sim {

// ---------------------------------------------------------------------------
// WorkloadGemm — describes the GEMM at the FA2 tile level.
//
//   M        = Br       — query tile rows
//   K        = d_head   — head dimension (NOT tiled, streams fully through)
//   N        = Bc       — key tile cols
//
//   src_a    = "shared_ibuf.Q_tile"    — Q tile already in IBUF (from DMA)
//   src_b    = "shared_ibuf.KT_tile"   — K^T tile already in IBUF (from DMA)
//   dst_c    = "shared_obuf.S_tile"    — output goes to OBUF
//
// If M <= SA_rows AND N <= SA_cols: single execution, no sub-tiling needed.
// Otherwise: Tiler decomposes into sub-tiles that fit the array.
// ---------------------------------------------------------------------------
struct WorkloadGemm {
    uint32_t    M    = 0;    // Br   — query rows
    uint32_t    K    = 0;    // d_head — streaming dimension (never tiled)
    uint32_t    N    = 0;    // Bc   — key cols
    std::string src_a = "shared_ibuf.Q_tile";   // always lives in IBUF
    std::string src_b = "shared_ibuf.KT_tile";  // always lives in IBUF
    std::string dst_c = "shared_obuf.S_tile";   // always lands in OBUF
    std::string fill  = "random";
};

// ---------------------------------------------------------------------------
// SubTileInfo — one physical array execution.
// ---------------------------------------------------------------------------
struct SubTileInfo {
    uint32_t sub_row = 0;      // sub-tile index in M (Q rows)
    uint32_t sub_col = 0;      // sub-tile index in N (K^T cols)
    uint32_t row_start = 0;    // first row of Q_tile this sub-tile covers
    uint32_t col_start = 0;    // first col of KT_tile this sub-tile covers
    uint32_t tm = 0;           // actual rows  (≤ SA_rows, partial on last)
    uint32_t tn = 0;           // actual cols  (≤ SA_cols, partial on last)
    uint32_t k  = 0;           // always full d_head
    std::string name_a;        // "shared_ibuf.Q_sub_r{i}"
    std::string name_b;        // "shared_ibuf.KT_sub_c{j}"
    std::string name_c;        // "shared_obuf.S_sub_r{i}_c{j}"
};

// ---------------------------------------------------------------------------
// TileDecomposition — result of Tiler::decompose().
// ---------------------------------------------------------------------------
struct TileDecomposition {
    WorkloadGemm              workload;
    uint32_t                  sa_rows  = 0;
    uint32_t                  sa_cols  = 0;
    uint32_t                  tiles_m  = 0;  // ceil(M / SA_rows)
    uint32_t                  tiles_n  = 0;  // ceil(N / SA_cols)
    std::vector<SubTileInfo>  tiles;          // ordered: outer=Q, inner=K^T
    std::vector<Instruction>  instructions;   // one gemm per tile
};

// ---------------------------------------------------------------------------
// Tiler — decomposes a WorkloadGemm into sub-tile GEMM instructions.
//
// LOOP ORDER (Q-stationary)
// ─────────────────────────────────────────────────────────────────────────
//   for i in Q row sub-tiles:          ← outer: Q stays in IBUF
//       slice Q_tile rows → shared_ibuf.Q_sub_r{i}   [tm × d_head]
//
//       for j in K^T col sub-tiles:    ← inner: K^T cycles through
//           slice KT_tile cols → shared_ibuf.KT_sub_c{j}  [d_head × tn]
//           GEMM: Q_sub_r{i} × KT_sub_c{j} → shared_obuf.S_sub_r{i}_c{j}
//
//   assemble: S_sub tiles → shared_obuf.S_tile  [M × N]
//
// WHY Q IS THE OUTER LOOP
//   Q sub-tile i is reused for ALL j K^T sub-tiles.
//   By keeping Q in IBUF across the inner loop, we avoid reloading it
//   for each K^T column — minimising IBUF traffic.
//
// K IS NEVER TILED
//   d_head streams fully through the array in one execution.
//   It is the K accumulation axis: per_tile = d_head + fill_latency.
//   Tiling K would require external accumulation in OBUF (K-split),
//   which is a future extension.
//
// SINGLE TILE (no decomposition needed)
//   If M ≤ SA_rows and N ≤ SA_cols: one instruction, no slicing.
//   source_a/b/dst pass through unchanged.
// ---------------------------------------------------------------------------
class Tiler {
public:
    static TileDecomposition decompose(const WorkloadGemm& wl,
                                       const ArchConfig&   arch,
                                       TensorStore&        ts,
                                       InstructionId       id_start = 0);

    static WorkloadGemm from_yaml_file(const std::string& path);
    static WorkloadGemm from_yaml_string(const std::string& yaml);

    static void print_decomposition(const TileDecomposition& td,
                                    std::ostream& os = std::cout);

    // Assemble S_sub tiles → full S_tile in shared_obuf after simulation.
    static void assemble_output(const TileDecomposition& td, TensorStore& ts);
};

}  // namespace sim
