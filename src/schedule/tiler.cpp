#include "schedule/tiler.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <unordered_map>

namespace sim {

// ---------------------------------------------------------------------------
// YAML parsing
// ---------------------------------------------------------------------------
WorkloadGemm Tiler::from_yaml_string(const std::string& yaml) {
    YAML::Node root = YAML::Load(yaml);
    auto w = root["workload"];
    if (!w) throw std::runtime_error("Tiler: missing 'workload' key");

    WorkloadGemm wl;
    // Accept FA2 names (Br/d_head/Bc) or generic (M/K/N)
    if (w["Br"])     wl.M = w["Br"].as<uint32_t>();
    if (w["M"])      wl.M = w["M"].as<uint32_t>();
    if (w["d_head"]) wl.K = w["d_head"].as<uint32_t>();
    if (w["K"])      wl.K = w["K"].as<uint32_t>();
    if (w["Bc"])     wl.N = w["Bc"].as<uint32_t>();
    if (w["N"])      wl.N = w["N"].as<uint32_t>();
    if (w["src_a"])  wl.src_a = w["src_a"].as<std::string>();
    if (w["src_b"])  wl.src_b = w["src_b"].as<std::string>();
    if (w["dst_c"])  wl.dst_c = w["dst_c"].as<std::string>();
    if (w["fill"])   wl.fill  = w["fill"].as<std::string>();
    return wl;
}

WorkloadGemm Tiler::from_yaml_file(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    std::ostringstream ss; ss << root;
    return from_yaml_string(ss.str());
}

// ---------------------------------------------------------------------------
// decompose
// ---------------------------------------------------------------------------
TileDecomposition Tiler::decompose(const WorkloadGemm& wl,
                                   const ArchConfig&   arch,
                                   InstructionId       id_start) {
    if (!wl.M || !wl.K || !wl.N)
        throw std::runtime_error("Tiler: M/K/N must all be > 0");

    const uint32_t SA_R = arch.systolic.rows;
    const uint32_t SA_C = arch.systolic.cols;
    const uint32_t TM   = (wl.M + SA_R - 1) / SA_R;   // Q row sub-tiles
    const uint32_t TN   = (wl.N + SA_C - 1) / SA_C;   // KT col sub-tiles

    TileDecomposition td;
    td.workload = wl;
    td.sa_rows  = SA_R;
    td.sa_cols  = SA_C;
    td.tiles_m  = TM;
    td.tiles_n  = TN;

    InstructionId next_id      = id_start;
    InstructionId prev_gemm_id = 0;
    bool          has_prev_gemm = false;

    // ── Q-stationary outer loop ─────────────────────────────────────────
    for (uint32_t i = 0; i < TM; i++) {
        const uint32_t row_start = i * SA_R;
        const uint32_t tm        = std::min(SA_R, wl.M - row_start);

        const std::string name_a = "shared_ibuf.Q_sub_r" + std::to_string(i);

        // ── STAGE: IBUF → systolic_array.Q_operand ─────────────────────
        // Must wait for array to drain (prev GEMM) before overwriting Q_operand.
        Instruction stage;
        stage.id   = next_id++;
        stage.op   = "dma_stage";
        stage.unit = "dma";
        if (has_prev_gemm)
            stage.depends_on.push_back(prev_gemm_id);
        stage.params["source"]      = name_a;
        stage.params["destination"] = std::string("systolic_array.Q_operand");
        stage.params["rows"]        = static_cast<int64_t>(tm);
        stage.params["cols"]        = static_cast<int64_t>(wl.K);
        stage.label = "STAGE Q_sub_r" + std::to_string(i)
                    + " [rows " + std::to_string(row_start)
                    + ":" + std::to_string(row_start + tm) + "]";
        td.instructions.push_back(stage);
        const InstructionId stage_id = stage.id;

        // ── KT inner loop ───────────────────────────────────────────────
        for (uint32_t j = 0; j < TN; j++) {
            const uint32_t col_start = j * SA_C;
            const uint32_t tn        = std::min(SA_C, wl.N - col_start);
            const std::string name_b = "shared_ibuf.KT_sub_c" + std::to_string(j);
            const std::string name_c = "shared_obuf.S_sub_r"
                                     + std::to_string(i) + "_c"
                                     + std::to_string(j);

            SubTileInfo info;
            info.sub_row   = i;
            info.sub_col   = j;
            info.row_start = row_start;
            info.col_start = col_start;
            info.tm        = tm;
            info.tn        = tn;
            info.k         = wl.K;
            info.name_a    = name_a;
            info.name_b    = name_b;
            info.name_c    = name_c;
            td.tiles.push_back(info);

            // ── GEMM ─────────────────────────────────────────────────────
            Instruction gemm;
            gemm.id   = next_id++;
            gemm.op   = "gemm";
            gemm.unit = "systolic";
            gemm.depends_on.push_back(stage_id);      // Q_operand ready
            if (has_prev_gemm)
                gemm.depends_on.push_back(prev_gemm_id); // array free
            gemm.params["M"]           = static_cast<int64_t>(tm);
            gemm.params["K"]           = static_cast<int64_t>(wl.K);
            gemm.params["N"]           = static_cast<int64_t>(tn);
            gemm.params["source_a"]    = std::string("systolic_array.Q_operand");
            gemm.params["source_b"]    = name_b;
            gemm.params["destination"] = name_c;
            gemm.label = "S[r" + std::to_string(i) + ",c" + std::to_string(j)
                       + "] Q_operand[" + std::to_string(tm)
                       + "x" + std::to_string(wl.K) + "]"
                       + " x KT_sub_c" + std::to_string(j)
                       + "[" + std::to_string(wl.K)
                       + "x" + std::to_string(tn) + "]";
            td.instructions.push_back(gemm);
            prev_gemm_id  = gemm.id;
            has_prev_gemm = true;
        }
    }

    return td;
}

// ---------------------------------------------------------------------------
// expand_gemm_subtiles
// ---------------------------------------------------------------------------
Schedule Tiler::expand_gemm_subtiles(const Schedule& sched,
                                     const ArchConfig& arch) {
    Schedule expanded;
    expanded.instructions.reserve(sched.instructions.size());

    InstructionId next_id = 0;
    for (const auto& inst : sched.instructions)
        next_id = std::max(next_id, inst.id + 1);

    std::unordered_map<InstructionId, InstructionId> terminal_by_original;

    for (const auto& inst : sched.instructions) {
        if (inst.op != "gemm") {
            expanded.instructions.push_back(inst);
            continue;
        }

        WorkloadGemm wl;
        wl.M = static_cast<uint32_t>(pget_int(inst.params, "M", arch.systolic.rows));
        wl.K = static_cast<uint32_t>(pget_int(inst.params, "K", arch.systolic.d_head));
        wl.N = static_cast<uint32_t>(pget_int(inst.params, "N", arch.systolic.cols));
        wl.src_a = pget_str(inst.params, "source_a");
        wl.src_b = pget_str(inst.params, "source_b");
        wl.dst_c = pget_str(inst.params, "destination");
        wl.fill = "zeros";

        const bool needs_subtiles = wl.M > arch.systolic.rows
                                 || wl.N > arch.systolic.cols;
        if (!needs_subtiles) {
            expanded.instructions.push_back(inst);
            continue;
        }

        TileDecomposition td = Tiler::decompose(wl, arch, next_id);
        if (td.instructions.empty()) {
            expanded.instructions.push_back(inst);
            continue;
        }

        td.instructions.front().depends_on.insert(
            td.instructions.front().depends_on.end(),
            inst.depends_on.begin(), inst.depends_on.end());

        for (auto& sub : td.instructions) {
            if (!inst.label.empty())
                sub.label = inst.label + " / " + sub.label;
            sub.params["subtiled_from"] = static_cast<int64_t>(inst.id);
            expanded.instructions.push_back(std::move(sub));
        }

        terminal_by_original[inst.id] = expanded.instructions.back().id;
        next_id = expanded.instructions.back().id + 1;
    }

    for (auto& inst : expanded.instructions) {
        for (auto& dep : inst.depends_on) {
            auto it = terminal_by_original.find(dep);
            if (it != terminal_by_original.end())
                dep = it->second;
        }
    }

    expanded.validate();
    return expanded;
}

// ---------------------------------------------------------------------------
// print_decomposition
// ---------------------------------------------------------------------------
void Tiler::print_decomposition(const TileDecomposition& td, std::ostream& os) {
    const auto& wl = td.workload;
    const bool  single = (td.tiles_m == 1 && td.tiles_n == 1);

    os << "\n┌─ Tile decomposition ───────────────────────────────────────────\n"
       << "│  " << wl.src_a << " [" << wl.M << "×" << wl.K << "]"
       << "  x  " << wl.src_b << " [" << wl.K << "×" << wl.N << "]"
       << "  →  " << wl.dst_c << " [" << wl.M << "×" << wl.N << "]\n"
       << "│\n"
       << "│  Array           : " << td.sa_rows << "×" << td.sa_cols << "\n"
       << "│  Q  row sub-tiles: " << td.tiles_m
       << "  (ceil(" << wl.M << "/" << td.sa_rows << "))\n"
       << "│  KT col sub-tiles: " << td.tiles_n
       << "  (ceil(" << wl.N << "/" << td.sa_cols << "))\n"
       << "│  Array executions: " << td.tiles_m * td.tiles_n << "\n"
       << "│  K=" << wl.K << " — streams fully through each execution, never tiled\n"
       << "│\n";

    if (single) {
        os << "│  Fits in one execution — no sub-tiling needed\n";
    } else {
        os << "│  Loop order: outer=Q rows (stationary), inner=KT cols\n"
           << "│  Q sub-tile staged once into Q_operand, reused for "
           << td.tiles_n << " KT col(s)\n│\n";
    }

    // Instruction table
    os << "├────────────────────────────────────────────────────────────────\n"
       << "│  " << std::left
       << std::setw(5)  << "id"
       << std::setw(12) << "op"
       << std::setw(30) << "label"
       << "deps\n"
       << "│  " << std::string(60, '-') << "\n";

    for (const auto& inst : td.instructions) {
        std::string deps;
        for (auto d : inst.depends_on) deps += std::to_string(d) + " ";
        std::string lbl = inst.label.size() > 28
                        ? inst.label.substr(0, 25) + "..."
                        : inst.label;
        os << "│  " << std::left
           << std::setw(5)  << inst.id
           << std::setw(12) << inst.op
           << std::setw(30) << lbl
           << (deps.empty() ? "-" : deps) << "\n";
    }
    os << "└────────────────────────────────────────────────────────────────\n\n";
}

}  // namespace sim
