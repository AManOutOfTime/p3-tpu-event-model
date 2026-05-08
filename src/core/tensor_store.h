#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace sim {

// ---------------------------------------------------------------------------
// TensorStore — shared, named float matrix storage.
//
// Units read and write named buffers here; the DMA unit initialises them on
// load, the systolic array writes the GEMM result on compute.  Every buffer
// is a flat row-major float vector.
//
// Naming convention used throughout the codebase:
//   "A"      M×K  input matrix
//   "B"      K×N  weight matrix
//   "C"      M×N  output / accumulator
//   "Q_tile", "K_tile", "V_tile", "S_tile", "P_tile", "O_tile"  — FA2 tiles
// ---------------------------------------------------------------------------
class TensorStore {
public:
    // ── Storage primitives ────────────────────────────────────────────────
    void set(const std::string& name, std::vector<float> data) {
        store_[name] = std::move(data);
    }

    const std::vector<float>& get(const std::string& name) const {
        auto it = store_.find(name);
        if (it == store_.end())
            throw std::runtime_error("TensorStore: buffer '" + name + "' not found");
        return it->second;
    }

    std::vector<float>& get_mutable(const std::string& name) {
        auto it = store_.find(name);
        if (it == store_.end())
            throw std::runtime_error("TensorStore: buffer '" + name + "' not found");
        return it->second;
    }

    bool has(const std::string& name) const {
        return store_.count(name) > 0;
    }

    void remove(const std::string& name) { store_.erase(name); }

    size_t size(const std::string& name) const { return get(name).size(); }

    // ── Initialisation helpers ─────────────────────────────────────────────
    void init_zeros(const std::string& name, size_t n) {
        store_[name].assign(n, 0.0f);
    }

    void init_ones(const std::string& name, size_t n) {
        store_[name].assign(n, 1.0f);
    }

    void init_value(const std::string& name, size_t n, float v) {
        store_[name].assign(n, v);
    }

    // Deterministic LCG random fill in [lo, hi].
    void init_random(const std::string& name, size_t n,
                     float lo = 0.0f, float hi = 1.0f,
                     uint32_t seed = 42) {
        auto& buf = store_[name];
        buf.resize(n);
        uint32_t s = seed;
        for (size_t i = 0; i < n; i++) {
            s = s * 1664525u + 1013904223u;  // LCG
            buf[i] = lo + (hi - lo) * static_cast<float>(s >> 8) / static_cast<float>(1 << 24);
        }
    }

    // N×N identity matrix (1 on diagonal, 0 elsewhere).
    void init_identity(const std::string& name, uint32_t N) {
        auto& buf = store_[name];
        buf.assign(static_cast<size_t>(N) * N, 0.0f);
        for (uint32_t i = 0; i < N; i++)
            buf[i * N + i] = 1.0f;
    }

    // Row-major sequential fill: 0, 1, 2, ... (good for tracing small tiles).
    void init_sequential(const std::string& name, size_t n, float start = 0.0f) {
        auto& buf = store_[name];
        buf.resize(n);
        for (size_t i = 0; i < n; i++)
            buf[i] = start + static_cast<float>(i);
    }

    // ── Verification helpers ───────────────────────────────────────────────
    // Maximum absolute difference between two same-sized buffers.
    float max_abs_diff(const std::string& a, const std::string& b) const {
        const auto& va = get(a);
        const auto& vb = get(b);
        if (va.size() != vb.size())
            throw std::runtime_error("max_abs_diff: size mismatch");
        float mx = 0.0f;
        for (size_t i = 0; i < va.size(); i++)
            mx = std::max(mx, std::abs(va[i] - vb[i]));
        return mx;
    }

    // Root-mean-square error.
    float rmse(const std::string& a, const std::string& b) const {
        const auto& va = get(a);
        const auto& vb = get(b);
        if (va.size() != vb.size())
            throw std::runtime_error("rmse: size mismatch");
        double sum = 0.0;
        for (size_t i = 0; i < va.size(); i++) {
            double d = va[i] - vb[i];
            sum += d * d;
        }
        return static_cast<float>(std::sqrt(sum / va.size()));
    }

    void copy(const std::string& src, const std::string& dst) {
        store_[dst] = get(src);
    }

    // ── Tiling helpers ─────────────────────────────────────────────────────

    // Extract a row-slice of a matrix: rows [row_start, row_start+nrows).
    // Source matrix is `src_name` with `full_cols` columns (row-major).
    // Result written to `dst_name`: shape nrows × full_cols.
    void slice_rows(const std::string& src_name, const std::string& dst_name,
                    uint32_t row_start, uint32_t nrows, uint32_t full_cols) {
        const auto& src = get(src_name);
        std::vector<float> dst(static_cast<size_t>(nrows) * full_cols);
        for (uint32_t r = 0; r < nrows; r++)
            for (uint32_t c = 0; c < full_cols; c++)
                dst[r * full_cols + c] = src[(row_start + r) * full_cols + c];
        store_[dst_name] = std::move(dst);
    }

    // Extract a column-slice of a matrix: cols [col_start, col_start+ncols).
    // Source matrix is `src_name` with `full_cols` columns (row-major).
    // Result written to `dst_name`: shape full_rows × ncols.
    void slice_cols(const std::string& src_name, const std::string& dst_name,
                    uint32_t col_start, uint32_t ncols,
                    uint32_t full_rows, uint32_t full_cols) {
        const auto& src = get(src_name);
        std::vector<float> dst(static_cast<size_t>(full_rows) * ncols);
        for (uint32_t r = 0; r < full_rows; r++)
            for (uint32_t c = 0; c < ncols; c++)
                dst[r * ncols + c] = src[r * full_cols + (col_start + c)];
        store_[dst_name] = std::move(dst);
    }

    // Write a tile result back into the correct position of the full output.
    // `tile_name` is a matrix of shape tile_rows × tile_cols.
    // `dst_name` is the full output matrix of shape full_rows × full_cols.
    // The tile is placed at row_start, col_start within dst.
    void place_tile(const std::string& tile_name, const std::string& dst_name,
                    uint32_t row_start, uint32_t col_start,
                    uint32_t tile_rows, uint32_t tile_cols,
                    uint32_t full_cols) {
        const auto& tile = get(tile_name);
        auto& dst = get_mutable(dst_name);
        for (uint32_t r = 0; r < tile_rows; r++)
            for (uint32_t c = 0; c < tile_cols; c++)
                dst[(row_start + r) * full_cols + (col_start + c)]
                    = tile[r * tile_cols + c];
    }

    // ── Debug printing ─────────────────────────────────────────────────────
    // Print up to `max_rows` rows of a matrix buffer.
    void print(const std::string& name, uint32_t rows, uint32_t cols,
               uint32_t max_rows = 8, std::ostream& os = std::cout) const {
        const auto& buf = get(name);
        os << "  TensorStore[\"" << name << "\"]  [" << rows << "x" << cols << "]:\n";
        uint32_t show = std::min(rows, max_rows);
        for (uint32_t r = 0; r < show; r++) {
            os << "    row " << std::setw(3) << r << ": ";
            uint32_t show_cols = std::min(cols, 8u);
            for (uint32_t c = 0; c < show_cols; c++)
                os << std::fixed << std::setprecision(4) << std::setw(10) << buf[r*cols+c];
            if (show_cols < cols) os << " ...";
            os << "\n";
        }
        if (show < rows) os << "    ... (" << rows - show << " more rows)\n";
    }

private:
    std::unordered_map<std::string, std::vector<float>> store_;
};

}  // namespace sim
