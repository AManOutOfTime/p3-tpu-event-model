#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace sim {

// ---------------------------------------------------------------------------
// TensorStore — named float matrix storage shared across all hardware units.
//
// Buffers represent physical on-chip SRAM regions:
//   "shared_ibuf.*"           — shared input buffer (IBUF)
//   "shared_obuf.*"           — shared output buffer (OBUF)
//   "systolic_array.Q_operand"— array input register (staged from IBUF)
//   "systolic_array.P_operand"— array input register (staged from IBUF)
//   "vector_scratch.*"        — vector core scratch space
//
// Units read/write named buffers here; the systolic unit writes GEMM results
// at OP_DONE; the DMA unit copies buffers at OP_DONE (staging).
// ---------------------------------------------------------------------------
class TensorStore {
public:
    void set(const std::string& name, std::vector<float> data) {
        store_[name] = std::move(data);
    }
    const std::vector<float>& get(const std::string& name) const {
        auto it = store_.find(name);
        if (it == store_.end())
            throw std::runtime_error("TensorStore: '" + name + "' not found");
        return it->second;
    }
    std::vector<float>& get_mutable(const std::string& name) {
        auto it = store_.find(name);
        if (it == store_.end())
            throw std::runtime_error("TensorStore: '" + name + "' not found");
        return it->second;
    }
    bool   has(const std::string& name) const { return store_.count(name) > 0; }
    void   remove(const std::string& name)    { store_.erase(name); }
    size_t size(const std::string& name) const { return get(name).size(); }
    void   copy(const std::string& src, const std::string& dst) {
        store_[dst] = get(src);
    }

    // ── Initialisation ──────────────────────────────────────────────────
    void init_zeros(const std::string& n, size_t sz)
        { store_[n].assign(sz, 0.f); }
    void init_ones(const std::string& n, size_t sz)
        { store_[n].assign(sz, 1.f); }
    void init_value(const std::string& n, size_t sz, float v)
        { store_[n].assign(sz, v); }
    void init_neg_inf(const std::string& n, size_t sz)
        { store_[n].assign(sz, -std::numeric_limits<float>::infinity()); }
    void init_random(const std::string& n, size_t sz,
                     float lo=-1.f, float hi=1.f, uint32_t seed=42) {
        auto& b = store_[n]; b.resize(sz);
        uint32_t s = seed;
        for (auto& v : b) {
            s = s*1664525u + 1013904223u;
            v = lo + (hi-lo)*static_cast<float>(s>>8)/static_cast<float>(1<<24);
        }
    }
    void init_identity(const std::string& n, uint32_t N) {
        auto& b = store_[n]; b.assign((size_t)N*N, 0.f);
        for (uint32_t i=0;i<N;i++) b[i*N+i]=1.f;
    }

    // ── Tiling helpers ───────────────────────────────────────────────────
    void slice_rows(const std::string& src, const std::string& dst,
                    uint32_t row_start, uint32_t nrows, uint32_t full_cols) {
        const auto& s = get(src);
        auto& d = store_[dst];
        d.resize((size_t)nrows*full_cols);
        for (uint32_t r=0;r<nrows;r++)
            for (uint32_t c=0;c<full_cols;c++)
                d[r*full_cols+c] = s[(row_start+r)*full_cols+c];
    }
    void slice_cols(const std::string& src, const std::string& dst,
                    uint32_t col_start, uint32_t ncols,
                    uint32_t full_rows, uint32_t full_cols) {
        const auto& s = get(src);
        auto& d = store_[dst];
        d.resize((size_t)full_rows*ncols);
        for (uint32_t r=0;r<full_rows;r++)
            for (uint32_t c=0;c<ncols;c++)
                d[r*ncols+c] = s[r*full_cols+(col_start+c)];
    }
    void place_tile(const std::string& tile, const std::string& dst,
                    uint32_t row_start, uint32_t col_start,
                    uint32_t tile_rows, uint32_t tile_cols, uint32_t full_cols) {
        const auto& t = get(tile);
        auto& d = get_mutable(dst);
        for (uint32_t r=0;r<tile_rows;r++)
            for (uint32_t c=0;c<tile_cols;c++)
                d[(row_start+r)*full_cols+(col_start+c)] = t[r*tile_cols+c];
    }

    // ── Verification ────────────────────────────────────────────────────
    float max_abs_diff(const std::string& a, const std::string& b) const {
        const auto& va=get(a); const auto& vb=get(b);
        if (va.size()!=vb.size()) throw std::runtime_error("max_abs_diff: size mismatch");
        float mx=0;
        for (size_t i=0;i<va.size();i++) mx=std::max(mx,std::abs(va[i]-vb[i]));
        return mx;
    }

    // ── Debug print ──────────────────────────────────────────────────────
    void print(const std::string& name, uint32_t rows, uint32_t cols,
               uint32_t max_rows=8, std::ostream& os=std::cout) const {
        const auto& buf = get(name);
        os << "  TensorStore[\"" << name << "\"]  ["<<rows<<"x"<<cols<<"]:\n";
        uint32_t show = std::min(rows, max_rows);
        for (uint32_t r=0;r<show;r++) {
            os << "    row " << std::setw(3) << r << ": ";
            uint32_t sc = std::min(cols, 8u);
            for (uint32_t c=0;c<sc;c++)
                os << std::fixed << std::setprecision(4) << std::setw(10) << buf[r*cols+c];
            if (sc<cols) os << " ...";
            os << "\n";
        }
        if (show<rows) os << "    ... (" << rows-show << " more rows)\n";
    }

private:
    std::unordered_map<std::string, std::vector<float>> store_;
};

}  // namespace sim
