#pragma once
#include "core/types.h"
#include <initializer_list>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sim {

// Flexible parameter value. Kept simple on purpose -- no nested maps, no lists.
using ParamVal = std::variant<int64_t, double, std::string, bool>;

// ---------------------------------------------------------------------------
// ParamMap -- small flat (vector-backed) string->ParamVal map.
//
// Instructions carry only a handful of params (3-8), and large schedules
// create millions of Instructions. A std::unordered_map per instruction means
// a bucket array plus one heap node per entry -- millions of tiny allocations
// that dominated both schedule construction and teardown. A flat vector does
// one allocation per map; linear lookup over <=8 entries is faster than hashing
// at this size. Interface mirrors the subset of unordered_map the code uses
// (operator[], find/end, count, begin/end, initializer-list construction), so
// this is a drop-in replacement with identical semantics -- timing/accuracy
// are unaffected (it is purely a container swap).
// ---------------------------------------------------------------------------
class ParamMap {
public:
    using value_type     = std::pair<std::string, ParamVal>;
    using storage        = std::vector<value_type>;
    using iterator       = storage::iterator;
    using const_iterator = storage::const_iterator;

    ParamMap() = default;
    ParamMap(std::initializer_list<value_type> init) : data_(init) {}

    iterator       begin()       { return data_.begin(); }
    iterator       end()         { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end()   const { return data_.end(); }

    iterator find(const std::string& k) {
        for (auto it = data_.begin(); it != data_.end(); ++it)
            if (it->first == k) return it;
        return data_.end();
    }
    const_iterator find(const std::string& k) const {
        for (auto it = data_.begin(); it != data_.end(); ++it)
            if (it->first == k) return it;
        return data_.end();
    }

    ParamVal& operator[](const std::string& k) {
        for (auto& kv : data_)
            if (kv.first == k) return kv.second;
        data_.emplace_back(k, ParamVal{});
        return data_.back().second;
    }

    std::size_t count(const std::string& k) const { return find(k) == end() ? 0u : 1u; }
    std::size_t size()  const { return data_.size(); }
    bool        empty() const { return data_.empty(); }

private:
    storage data_;
};

// Helper accessors. Return `def` if key is missing or stored as a wrong type.
inline int64_t pget_int(const ParamMap& p, const std::string& k, int64_t def = 0) {
    auto it = p.find(k);
    if (it == p.end()) return def;
    if (auto* v = std::get_if<int64_t>(&it->second)) return *v;
    if (auto* v = std::get_if<double>(&it->second))  return static_cast<int64_t>(*v);
    return def;
}
inline double pget_dbl(const ParamMap& p, const std::string& k, double def = 0.0) {
    auto it = p.find(k);
    if (it == p.end()) return def;
    if (auto* v = std::get_if<double>(&it->second))  return *v;
    if (auto* v = std::get_if<int64_t>(&it->second)) return static_cast<double>(*v);
    return def;
}
inline std::string pget_str(const ParamMap& p, const std::string& k,
                            const std::string& def = "") {
    auto it = p.find(k);
    if (it == p.end()) return def;
    if (auto* v = std::get_if<std::string>(&it->second)) return *v;
    return def;
}
inline bool pget_bool(const ParamMap& p, const std::string& k, bool def = false) {
    auto it = p.find(k);
    if (it == p.end()) return def;
    if (auto* v = std::get_if<bool>(&it->second)) return *v;
    return def;
}

// One entry in a schedule file. The `op` string is looked up in an OpRegistry,
// and `params` + `unit` are passed straight through to the handler -- so the
// schema is open: any op can invent its own parameter names. This is what
// makes the schedule format flexible across granularities:
//
//   - fine-grained:     op: dma_load,    params: {bytes:1024, src:hbm}
//   - middle:           op: gemm,        params: {M:128,K:128,N:128}
//   - coarse-grained:   op: flash_attn2, params: {seq:2048,head:128,heads:32}
//
// All three coexist in the same schedule; only the registered handler differs.
struct Instruction {
    InstructionId              id = 0;
    std::string                op;          // registry key
    std::string                unit;        // target unit name ("" for composite)
    ParamMap                   params;
    std::vector<InstructionId> depends_on;
    std::string                label;       // human description (optional)
};

}  // namespace sim
