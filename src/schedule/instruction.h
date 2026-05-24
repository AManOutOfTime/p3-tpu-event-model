#pragma once
#include "core/types.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sim {

// Flexible parameter value. Kept simple on purpose -- no nested maps, no lists.
using ParamVal = std::variant<int64_t, double, std::string, bool>;
using ParamMap = std::unordered_map<std::string, ParamVal>;

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
