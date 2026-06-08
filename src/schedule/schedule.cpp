#include "schedule/schedule.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <algorithm>

namespace sim {

namespace {

ParamVal node_to_param(const YAML::Node& n) {
    if (!n.IsScalar()) return std::string{};
    const std::string& s = n.Scalar();
    if (s == "true")  return true;
    if (s == "false") return false;
    try { size_t p; int64_t i = std::stoll(s, &p); if (p == s.size()) return i; } catch (...) {}
    try { size_t p; double  d = std::stod(s,  &p); if (p == s.size()) return d; } catch (...) {}
    return s;
}

Schedule from_node(const YAML::Node& root) {
    auto seq = root["schedule"];
    if (!seq || !seq.IsSequence())
        throw std::runtime_error("Schedule YAML must have a top-level 'schedule' sequence");

    Schedule sched;
    sched.instructions.reserve(seq.size());
    InstructionId auto_id = 0;

    for (const auto& item : seq) {
        Instruction inst;
        inst.id   = item["id"] ? item["id"].as<InstructionId>() : auto_id;
        auto_id   = inst.id + 1;

        if (!item["op"])
            throw std::runtime_error("Instruction missing required 'op' field");
        // op may be a scalar string OR a YAML flow-sequence like [placeholder].
        // In both cases we want the string value of the first (or only) element.
        if (item["op"].IsSequence()) {
            if (item["op"].size() == 0)
                throw std::runtime_error("Instruction 'op' sequence is empty");
            inst.op = item["op"][0].as<std::string>();
        } else {
            inst.op = item["op"].as<std::string>();
        }

        if (item["unit"])  inst.unit  = item["unit"].as<std::string>();
        if (item["label"]) inst.label = item["label"].as<std::string>();

        if (item["params"])
            for (auto kv : item["params"])
                inst.params[kv.first.as<std::string>()] = node_to_param(kv.second);

        if (item["depends_on"])
            for (auto d : item["depends_on"])
                inst.depends_on.push_back(d.as<InstructionId>());

        sched.instructions.push_back(std::move(inst));
    }
    sched.validate();
    return sched;
}

}  // namespace

// ---------------------------------------------------------------------------
// validate() — flat-vector rewrite
//
// The original used three unordered hash containers (set + two maps) which
// at 11M instructions cost ~1.16 GB and hundreds of seconds to build and
// destroy. IDs are sequential integers, so a flat vector indexed by ID
// replaces hashing entirely: one subtraction per access, contiguous layout,
// no per-node heap allocation.
//
// Correctness is unchanged: duplicate-ID check, unknown-dep check, and
// Kahn's cycle detection all produce identical errors to before.
// ---------------------------------------------------------------------------
void Schedule::validate() const {
    if (instructions.empty()) return;

    // Find ID range. In practice IDs are 0..N-1 (builder assigns sequentially)
    // but we handle YAML files that start at any offset.
    InstructionId min_id = instructions[0].id;
    InstructionId max_id = instructions[0].id;
    for (const auto& i : instructions) {
        if (i.id < min_id) min_id = i.id;
        if (i.id > max_id) max_id = i.id;
    }
    const size_t base     = min_id;
    const size_t id_range = static_cast<size_t>(max_id - min_id) + 1;

    // --- 1. Duplicate-ID check (replaces unordered_set) -------------------
    // vector<uint8_t>: 1 byte per possible ID, 0 = unseen, 1 = seen.
    std::vector<uint8_t> seen(id_range, 0u);
    for (const auto& i : instructions) {
        uint8_t& flag = seen[i.id - base];
        if (flag)
            throw std::runtime_error("Duplicate instruction id: " + std::to_string(i.id));
        flag = 1;
    }

    // --- 2. Unknown-dep check ---------------------------------------------
    for (const auto& i : instructions)
        for (auto d : i.depends_on) {
            const size_t idx = static_cast<size_t>(d) - base;
            if (idx >= id_range || !seen[idx])
                throw std::runtime_error("Instruction " + std::to_string(i.id) +
                    " depends on unknown id " + std::to_string(d));
        }

    // --- 3. Kahn's cycle detection (replaces two unordered_maps) ----------
    // indeg: vector<int> indexed by (id - base)
    // succ:  vector<vector<InstructionId>> indexed by (id - base)
    std::vector<int>                        indeg(id_range, 0);
    std::vector<std::vector<InstructionId>> succ(id_range);

    for (const auto& i : instructions)
        for (auto d : i.depends_on) {
            succ[d - base].push_back(i.id);
            indeg[i.id - base]++;
        }

    std::vector<InstructionId> ready;
    ready.reserve(256);
    for (const auto& i : instructions)
        if (indeg[i.id - base] == 0)
            ready.push_back(i.id);

    size_t visited = 0;
    while (!ready.empty()) {
        InstructionId id = ready.back(); ready.pop_back();
        visited++;
        for (auto s : succ[id - base])
            if (--indeg[s - base] == 0)
                ready.push_back(s);
    }

    if (visited != instructions.size())
        throw std::runtime_error("Schedule contains a dependency cycle");
}

Schedule Schedule::from_yaml_file(const std::string& path)   { return from_node(YAML::LoadFile(path)); }
Schedule Schedule::from_yaml_string(const std::string& yaml) { return from_node(YAML::Load(yaml)); }

}  // namespace sim