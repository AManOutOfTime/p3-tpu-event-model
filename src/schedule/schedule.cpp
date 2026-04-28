#include "schedule/schedule.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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
        inst.op   = item["op"].as<std::string>();

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

void Schedule::validate() const {
    std::unordered_set<InstructionId> ids;
    for (const auto& i : instructions) {
        if (!ids.insert(i.id).second)
            throw std::runtime_error("Duplicate instruction id: " + std::to_string(i.id));
    }
    for (const auto& i : instructions)
        for (auto d : i.depends_on)
            if (!ids.count(d))
                throw std::runtime_error("Instruction " + std::to_string(i.id) +
                    " depends on unknown id " + std::to_string(d));

    // Kahn's algorithm to detect cycles.
    std::unordered_map<InstructionId, int> indeg;
    std::unordered_map<InstructionId, std::vector<InstructionId>> succ;
    for (const auto& i : instructions) {
        indeg[i.id];
        for (auto d : i.depends_on) { succ[d].push_back(i.id); indeg[i.id]++; }
    }
    std::vector<InstructionId> ready;
    for (auto& [id, deg] : indeg) if (deg == 0) ready.push_back(id);
    size_t visited = 0;
    while (!ready.empty()) {
        auto id = ready.back(); ready.pop_back(); visited++;
        for (auto s : succ[id]) if (--indeg[s] == 0) ready.push_back(s);
    }
    if (visited != instructions.size())
        throw std::runtime_error("Schedule contains a dependency cycle");
}

Schedule Schedule::from_yaml_file(const std::string& path)   { return from_node(YAML::LoadFile(path)); }
Schedule Schedule::from_yaml_string(const std::string& yaml) { return from_node(YAML::Load(yaml)); }

}  // namespace sim
