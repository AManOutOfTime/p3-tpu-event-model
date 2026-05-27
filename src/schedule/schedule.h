#pragma once
#include "schedule/instruction.h"
#include <string>
#include <vector>

namespace sim {

// Optional metadata block at the top of a schedule YAML.
// Used to carry tile dimensions for buffer seeding.
struct ScheduleMetadata {
    std::string type;       // e.g. "fa2_full_matrix"
    int Nq = 0, Nkv = 0;   // number of Q / KV tiles
    int Br = 0, Bc = 0;    // tile dimensions (rows / cols)
    int d_head = 0;         // head dimension

    bool is_fa2_full_matrix() const { return type == "fa2_full_matrix"; }
};

struct Schedule {
    ScheduleMetadata         metadata;
    std::vector<Instruction> instructions;

    // Throws std::runtime_error if:
    //   - two instructions share an id
    //   - depends_on references an unknown id
    //   - the dependency graph has a cycle
    void validate() const;

    static Schedule from_yaml_file(const std::string& path);
    static Schedule from_yaml_string(const std::string& yaml);
};

}  // namespace sim
