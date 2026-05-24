#pragma once
#include "schedule/instruction.h"
#include <string>
#include <vector>

namespace sim {

struct Schedule {
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
