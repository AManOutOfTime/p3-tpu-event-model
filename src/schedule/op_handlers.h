#pragma once

#include "config/arch_config.h"
#include "schedule/op_registry.h"
#include <cstdint>
#include <string>

namespace sim {

// Register the simulator's built-in typed operations. These handlers translate
// open-ended schedule params into concrete unit payloads for DMA, access,
// systolic, and vector units.
void register_builtin_ops(OpRegistry& registry, const ArchConfig& arch);

uint32_t resolve_dim(const ParamMap& params, const std::string& key,
                     const ArchConfig& arch, uint32_t def = 0);
uint32_t dtype_bytes(const std::string& precision);

}  // namespace sim
