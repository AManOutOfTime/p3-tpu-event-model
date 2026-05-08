#pragma once
#include <cstdint>
#include <limits>

namespace sim {

// Time in this simulator is measured in *cycles*. The clock_ghz parameter on
// the engine is only used to convert cycles -> wall-clock ns for reporting.
using Cycle          = uint64_t;
using UnitId         = uint32_t;
using EventId        = uint64_t;
using InstructionId  = uint32_t;

inline constexpr UnitId  INVALID_UNIT  = std::numeric_limits<UnitId>::max();
inline constexpr Cycle   CYCLE_MAX     = std::numeric_limits<Cycle>::max();

// Cycles -> ns. clock_ghz = cycles per ns, so ns = cycles / clock_ghz.
inline double cycles_to_ns(Cycle c, double clock_ghz) {
    return static_cast<double>(c) / clock_ghz;
}

}  // namespace sim
