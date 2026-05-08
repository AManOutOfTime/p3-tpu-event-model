#include "core/logger.h"
#include <iomanip>

namespace sim {

void ConsoleLogger::operator()(const Event& e) const {
    Unit* u = engine_.get_unit(e.target);
    os_ << "[cycle " << std::setw(8) << e.cycle
        << " | " << std::fixed << std::setprecision(3)
        << cycles_to_ns(e.cycle, engine_.clock_ghz()) << " ns]  "
        << std::left << std::setw(12) << to_string(e.type)
        << " -> " << std::setw(18) << (u ? u->name() : "<none>")
        << (e.label.empty() ? "" : "  \"" + e.label + "\"")
        << "\n";
}

}  // namespace sim
