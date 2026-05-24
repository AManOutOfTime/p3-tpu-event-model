#include "units/printing_unit.h"

namespace sim {

void PrintingUnit::handle(const Event& e, EventEngine& /*engine*/) {
    os_ << "  [" << name() << "]  "
        << to_string(e.type)
        << "  @cycle=" << e.cycle
        << "  instr=" << e.instr
        << (e.label.empty() ? "" : "  (\"" + e.label + "\")")
        << "\n";
}

}  // namespace sim
