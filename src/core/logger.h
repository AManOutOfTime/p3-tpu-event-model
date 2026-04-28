#pragma once
#include "core/event_engine.h"
#include <iostream>
#include <ostream>

namespace sim {

// Console trace callback. Attach to the engine with:
//   ConsoleLogger logger(engine);
//   engine.set_trace([&](const Event& e){ logger(e); });
//
// Output format per event:
//   [cycle NNNNNNNN | TTTT.TTT ns]  TYPE_STR     -> unit_name   "label"
class ConsoleLogger {
public:
    explicit ConsoleLogger(const EventEngine& engine, std::ostream& os = std::cout)
        : engine_(engine), os_(os) {}

    void operator()(const Event& e) const;

private:
    const EventEngine& engine_;
    std::ostream&      os_;
};

}  // namespace sim
