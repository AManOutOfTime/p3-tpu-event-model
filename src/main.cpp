#include "engine/engine.h"
#include "schedule/dummy_schedule.h"

#include <cstdio>

int main() {
    tpu::EventEngine eng;

    // Phase 0 handler: just print a one-line trace per event. Format is
    // fixed-width so the output stays readable when a later phase starts
    // dumping thousands of events.
    eng.set_handler([](const tpu::Event& e) {
        std::printf("[t=%10lld ps] unit=%3u  %-13.*s  %s\n",
                    static_cast<long long>(e.time),
                    static_cast<unsigned>(e.unit),
                    static_cast<int>(tpu::to_string(e.kind).size()),
                    tpu::to_string(e.kind).data(),
                    e.label.c_str());
    });

    tpu::load_dummy_schedule(eng);

    std::printf("-- running dummy schedule (%zu events queued) --\n", eng.pending());
    const auto n = eng.run();
    std::printf("-- done: %zu events dispatched, final sim time = %lld ps --\n",
                n, static_cast<long long>(eng.now()));
    return 0;
}
