// Minimal hand-rolled tests — no gtest dependency in Phase 0.
// Each check aborts on failure with a clear message; CTest reports pass/fail
// on exit code.

#include "engine/engine.h"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace tpu;

static void test_orders_by_time() {
    EventEngine eng;
    std::vector<Tick> seen;
    eng.set_handler([&](const Event& e){ seen.push_back(e.time); });

    eng.schedule(300, 0, EventKind::Tick);
    eng.schedule(100, 0, EventKind::Tick);
    eng.schedule(200, 0, EventKind::Tick);
    eng.run();

    assert(seen.size() == 3);
    assert(seen[0] == 100 && seen[1] == 200 && seen[2] == 300);
}

static void test_tiebreak_unit_then_seq() {
    EventEngine eng;
    std::vector<std::pair<UnitId, Seq>> seen;
    eng.set_handler([&](const Event& e){ seen.push_back({e.unit, e.seq}); });

    // Same time, different units -> lower unit id first.
    eng.schedule(100, 5, EventKind::Tick);   // seq 0
    eng.schedule(100, 2, EventKind::Tick);   // seq 1
    // Same time, same unit -> lower seq first (i.e. insertion order).
    eng.schedule(100, 2, EventKind::Tick);   // seq 2
    eng.run();

    assert(seen.size() == 3);
    assert(seen[0].first == 2 && seen[0].second == 1);
    assert(seen[1].first == 2 && seen[1].second == 2);
    assert(seen[2].first == 5 && seen[2].second == 0);
}

static void test_now_advances_monotonically() {
    EventEngine eng;
    Tick last = -1;
    bool ok = true;
    eng.set_handler([&](const Event&){
        if (eng.now() < last) ok = false;
        last = eng.now();
    });
    eng.schedule(10, 0, EventKind::Tick);
    eng.schedule(50, 0, EventKind::Tick);
    eng.schedule(20, 0, EventKind::Tick);
    eng.run();
    assert(ok);
    assert(eng.now() == 50);
}

static void test_rejects_past_events() {
    EventEngine eng;
    eng.set_handler([](const Event&){});
    eng.schedule(100, 0, EventKind::Tick);
    eng.run_until(100);  // now_ == 100
    bool threw = false;
    try { eng.schedule(50, 0, EventKind::Tick); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

static void test_run_until_leaves_tail() {
    EventEngine eng;
    std::size_t count = 0;
    eng.set_handler([&](const Event&){ ++count; });
    eng.schedule(10, 0, EventKind::Tick);
    eng.schedule(20, 0, EventKind::Tick);
    eng.schedule(30, 0, EventKind::Tick);
    const auto dispatched = eng.run_until(20);
    assert(dispatched == 2);
    (void)dispatched;
    assert(count == 2);
    assert(eng.pending() == 1);
    assert(eng.now() == 20);
}

int main() {
    test_orders_by_time();
    test_tiebreak_unit_then_seq();
    test_now_advances_monotonically();
    test_rejects_past_events();
    test_run_until_leaves_tail();
    std::printf("all engine tests passed\n");
    return 0;
}
