#include <doctest/doctest.h>
#include <stdexcept>
#include <vector>
#include "core/event_engine.h"

using namespace sim;

// Records every cycle it receives an event, in order.
namespace {
class Recorder : public Unit {
public:
    Recorder(std::string n, std::vector<Cycle>* out)
        : Unit(std::move(n)), out_(out) {}
    void handle(const Event& e, EventEngine&) override {
        if (out_) out_->push_back(e.cycle);
    }
private:
    std::vector<Cycle>* out_;
};
}  // namespace

TEST_CASE("events fire in cycle order regardless of insertion order") {
    EventEngine engine(1.0);
    std::vector<Cycle> fired;
    UnitId u = engine.register_unit(std::make_unique<Recorder>("r", &fired));

    for (Cycle c : std::vector<Cycle>{50, 10, 30, 20}) {
        Event e; e.cycle = c; e.target = u;
        engine.schedule(e);
    }
    engine.run();
    REQUIRE(fired == std::vector<Cycle>{10, 20, 30, 50});
}

TEST_CASE("same-cycle events are ordered by seq (insertion order)") {
    EventEngine engine;
    std::vector<EventId> seqs;

    class SeqRecorder : public Unit {
    public:
        explicit SeqRecorder(std::vector<EventId>* o) : Unit("sr"), out_(o) {}
        void handle(const Event& e, EventEngine&) override { out_->push_back(e.seq); }
        std::vector<EventId>* out_;
    };
    UnitId u = engine.register_unit(std::make_unique<SeqRecorder>(&seqs));

    Event a; a.cycle = 5; a.target = u;
    Event b; b.cycle = 5; b.target = u;
    Event c; c.cycle = 5; c.target = u;
    auto sa = engine.schedule(a);
    auto sb = engine.schedule(b);
    auto sc = engine.schedule(c);
    engine.run();
    REQUIRE(seqs == std::vector<EventId>{sa, sb, sc});
}

TEST_CASE("run(stop_at) only fires events up to the limit") {
    EventEngine engine;
    std::vector<Cycle> fired;
    UnitId u = engine.register_unit(std::make_unique<Recorder>("r", &fired));

    for (Cycle c : std::vector<Cycle>{1, 5, 10, 100}) {
        Event e; e.cycle = c; e.target = u; engine.schedule(e);
    }
    Cycle last = engine.run(10);
    REQUIRE(fired == std::vector<Cycle>{1, 5, 10});
    REQUIRE(last == 10);
    REQUIRE(engine.pending() == 1);

    engine.run();  // drain remainder
    REQUIRE(fired == std::vector<Cycle>{1, 5, 10, 100});
}

TEST_CASE("schedule_after fires relative to current cycle") {
    EventEngine engine;
    std::vector<Cycle> fired;
    UnitId rec = engine.register_unit(std::make_unique<Recorder>("rec", &fired));

    class Chainer : public Unit {
    public:
        Chainer(UnitId fwd, std::vector<Cycle>* o)
            : Unit("chain"), fwd_(fwd), out_(o) {}
        void handle(const Event& e, EventEngine& eng) override {
            out_->push_back(e.cycle);
            if (e.cycle == 10) {
                Event next; next.target = fwd_;
                eng.schedule_after(7, next);  // should fire at cycle 17
            }
        }
        UnitId fwd_; std::vector<Cycle>* out_;
    };
    UnitId ch = engine.register_unit(std::make_unique<Chainer>(rec, &fired));

    Event start; start.cycle = 10; start.target = ch;
    engine.schedule(start);
    engine.run();
    REQUIRE(fired == std::vector<Cycle>{10, 17});
}

TEST_CASE("scheduling an event in the past throws") {
    EventEngine engine;
    std::vector<Cycle> dummy;
    UnitId u = engine.register_unit(std::make_unique<Recorder>("r", &dummy));

    Event e; e.cycle = 100; e.target = u;
    engine.schedule(e);
    engine.run();  // now_ = 100

    Event past; past.cycle = 50; past.target = u;
    REQUIRE_THROWS_AS(engine.schedule(past), std::runtime_error);
}

TEST_CASE("trace callback fires once per dispatched event") {
    EventEngine engine;
    std::vector<Cycle> dummy;
    int count = 0;
    UnitId u = engine.register_unit(std::make_unique<Recorder>("r", &dummy));
    engine.set_trace([&](const Event&) { count++; });

    for (Cycle c : std::vector<Cycle>{1, 2, 3}) {
        Event e; e.cycle = c; e.target = u; engine.schedule(e);
    }
    engine.run();
    REQUIRE(count == 3);
}

TEST_CASE("events with no registered unit are silently skipped") {
    EventEngine engine;
    Event e; e.cycle = 5; e.target = INVALID_UNIT;
    engine.schedule(e);
    REQUIRE_NOTHROW(engine.run());
    REQUIRE(engine.current_cycle() == 5);
}
