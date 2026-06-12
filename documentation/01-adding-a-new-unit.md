# Guide 1 — Adding a New Hardware Unit

This guide walks through adding a brand-new hardware unit to the simulator, from
the C++ class all the way to running it inside a schedule and a unit test. We use
a running example: a **`ScalarUnit`** that models a small scalar/control core with
a fixed per-element throughput.

> **Mental model first.** A "unit" is a *latency model*, nothing more. When the
> engine fires an `OP_START` event at your unit, your job is to (a) compute how
> many cycles the operation takes and (b) schedule an `OP_DONE` event that many
> cycles later. When `OP_DONE` fires, you tell the scheduler the instruction is
> finished so its dependents can run. **No real math is performed** in this
> branch — see [03-simulator-engine.md](03-simulator-engine.md) for why.

The end-to-end checklist is:

1. Write `src/units/scalar_unit.h` / `.cpp` deriving from `Unit`.
2. Register the source file in `src/CMakeLists.txt`.
3. Instantiate + `register_unit()` it in `apps/sim_main.cpp`.
4. Add a `dynamic_cast` branch in `wire_units()` so it receives the `Scheduler*`.
5. (Usually) add an **op handler** that emits `OP_START` events at it — see
   [02-adding-a-new-operation.md](02-adding-a-new-operation.md).
6. Reference it from a schedule (YAML or programmatic) and/or a test.

---

## Step 0 — Understand the `Unit` contract

Every unit derives from `sim::Unit` ([src/core/unit.h](../src/core/unit.h)):

```cpp
class Unit {
public:
    explicit Unit(std::string name);
    UnitId             id()   const;     // assigned by the engine on registration
    const std::string& name() const;
    void set_verbose(bool v);            // gates per-event trace prints
    bool verbose() const;
    virtual void handle(const Event& e, EventEngine& engine) = 0;  // <-- you implement this
};
```

Key facts:

- **The engine assigns `id_`** when you call `register_unit()` — you never set it.
  `Unit` declares `friend class EventEngine` precisely so the engine can write it.
- **`handle()` is the only method you must implement.** It is called *once per
  event* targeting this unit. Inside `handle()`, `engine.current_cycle()` equals
  `event.cycle`.
- **`verbose_`** (default `true`) gates trace logging. `--no-trace` sets it to
  `false` on every unit. Because *formatting* the log lines is itself the cost on
  large runs, you must gate the work behind `if (verbose_)`, not just redirect the
  stream.
- A unit needs a pointer to the `Scheduler` so it can call `notify_done()`. The
  convention is a `set_scheduler(Scheduler*)` setter, wired up later in
  `sim_main.cpp`'s `wire_units()`.

The reference implementation to copy is **`DelayUnit`**
([src/units/delay_unit.h](../src/units/delay_unit.h),
[.cpp](../src/units/delay_unit.cpp)) — a fixed-latency stub that is the canonical
template for real units.

---

## Step 1 — Write the header

Create `src/units/scalar_unit.h`. Two things go here: a **payload struct** (the
typed data your op handler will attach to the event) and the **unit class**.

```cpp
#pragma once
#include "core/unit.h"
#include "config/arch_config.h"
#include <iostream>
#include <string>

namespace sim {

class Scheduler;

// ---------------------------------------------------------------------------
// ScalarOp — payload for scalar_core operations.
//
// The unit models latency only; buffer-name fields are symbolic (used in trace
// labels / dependency reasoning, never dereferenced as real memory).
// ---------------------------------------------------------------------------
struct ScalarOp {
    std::string kind;            // e.g. "bias_add", "clamp"
    uint64_t    elements = 0;    // drives the latency formula
    std::string src;
    std::string dst;
};

// ---------------------------------------------------------------------------
// ScalarUnit — a scalar/control core.
//
// TIMING:  latency = ceil(elements / throughput)
// EVENT PROTOCOL:
//   OP_START → decode ScalarOp → compute latency → schedule OP_DONE
//   OP_DONE  → scheduler.notify_done(instr)
// ---------------------------------------------------------------------------
class ScalarUnit : public Unit {
public:
    ScalarUnit(std::string name, uint32_t throughput,
               Scheduler* sched = nullptr, std::ostream& os = std::cout);

    void set_scheduler(Scheduler* s) { sched_ = s; }

    void  handle(const Event& e, EventEngine& engine) override;
    Cycle compute_latency(uint64_t elements) const;

private:
    uint32_t      throughput_;   // elements processed per cycle
    Scheduler*    sched_;
    std::ostream& os_;
};

}  // namespace sim
```

**Design notes that match the rest of the codebase:**

- The payload is a plain struct carried inside `Event::payload` (a `std::any`).
  Look at `GemmShape` ([systolic_unit.h](../src/units/systolic_unit.h)),
  `DmaTransfer` ([dma_unit.h](../src/units/dma_unit.h)),
  `VectorOp` ([vector_unit.h](../src/units/vector_unit.h)), and
  `AccessOp` ([access_unit.h](../src/units/access_unit.h)) for the established
  pattern.
- Take config by value or by a small typed sub-config (like `VectorCoreConfig`).
  `DmaUnit` holds `const ArchConfig&` because it needs many fields; prefer the
  narrow form unless you genuinely need the whole config.
- Provide a public `compute_latency()` so **tests can assert the formula directly**
  without running the engine.

---

## Step 2 — Write the implementation

Create `src/units/scalar_unit.cpp`. The `handle()` method is a two-branch state
machine on `EventType`:

```cpp
#include "units/scalar_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>

namespace sim {

ScalarUnit::ScalarUnit(std::string name, uint32_t throughput,
                       Scheduler* sched, std::ostream& os)
    : Unit(std::move(name)), throughput_(throughput), sched_(sched), os_(os) {}

Cycle ScalarUnit::compute_latency(uint64_t elements) const {
    if (!elements || !throughput_) return 0;
    return static_cast<Cycle>(std::ceil(
        static_cast<double>(elements) / throughput_));
}

void ScalarUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        Cycle lat = 0;
        if (const auto* op = std::any_cast<ScalarOp>(&e.payload)) {
            lat = compute_latency(op->elements);
            if (verbose_)
                os_ << "  [" << name() << "]  SCALAR_START"
                    << "  instr=" << e.instr << "  @cycle=" << e.cycle
                    << "  kind=" << op->kind << "  elems=" << op->elements
                    << "  lat=" << lat
                    << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";
        } else if (const auto* p = std::any_cast<int64_t>(&e.payload)) {
            // Backward-compat: lets `op: delay` target this unit too.
            lat = static_cast<Cycle>(*p);
        }

        // Schedule the completion event `lat` cycles in the future. Copy the
        // incoming event so instr/label/payload are preserved for OP_DONE.
        Event done  = e;
        done.type   = EventType::OP_DONE;
        done.cycle  = e.cycle + lat;
        done.seq    = engine.next_seq();   // fresh tie-breaker
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        if (verbose_)
            os_ << "  [" << name() << "]  SCALAR_DONE"
                << "  instr=" << e.instr << "  @cycle=" << e.cycle
                << (e.label.empty() ? "" : "  \"" + e.label + "\"") << "\n";

        // CRITICAL: unblock dependents. Without this the schedule stalls.
        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
```

### The five rules every `handle()` must obey

1. **Decode the payload with `std::any_cast<T>(&e.payload)`** (pointer form — it
   returns `nullptr` instead of throwing if the type doesn't match). Always handle
   the "wrong/empty payload" case gracefully.
2. **On `OP_START`: compute latency, then `schedule()` an `OP_DONE`** at
   `e.cycle + lat`. Copy the event first (`Event done = e;`) so `instr`, `label`,
   and `payload` carry through to `OP_DONE`.
3. **Assign a fresh `seq`** on the new event via `engine.next_seq()` so same-cycle
   ordering stays deterministic. (If you leave `seq == 0`, `engine.schedule()`
   assigns one for you — but doing it explicitly is the house style.)
4. **On `OP_DONE`: call `sched_->notify_done(e.instr)`.** This is what advances the
   DAG. Forgetting it is the #1 bug — the sim runs, your op "completes," but every
   downstream instruction hangs and the run ends with `outstanding > 0`.
5. **Never schedule an event in the past.** `engine.schedule()` throws if
   `e.cycle < now_`. Since you always add a non-negative latency to `e.cycle`,
   you're safe — just don't subtract.

### Where does the unit get reserved?

Note that `ScalarUnit::handle()` does **not** reserve itself. Reservation (picking
*which* physical unit in a pool and *when* it's free) happens earlier, in the **op
handler**, via `scheduler.reserve_unit_pool()`. The handler computes the same
latency, reserves the unit for that duration, and emits the `OP_START` at the
returned start cycle. The unit then re-derives the latency to schedule `OP_DONE`.

This is why the **latency formula must live in one place**. The systolic path
solves this elegantly: `systolic_gemm_latency()` is a *free function* in
[systolic_unit.h](../src/units/systolic_unit.h) called by **both** the `gemm` op
handler (for the reservation) and `SystolicUnit::handle()` (for `OP_DONE`), so the
reservation duration and the actual completion can never desync. For a simple unit
like `ScalarUnit`, putting `compute_latency()` on the class and calling it from the
handler achieves the same thing — just make sure both sides call the same function.

---

## Step 3 — Register the source file

Add the `.cpp` to `SIM_CORE_SOURCES` in [src/CMakeLists.txt](../src/CMakeLists.txt):

```cmake
set(SIM_CORE_SOURCES
    ...
    units/access_unit.cpp
    units/scalar_unit.cpp        # <-- add this line
)
```

Nothing else in CMake needs touching — the whole simulator is one static library
(`sim_core`) and both `sim_main` and `unit_tests` link it.

---

## Step 4 — Instantiate and register in `sim_main.cpp`

Two edits in [apps/sim_main.cpp](../apps/sim_main.cpp).

**(a) Register a pool of instances** next to the other pools (around the
`SystolicUnit` / `DmaUnit` / `VectorUnit` / `AccessUnit` registration block):

```cpp
// Scalar core pool
for (uint32_t i = 0; i < arch.scalar_cores; i++)
    engine.register_unit(std::make_unique<ScalarUnit>(
        "scalar_core_" + std::to_string(i), arch.scalar_throughput));
```

> **Naming convention matters.** The engine's `find_unit_pool("scalar_core")`
> matches every unit whose name starts with `"scalar_core_"`. So a *logical* pool
> name `scalar_core` (what schedules reference) maps to physical units
> `scalar_core_0`, `scalar_core_1`, … This is how `vector_cores: 3` in the config
> becomes three units that the scheduler load-balances across. If you only ever
> want one, register `scalar_core_0` and reference `scalar_core`.

If you added new config fields (`scalar_cores`, `scalar_throughput`), wire them
into `ArchConfig` ([arch_config.h](../src/config/arch_config.h)) and its YAML
parser/serializer in [arch_config.cpp](../src/config/arch_config.cpp). For a quick
experiment you can hardcode the count/throughput instead.

**(b) Add a `wire_units()` branch** so the unit receives the `Scheduler*`:

```cpp
static void wire_units(EventEngine& engine, Scheduler& sched) {
    for (UnitId uid = 0; uid < (UnitId)engine.num_units(); uid++) {
        Unit* u = engine.get_unit(uid);
        if (auto* x = dynamic_cast<DelayUnit*>   (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<SystolicUnit*>(u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<DmaUnit*>     (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<VectorUnit*>  (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<AccessUnit*>  (u)) { x->set_scheduler(&sched); continue; }
        if (auto* x = dynamic_cast<ScalarUnit*>  (u)) { x->set_scheduler(&sched); continue; }  // <-- add
    }
}
```

**Forgetting this branch is a silent failure**: `sched_` stays `nullptr`, the
`if (sched_)` guard in `handle()` skips `notify_done()`, and the schedule deadlocks
exactly as if you'd forgotten rule 4.

---

## Step 5 — Make it callable from a schedule

A unit does nothing until some **op handler** emits an `OP_START` at it. You have
two options:

### Option A — reuse the generic `delay` op (fastest smoke test)

The built-in `delay` op
([op_handlers.cpp](../src/schedule/op_handlers.cpp)) fires a single `OP_START` at
`inst.unit` with an `int64_t` latency payload. Because `ScalarUnit::handle()` has
an `int64_t` fallback branch, this works immediately:

```yaml
schedule:
  - id: 0
    op: delay
    unit: scalar_core          # resolves to scalar_core_0, scalar_core_1, ...
    params: { latency_cycles: 40 }
    label: "scalar smoke"
```

Run it:

```bash
./build/apps/sim_main --schedule schedules/scalar_smoke.yaml
```

### Option B — a typed op handler (the real integration)

To pass a `ScalarOp` payload (with `elements`, `kind`, etc.) you register a typed
handler. This is covered in full in
[02-adding-a-new-operation.md](02-adding-a-new-operation.md); the short version:

```cpp
reg.register_op("scalar_bias_add", [arch](const IssueCtx& ctx) {
    auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
    if (targets.empty()) throw std::runtime_error("scalar_bias_add: unknown unit");
    const auto& p = ctx.inst.params;

    ScalarOp op;
    op.kind     = "bias_add";
    op.elements = static_cast<uint64_t>(pget_int(p, "length", 0));
    op.src      = pget_str(p, "source");
    op.dst      = pget_str(p, "destination");

    // Reserve the earliest-free unit in the pool for the op's duration,
    // then emit OP_START at the returned start cycle.
    Cycle lat = static_cast<Cycle>(std::ceil(op.elements / 64.0));  // or call a shared fn
    auto res = ctx.scheduler.reserve_unit_pool(targets, lat);

    Event e;
    e.type    = EventType::OP_START;
    e.target  = res.id;
    e.cycle   = res.start;
    e.instr   = ctx.inst.id;
    e.label   = ctx.inst.label;
    e.payload = op;
    ctx.engine.schedule(std::move(e));
});
```

Register it inside `register_builtin_ops()`
([op_handlers.cpp](../src/schedule/op_handlers.cpp)) so it's available to
`sim_main`, then use `op: scalar_bias_add` in any schedule.

---

## Step 6 — Test it

Tests use **doctest** and live in [tests/](../tests). The pattern (see
[tests/test_dummy_units.cpp](../tests/test_dummy_units.cpp)) is: build a tiny
engine, register your unit, register the op, build a small `Schedule`, wire the
scheduler, `launch()`, `run()`, then assert on the final cycle and `all_done()`.

Create `tests/test_scalar_unit.cpp`:

```cpp
#include <doctest/doctest.h>
#include "core/event_engine.h"
#include "schedule/schedule.h"
#include "schedule/op_registry.h"
#include "schedule/scheduler.h"
#include "units/scalar_unit.h"
#include <sstream>

using namespace sim;

// 1. Pure latency-formula test — no engine needed.
TEST_CASE("ScalarUnit latency = ceil(elements / throughput)") {
    std::stringstream ss;
    ScalarUnit u("scalar_core_0", /*throughput=*/64, nullptr, ss);
    REQUIRE(u.compute_latency(0)   == 0);
    REQUIRE(u.compute_latency(64)  == 1);
    REQUIRE(u.compute_latency(65)  == 2);   // ceil
    REQUIRE(u.compute_latency(128) == 2);
}

// 2. End-to-end through the engine via the generic `delay` op.
TEST_CASE("ScalarUnit completes a scheduled op and unblocks dependents") {
    std::stringstream ss;
    EventEngine engine;
    engine.register_unit(std::make_unique<ScalarUnit>("scalar_core_0", 64, nullptr, ss));

    OpRegistry reg;
    reg.register_op("delay", [](const IssueCtx& ctx) {
        auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
        REQUIRE(!targets.empty());
        Cycle lat = static_cast<Cycle>(pget_int(ctx.inst.params, "latency_cycles", 10));
        auto res = ctx.scheduler.reserve_unit_pool(targets, lat);
        Event e;
        e.type = EventType::OP_START; e.target = res.id; e.cycle = res.start;
        e.instr = ctx.inst.id; e.payload = static_cast<int64_t>(lat);
        ctx.engine.schedule(std::move(e));
    });

    const char* yaml = R"(
schedule:
  - id: 0
    op: delay
    unit: scalar_core
    params: { latency_cycles: 40 }
  - id: 1
    op: delay
    unit: scalar_core
    params: { latency_cycles: 10 }
    depends_on: [0]
)";
    Schedule s = Schedule::from_yaml_string(yaml);
    Scheduler sched(engine, reg, s);

    // Wire the scheduler into the unit (mirrors sim_main's wire_units()).
    for (UnitId id = 0; id < (UnitId)engine.num_units(); id++)
        if (auto* su = dynamic_cast<ScalarUnit*>(engine.get_unit(id)))
            su->set_scheduler(&sched);

    sched.launch();
    Cycle final_cycle = engine.run();
    REQUIRE(sched.all_done());
    REQUIRE(final_cycle == 50);   // 40 then 10, serialized by depends_on
}
```

Register the file in [tests/CMakeLists.txt](../tests/CMakeLists.txt):

```cmake
set(SIM_TEST_SOURCES
    doctest_main.cpp
    ...
    test_llama_schedule.cpp
    test_scalar_unit.cpp        # <-- add
)
```

Build and run:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
# Or just your case:
./build/tests/unit_tests -tc="ScalarUnit latency = ceil(elements / throughput)"
```

> Tip: tests can load real YAML from the repo because `SIM_PROJECT_ROOT` is baked
> into the test binary and `ctest` runs with the project root as the working dir
> (see [tests/CMakeLists.txt](../tests/CMakeLists.txt)).

---

## Common pitfalls (read before you debug)

| Symptom | Cause | Fix |
|---|---|---|
| Run ends with `outstanding=N` (>0), schedule "hangs" | `handle()` never calls `notify_done()` on `OP_DONE` | Add rule 4; check `sched_` is non-null |
| Schedule hangs even though `handle()` looks right | `wire_units()` has no branch for your unit, so `sched_ == nullptr` | Add the `dynamic_cast` branch in Step 4 |
| `reserve_unit_pool: empty unit pool` thrown | `inst.unit` name doesn't match any registered prefix | Check the logical name vs the `name_` you registered (`"scalar_core"` ↔ `"scalar_core_0"`) |
| Latency in the trace ≠ what `OP_DONE` lands on | Reservation duration and `handle()` latency computed differently | Use one shared latency function for both |
| `engine.schedule: event ... is in the past` thrown | You scheduled an `OP_DONE` at a cycle < now | Never subtract from `e.cycle`; always `e.cycle + lat` |
| Wrong/garbage payload values | `std::any_cast<T>` to the wrong type silently returns `nullptr` | Match the cast type to exactly what the handler attached |
| Trace floods / slow on big runs | Printing not gated behind `verbose_` | Wrap *all* logging in `if (verbose_)` |

---

## Summary

Adding a unit is mechanical once you internalize the contract: **a unit is a
latency model that turns `OP_START` into a future `OP_DONE`, then calls
`notify_done()` so the DAG advances.** The four structural touch-points are
(1) the class, (2) `src/CMakeLists.txt`, (3) registration + (4) `wire_units()` in
`sim_main.cpp`. To make it *do* anything you also need an op handler
([Guide 2](02-adding-a-new-operation.md)). The whole flow — events, the
scheduler, the DAG, and the performance/memory work behind it — is explained in
[Guide 3](03-simulator-engine.md).
