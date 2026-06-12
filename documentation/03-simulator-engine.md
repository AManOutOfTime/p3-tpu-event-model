# Guide 3 — How the Simulator Engine Works

This is the deep-dive: how an architecture config + a schedule become a cycle
count and a metrics block. It covers the event engine, the units, how the
scheduler turns an instruction DAG into events, how `llama_schedule.cpp` and the
tiler generate those instructions, and — importantly — **the performance and
memory engineering** that took the full LLaMA-3-8B run from 10+ minutes / ~9 GB
down to under ~4 minutes / ~7 GB.

Read [Guide 1](01-adding-a-new-unit.md) and [Guide 2](02-adding-a-new-operation.md)
first if you haven't; this guide assumes you know what a unit and an op are.

---

## 1. The big picture

```
  YAML arch config ──▶ ArchConfig                 (configs/*.yaml)
  YAML/workload/llama ─▶ Schedule (instructions)  (build + tile = "pre-processing")
                          │
                          ▼
         ┌──────────────────────────────────────┐
         │  Scheduler (owns the DAG)             │
         │  - remaining_deps_ / successors_      │
         │  - launch(): issue ready instrs       │
         │  - notify_done(): decrement & issue   │
         └───────────────┬──────────────────────┘
                         │ try_issue → OpRegistry handler
                         ▼
         ┌──────────────────────────────────────┐
         │  EventEngine (owns the clock)         │
         │  - min-heap of Events by (cycle,seq)  │
         │  - run(): pop, advance now_, dispatch │
         │  - reserve_unit_pool(): hardware time │
         └───────────────┬──────────────────────┘
                         │ event.target
                         ▼
         ┌──────────────────────────────────────┐
         │  Units (latency models)              │
         │  OP_START → schedule OP_DONE          │
         │  OP_DONE  → scheduler.notify_done()   │
         └──────────────────────────────────────┘
```

Two cooperating state machines:

- The **EventEngine** owns *time*. It never loops over cycles; it jumps to the
  next event.
- The **Scheduler** owns the *dependency graph*. It decides *which* instructions
  are allowed to start, and hands each one to an op handler that injects events.

The cycle never goes backward, and an instruction's events are only ever created
once all its dependencies have completed — so the two machines together produce a
correct, deterministic cycle-level timing of the whole schedule.

---

## 2. Events

An `Event` ([src/core/event.h](../src/core/event.h)) is the atom of simulation:

```cpp
struct Event {
    Cycle         cycle  = 0;     // when it fires
    EventId       seq    = 0;     // engine-assigned tie-breaker
    EventType     type   = CUSTOM;// OP_START / OP_DONE / DMA_DONE / ...
    UnitId        target = INVALID_UNIT;  // which unit handles it
    InstructionId instr  = 0;     // which schedule instruction it belongs to
    std::string   label;          // human-readable (trace only)
    std::any      payload;        // typed op data (GemmShape, VectorOp, ...)
};
```

`payload` being `std::any` is what lets one generic engine carry wildly different
op data — a `GemmShape`, a `DmaTransfer`, an `int64_t` latency — without the engine
knowing any of those types. The unit `any_cast`s it back to the concrete type it
expects.

Two event types matter for the basic protocol:

- **`OP_START`** — "begin this op on this unit at this cycle." Emitted by op
  handlers.
- **`OP_DONE`** — "this op finished." Emitted by the unit `lat` cycles after its
  `OP_START`; its handler calls `notify_done()`.

`DMA_DONE`, `BUFFER_SWAP`, `BARRIER`, and `CUSTOM` exist for richer modeling but
the current units only use `OP_START`/`OP_DONE` plus payload-typed `CUSTOM`.

---

## 3. The min-heap and deterministic ordering

The engine stores pending events in a **binary min-heap**
(`std::priority_queue<Event, vector<Event>, std::greater<Event>>` in
[event_engine.h](../src/core/event_engine.h)). The ordering is defined by
`Event::operator>`:

```cpp
bool operator>(const Event& o) const {
    if (cycle != o.cycle) return cycle > o.cycle;  // earlier cycle = higher priority
    return seq > o.seq;                            // tie: earlier insertion wins
}
```

So events come out **earliest-cycle-first**, and ties at the same cycle break by
**insertion order** (`seq`, a monotonically increasing counter assigned by the
engine). This second rule is what makes runs **bit-reproducible across machines**:
without it, two events at the same cycle could fire in heap-arbitrary order and a
sweep on one host wouldn't match another.

Why a heap and not a sorted list or a calendar queue? A heap gives `O(log n)` push
and `O(log n)` pop with a single contiguous `vector` backing it (cache-friendly,
one allocation that grows geometrically). For this workload — where the number of
*simultaneously pending* events is far smaller than the total instruction count —
it's the right structure.

### The run loop — event-driven, not cycle-driven

The entire simulation is this loop ([event_engine.cpp](../src/core/event_engine.cpp)):

```cpp
Cycle EventEngine::run(Cycle stop_at) {
    while (!queue_.empty()) {
        if (queue_.top().cycle > stop_at) break;
        Event e = queue_.top();
        queue_.pop();
        now_ = e.cycle;            // TIME JUMPS to the event — no per-cycle tick
        if (trace_) trace_(e);     // optional logging hook
        Unit* u = get_unit(e.target);
        if (u) u->handle(e, *this);// the unit may schedule more events
    }
    return now_;
}
```

This is the heart of "event-driven cycle-level" simulation: if nothing happens
between cycle 50 and cycle 5000, the clock **jumps** straight from 50 to 5000 in
one iteration. Idle gaps cost zero wall-time. A naive per-cycle simulator would
tick 4950 empty cycles. This is the same model used by gem5 and most
architectural simulators.

`schedule()` enforces the arrow of time: it throws if you try to enqueue an event
with `cycle < now_`. It also auto-assigns `seq` if you left it 0.

---

## 4. Units as latency models

A unit's `handle()` is invoked by the run loop when an event targets it, with the
guarantee that `engine.current_cycle() == event.cycle`. The universal pattern (see
[Guide 1](01-adding-a-new-unit.md)) is:

```
OP_START:  lat = compute_latency(payload);
           schedule OP_DONE at cycle + lat
OP_DONE:   scheduler.notify_done(instr)
```

The concrete latency models, all timing-only (no float math):

| Unit | File | Latency |
|---|---|---|
| `SystolicUnit` | [systolic_unit.cpp](../src/units/systolic_unit.cpp) | weight-stationary GEMM model via shared `systolic_gemm_latency()`; acquires/releases SRAM working set across `[OP_START, OP_DONE]` |
| `DmaUnit` | [dma_unit.cpp](../src/units/dma_unit.cpp) | off-chip: `hbm.latency + ceil(bytes/(bw·channels))`; on-chip stage: `ceil(elements/rows)` |
| `VectorUnit` | [vector_unit.cpp](../src/units/vector_unit.cpp) | `passes·groups + exp_ops·exp_latency·groups`, `groups = ceil(elements/simd_width)` |
| `AccessUnit` | [access_unit.cpp](../src/units/access_unit.cpp) | `ceil(elements / bandwidth)` |
| `DelayUnit` | [delay_unit.cpp](../src/units/delay_unit.cpp) | fixed or payload-supplied latency (reference stub) |

The systolic latency function deserves a note because it encodes the "no free
speedup" physics that the array-size sweep needs. Weight-stationary:

```
tiles_k = ceil(K / rows),  tiles_n = ceil(N / cols),  wload = weight_load_cycles or rows
double-buffered:  per_n = wload + tiles_k * max(wload, M) + fill
serial:           per_n = tiles_k * (wload + M)          + fill
lat = tiles_n * per_n
```

A bigger array shrinks `tiles_k`/`tiles_n` but each weight load still costs ~`rows`
cycles, so doubling the array does **not** double throughput — reproducing the
TPU-v1 finding. The function lives in
[systolic_unit.h](../src/units/systolic_unit.h)/`.cpp` as a *free function* shared
by both the `gemm` op handler (reservation) and `SystolicUnit::handle()`
(`OP_DONE`), so the two can never disagree on the latency.

### Hardware reservation — where contention is modeled

Units don't model their own occupancy; the **engine's resource table** does. Each
registered unit has a `HardwareState { available_at, buffer_used/capacity,
busy_cycles }`. `reserve_unit_pool(ids, duration, buffer_bytes)`
([event_engine.cpp](../src/core/event_engine.cpp)):

1. Skips units whose buffer can't fit `buffer_bytes`.
2. Among the rest, picks **least buffer used, then earliest `max(now, available_at)`,
   then lowest id** (a stable deterministic tiebreak).
3. Marks the chosen unit busy until `start + duration`, adds `buffer_bytes`, and
   accumulates `busy_cycles += duration` (the raw material for the P0.2 utilization
   metric).

This is how `vector_cores: 3` becomes real parallelism: three independent
`available_at` clocks, and the scheduler hands each ready vector op to whichever
core frees up first. It's also how a *single* systolic array serializes its GEMMs
even though the scheduler issues them far ahead — they all contend for the same
`available_at`.

---

## 5. Instructions, the DAG, and the Scheduler

### What a schedule is

A `Schedule` ([schedule.h](../src/schedule/schedule.h)) is just
`vector<Instruction>`. An `Instruction`
([instruction.h](../src/schedule/instruction.h)):

```cpp
struct Instruction {
    InstructionId id;
    OpStr         op;      // interned 1-byte op name
    UnitStr       unit;    // interned 1-byte unit name
    LabelStr      label;   // interned 2-byte label (empty in --no-trace)
    ParamMap      params;
    std::vector<InstructionId> depends_on;
};
```

The `depends_on` edges form a **DAG** (directed acyclic graph). An instruction may
run only after every instruction it depends on has completed. There is no implicit
ordering beyond these edges — independent instructions run concurrently, limited
only by hardware reservation.

### Validation — Kahn's algorithm

`Schedule::validate()` ([schedule.cpp](../src/schedule/schedule.cpp)) rejects three
errors before simulation: duplicate ids, dependencies on unknown ids, and **cycles**
in the graph. Cycle detection is **Kahn's algorithm**:

1. Compute in-degree (number of unmet deps) for every node.
2. Seed a worklist with all in-degree-0 nodes.
3. Repeatedly pop a node, "remove" it, and decrement each successor's in-degree;
   when a successor hits 0, push it.
4. If the number of nodes visited ≠ total nodes, some nodes never reached in-degree
   0 → they're part of a cycle → throw.

Kahn's is `O(V + E)` and, conveniently, the *same* topological-processing logic the
Scheduler uses at runtime — except the Scheduler advances nodes as hardware
completes them rather than instantly.

> Programmatic LLaMA schedules **skip** `validate()` (`finish()` in
> [llama_schedule.cpp](../src/schedule/llama_schedule.cpp)): the builder assigns
> sequential ids and wires deps correctly by construction, and the Scheduler
> constructor rebuilds the equivalent structures anyway, so validating an 11.6M-node
> graph twice would be wasted minutes. YAML-loaded schedules *are* validated, since
> humans make mistakes.

### The Scheduler: DAG → events

The `Scheduler` ([scheduler.cpp](../src/schedule/scheduler.cpp)) is a runtime
topological executor. At construction it builds, in one pass:

- `remaining_deps_[i]` — how many unfinished deps instruction `i` has.
- `successors_[i]` — the instructions that depend on `i` (reverse edges).
- `by_id_[i]` — pointer to the instruction, for O(1) lookup.

Then:

```cpp
void launch() {                       // kick off everything with no deps
    for (inst : instructions)
        if (remaining_deps_[idx(inst.id)] == 0) try_issue(inst.id);
}

void notify_done(InstructionId id) {  // called by units on OP_DONE
    done_count_++;
    for (s : successors_[idx(id)])
        if (--remaining_deps_[idx(s)] == 0) try_issue(s);  // newly unblocked
}

void try_issue(InstructionId id) {    // hand the instr to its op handler
    if (issued_[idx(id)]) return;     // guard against double-issue
    issued_[idx(id)] = 1;
    registry_.get(inst->op.code())(IssueCtx{engine_, *this, *inst});
}
```

This is exactly Kahn's algorithm again, but **time-driven**: instead of instantly
visiting a node when its in-degree hits 0, the Scheduler *issues* it (creating
`OP_START`), and the node is only truly "removed" — `notify_done` — when the unit
fires `OP_DONE` cycles later. The interplay:

```
launch() ─issue─▶ OP_START (queued at reserved cycle)
                      │  engine.run() pops it
                      ▼
                  Unit.handle(OP_START) ─schedule─▶ OP_DONE (cycle+lat)
                      │  engine.run() pops it
                      ▼
                  Unit.handle(OP_DONE) ─▶ scheduler.notify_done(id)
                      │  decrements successors' remaining_deps_
                      ▼
                  try_issue(newly-ready) ─▶ more OP_STARTs ...
```

The run finishes when the heap drains. `all_done()` checks `done_count == N`;
`outstanding()` is the deadlock canary (should be 0).

### `idx()` — why integer indexing instead of hashing

`idx(id) = id - id_base_`. Because every builder assigns ids `0..N-1`
sequentially, `id_base_` is 0 and `idx` is the identity. That lets all four
structures be **flat `vector`s indexed directly by id** rather than hash maps. This
is both a speed and a memory win — see §7.

---

## 6. Where the instructions come from

Three front-ends produce a `Schedule`, all funneling through the same Scheduler.
This happens in `preprocess_schedule()`
([apps/sim_main.cpp](../apps/sim_main.cpp)), timed separately from simulation as
the "pre-processing" phase.

### (a) Hand-written YAML — `--schedule`
`Schedule::from_yaml_file()` parses a top-level `schedule:` sequence into
`Instruction`s (`from_node` in [schedule.cpp](../src/schedule/schedule.cpp)),
validates it, then runs it through the tiler to expand any oversized GEMMs.

### (b) Workload GEMM — `--workload`
`Tiler::from_yaml_file()` reads one logical GEMM; `Tiler::decompose()`
([tiler.cpp](../src/schedule/tiler.cpp)) splits it into hardware-sized
STAGE+GEMM sub-tiles and emits the instruction list directly.

### (c) Programmatic LLaMA — `--llama-workload`
`build_llama_schedule(cfg, minimal)` constructs a full transformer workload in
C++. This is the main path and the one worth understanding.

#### The `Builder` and `add()`
[llama_schedule.cpp](../src/schedule/llama_schedule.cpp) builds the instruction
vector with a tiny helper:

```cpp
struct Builder {
    std::vector<Instruction> out;
    InstructionId next = 0;
    bool minimal = false;
    InstructionId add(op, unit, label, params = {}, deps = {}) {
        Instruction inst;
        inst.id = next++;                 // sequential ids → flat-vector indexing
        inst.op = op; inst.unit = unit;
        if (minimal) { /* keep numeric params, drop strings+label */ }
        else        { inst.label = label; inst.params = params; }
        inst.params.shrink();             // release over-allocated capacity
        inst.depends_on = deps;
        out.push_back(inst);
        return out.back().id;             // callers thread this id into later deps
    }
};
```

Every `add()` returns the new id; higher-level builders thread those ids into the
`deps` of subsequent instructions, *constructing the DAG edge-by-edge as they go*.

#### Dispatch by mode
`build_llama_schedule` ([llama_schedule.cpp](../src/schedule/llama_schedule.cpp))
routes on `cfg.mode`:

- `attention` → `build_attention_schedule` → `append_attention`
- `layer` | `prefill` | `decode` → `build_transformer_layer_schedule`
  (embedding → `append_layer_stack` → output head)
- `prefill_decode` → `build_prefill_decode_schedule` (prefill once, then loop
  `generation_steps` decode iterations, threading the sampled-token feedback id
  from each step into the next)

`normalize_cfg()` derives/validates dims first (e.g. `head_dim = hidden_dim /
num_q_heads`, `gqa_group_size = num_q_heads / num_kv_heads`, KV-cache capacity
checks). The leaf builders (`append_detailed_tiled_gemm`, `append_detailed_rmsnorm`,
`append_detailed_rope`, `append_detailed_mlp_kernel`, `append_attention`, …) emit
the actual op instructions — a GEMM becomes weight-DMA-load → activation stage →
weight stage → `gemm` → output placement, an RMSNorm becomes square → row-reduce →
add-epsilon → rsqrt → scale, and so on. `append_attention` emits the exact
FlashAttention-2 inner loop (QK → scale → causal_mask → rowmax → online m/l/O
update → PV → accumulate, then post-loop normalize + logsumexp), with GQA KV-tile
reuse and P1.3 causal block-skip.

The takeaway: **the LLaMA builder is just nested loops calling `b.add(...)` and
threading returned ids into deps.** That's how 11.6M instructions and their DAG
get built in one pass.

### The Tiler pass
After building, `Tiler::expand_gemm_subtiles(std::move(sched), arch)`
([tiler.cpp](../src/schedule/tiler.cpp)) runs. It takes the schedule **by value**
so callers `std::move` into it, and **short-circuits to a no-op `return sched;`**
when no GEMM exceeds the array and structural K-tiling is off — which is the case
for every LLaMA path, because the builder already emits hardware-sized tiles. So on
the hot path the multi-million-instruction schedule is *moved through, never
copied*. (When a GEMM *is* oversized, it's rewritten into STAGE+GEMM sub-events and
dependents are rewired to the last sub-event; with `structural_k_tiling` it's
further split into per-K-block partial GEMMs + `accumulate` ops.)

---

## 7. Performance & memory engineering (10+ min → <4 min, ~9 GB → ~7 GB)

The full LLaMA-3-8B prefill is **11.6 million instructions**. At that scale the
simulator is dominated not by "simulation" but by *allocating, hashing, and
freeing data structures*. Almost all the speedups are about **not paying per-
instruction overheads 11 million times**. (See [VALIDATION.md](../VALIDATION.md)
for the measured tables; the numbers below are from the code comments that
implement each optimization.)

### 7.1 Event-driven, not cycle-driven (the foundational win)
The `run()` loop jumps to the next event instead of ticking every cycle (§3). A
schedule that finishes at cycle ~10⁹ never iterates 10⁹ times — it iterates once
per event. This is the difference between "tractable" and "never finishes."

### 7.2 Flat vectors instead of hash containers (Scheduler & validate)
The original Scheduler used three `unordered_map`/`unordered_set` containers keyed
by `InstructionId`. Because ids are sequential, they were replaced with flat vectors
indexed by `id - id_base_` ([scheduler.h](../src/schedule/scheduler.h)):

| Structure | Was (hash) | Now (flat vector) |
|---|---|---|
| `remaining_deps_` | ~370 MB | `vector<int>` ≈ 46 MB |
| `issued_` | ~280 MB | `vector<uint8_t>` ≈ 12 MB |
| `successors_` | ~510 MB | `vector<vector<uint32_t>>` ≈ 320 MB |
| `by_id_` | ~310 MB (map) | `vector<const Instruction*>` |

That's **~780 MB + ~310 MB** saved on the scheduler alone, plus the elimination of
a hash computation and pointer-chase on every `try_issue`/`notify_done` — pure
integer indexing into contiguous memory. `Schedule::validate()` got the identical
treatment (its Kahn's-algorithm sets/maps became flat vectors), turning a
~1.16 GB / hundreds-of-seconds validation into a linear pass.

### 7.3 Shrinking the `Instruction` struct (~1.7 GB)
At 11.1M instructions, every byte in `Instruction` costs ~11 MB. The struct was
attacked field by field in [instruction.h](../src/schedule/instruction.h):

- **`SmallStr` interning.** `op`, `unit`, `label` were `std::string` (32 B each).
  They became interned handles — `OpStr`/`UnitStr` are **1 byte** (a code into a
  global string pool), `LabelStr` is 2 bytes. Comparing two op names is now a byte
  compare, not a string compare. This both shrinks the struct and speeds up the
  scheduler's op lookup.
- **`CompactParamVal` (16 B tagged union)** replaced `std::variant<int64_t, double,
  string, bool>` (always 40 B because it must hold a 32-B string). Savings:
  24 B × ~2.5 params × 11.1M ≈ **667 MB**.
- **`ParamMap` flat vector with 1-byte keys.** Param keys went from `std::string`
  (32 B) to `KeyStr` (1 B). Each entry dropped from 72 B to 24 B. At ~2.5 params ×
  11.1M ≈ **1.33 GB** saved. The map is a flat vector with linear search — for
  3–8 params that beats hashing and avoids millions of tiny heap allocations.

Net struct + param savings: roughly **1.7 GB**.

### 7.4 `--no-trace` minimal-mode schedule build (~9.25 GB → ~7.0 GB, ~400 s → ~115 s)
`sim_main` sets `minimal = !trace`. In minimal mode the LLaMA `Builder::add` drops
**all string params** (buffer names like `source`/`destination`) and the label,
keeping only the numeric params the timing model actually reads (plus `init_value`,
whose `-inf` is a fill semantic). This is **timing-neutral** — verified
bit-identical `cycle`/`MACs`/`HBM_bytes` — because no latency formula depends on a
string (see [Guide 2](02-adding-a-new-operation.md)). It also silences the per-unit
`OP_START`/`OP_DONE` prints (the `verbose_` gate), whose *formatting* — not I/O —
dominates wall-clock on huge runs. This is why sweeps always use `--no-trace`.

### 7.5 One upfront reservation instead of 24 vector doublings
`Builder::out.reserve(estimate_instruction_count(cfg))`
([llama_schedule.cpp](../src/schedule/llama_schedule.cpp)) pre-sizes the instruction
vector to ~10% over the true count. Without it, a `vector` growing to 11.6M elements
doubles ~24 times, and glibc never returns the freed intermediate buffers to the OS
(they're sub-mmap-threshold and interleaved with live param data above them),
leaving ~2 GB of scattered dead heap. One reservation = **one allocation, zero
doublings, zero holes.** The estimate runs deliberately hot so it never
under-reserves and falls back to doubling.

### 7.6 Freeing per-instruction dep vectors after construction
Once the Scheduler has extracted `remaining_deps_` and `successors_` from each
instruction's `depends_on`, it frees those vectors
(`std::vector<InstructionId>{}.swap(inst.depends_on)` in
[scheduler.cpp](../src/schedule/scheduler.cpp)) — eliminating ~11M small heap
allocations and ~88 MB, and reducing minor page faults during the run.

### 7.7 `ParamMap::shrink()` and `malloc_trim`
Every `operator[]` on a `ParamMap` can double its capacity, leaving up to 50%
slack that would persist for the whole run. `Builder::add` calls `params.shrink()`
to fit exactly. And `sim_main` calls `malloc_trim(0)` after pre-processing and
after scheduler construction to hand freed pages back to the OS at the two big
high-water transitions.

### 7.8 Move-through tiler (avoids doubling peak RAM)
As described in §6, `expand_gemm_subtiles` takes the schedule by value and returns
it moved on the no-op fast path. Before this, tiling deep-copied the entire
multi-million-instruction schedule, momentarily **doubling** peak RAM. Now the hot
path copies nothing.

### Summary of the wins

| Optimization | Kind | Approx. impact |
|---|---|---|
| Event-driven run loop | speed | makes it tractable at all |
| Flat-vector Scheduler + validate | speed + RAM | ~1.1 GB, no hashing in hot loop |
| `SmallStr` / `CompactParamVal` / `ParamMap` | RAM | ~1.7 GB struct shrink |
| Minimal-mode build (`--no-trace`) | speed + RAM | ~2.25 GB, ~400 s → ~115 s |
| `out.reserve()` upfront | RAM | ~2 GB of avoided heap holes |
| Free `depends_on` post-construction | RAM | ~88 MB + ~11M fewer allocs |
| `shrink()` + `malloc_trim` | RAM | trims slack + returns pages to OS |
| Move-through tiler | RAM | avoids doubling peak |
| `verbose_` gate + integer op-code lookup | speed | no formatting / string-hash per instr |

Together these took the end-to-end full-8B run from **10+ minutes and ~9.25 GB** to
**under ~4 minutes and ~7.0 GB**, while staying bit-identical on every timing
metric.

---

## 8. Metrics — turning the run into numbers

After `engine.run()`, `sim_main` reads counters the engine accumulated *during*
the run ([apps/sim_main.cpp](../apps/sim_main.cpp), `== metrics ==` block):

- **Per-pool utilization** — from each unit's `busy_cycles` (summed in
  `reserve_unit_pool`) ÷ `final_cycle`. Multi-unit pools also print per-unit lines
  so you can see load balance.
- **Roofline** — `total_macs` (from `gemm` handlers) and `total_hbm_bytes` (from
  `dma` handlers) give `compute_cyc = ceil(MACs / peak_macs_per_cycle)` and
  `mem_cyc = ceil(bytes / hbm_bytes_per_cycle)`. The larger is the bound;
  `roofline_efficiency = bound / final_cycle`. `peak_macs_per_cycle` includes the
  bidirectional ×2 factor so bidir vs unidir compares correctly.
- **SRAM pressure** (when `model_sram`) — peak working set and spill count from the
  engine's `sram_acquire`/`sram_release` tracker.
- **TTFT / throughput** (LLaMA only) — `cycles_to_ns(final_cycle)` and tokens/sec,
  where the token count comes from the mode (`prompt_len`, `generation_steps`, …).

These counters live on the `EventEngine` precisely so they're collected for free as
events flow, with no second pass over the schedule.

---

## 9. Tracing the flow end-to-end (a tiny example)

For the 2-instruction schedule "delay 100 on u0, then delay 50 on u1 (depends on
0)" (from [test_dummy_units.cpp](../tests/test_dummy_units.cpp)):

```
construct Scheduler:
    remaining_deps_ = [0, 1]      successors_ = [[1], []]
launch():
    instr 0 has 0 deps → try_issue(0)
        delay handler reserves u0 for 100 cyc → OP_START{cycle 0, target u0, instr 0}
    instr 1 has 1 dep  → not issued
engine.run():
    pop OP_START(0,u0,0)   now_=0   → u0.handle → schedule OP_DONE{cycle 100, instr 0}
    pop OP_DONE(100,u0,0)  now_=100 → u0.handle → notify_done(0)
        successors_[0] = [1]; --remaining_deps_[1] → 0 → try_issue(1)
            delay handler reserves u1 for 50 → OP_START{cycle 100, target u1, instr 1}
    pop OP_START(100,u1,1) now_=100 → u1.handle → schedule OP_DONE{cycle 150, instr 1}
    pop OP_DONE(150,u1,1)  now_=150 → u1.handle → notify_done(1)   done_count=2
    queue empty → return 150
all_done() == true ;  final_cycle == 150 == 100 + 50
```

Every real run — including the 11.6M-instruction 8B prefill — is this same loop,
just with millions of events and real latency formulas instead of fixed delays.

---

## Key files

| File | Role |
|---|---|
| [core/event.h](../src/core/event.h) | `Event` struct, `EventType`, heap ordering |
| [core/event_engine.h](../src/core/event_engine.h) / [.cpp](../src/core/event_engine.cpp) | min-heap, `run()`, `reserve_unit_pool`, metrics counters |
| [core/unit.h](../src/core/unit.h) | unit base class / `handle()` contract |
| [schedule/instruction.h](../src/schedule/instruction.h) | `Instruction`, `ParamMap`, `SmallStr`, `CompactParamVal`, `pget_*` |
| [schedule/schedule.cpp](../src/schedule/schedule.cpp) | YAML parse + `validate()` (Kahn's) |
| [schedule/scheduler.h](../src/schedule/scheduler.h) / [.cpp](../src/schedule/scheduler.cpp) | DAG executor: `launch`, `notify_done`, `try_issue` |
| [schedule/op_registry.cpp](../src/schedule/op_registry.cpp) | name/code → handler lookup |
| [schedule/op_handlers.cpp](../src/schedule/op_handlers.cpp) | all built-in op handlers |
| [schedule/tiler.cpp](../src/schedule/tiler.cpp) | GEMM subtiling, move-through fast path |
| [schedule/llama_schedule.cpp](../src/schedule/llama_schedule.cpp) | programmatic transformer schedule builder |
| [apps/sim_main.cpp](../apps/sim_main.cpp) | CLI, pre-processing phase, engine wiring, metrics |

---

## Summary

The engine is two cooperating machines: an **EventEngine** that owns a min-heap of
`(cycle, seq)`-ordered events and advances time by *jumping* to the next one, and a
**Scheduler** that owns the instruction DAG and uses a runtime Kahn's algorithm to
issue instructions as their dependencies complete. Op handlers translate
instructions into typed `OP_START` events; units turn those into future `OP_DONE`
events and call `notify_done()` to walk the DAG forward; the engine's resource
table models hardware contention and accumulates the metrics. The dramatic
speed/memory improvements come almost entirely from refusing to pay per-instruction
overhead 11 million times — event-driven time, flat-vector graph structures,
interned/compacted instruction fields, an upfront reservation, and a timing-neutral
minimal build mode under `--no-trace`.
