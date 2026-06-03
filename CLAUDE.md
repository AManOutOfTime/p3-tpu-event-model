# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sP3** is a C++17 cycle-level, event-driven simulator for heterogeneous LLM accelerators (systolic array + vector/access cores + DMA), targeting LLaMA-3-8B inference. It is timing-only (no actual computation); units model latency and fire events. The simulator is parametric via YAML architecture configs and supports hand-written or programmatically-generated instruction schedules.

## Build & Run

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run with defaults (configs/default.yaml + schedules/dummy_example.yaml)
./build/apps/sim_main

# Custom config + YAML schedule
./build/apps/sim_main --config <cfg.yaml> --schedule <sched.yaml>

# Programmatic LLaMA workload (generates schedule internally)
./build/apps/sim_main --config <cfg.yaml> --llama-workload <llama.yaml>

# Suppress per-event trace (summary only)
./build/apps/sim_main --no-trace
```

Dependencies (yaml-cpp 0.8.0, doctest v2.4.11) are auto-fetched by CMake — no manual install needed.

## Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run the test binary directly
./build/tests/unit_tests                   # Linux/macOS
build\tests\Debug\unit_tests.exe           # Windows (MSVC)

# Run a single named test case (doctest filter)
./build/tests/unit_tests -tc="My test name"
```

Test files live in `tests/`. To add a test: create a `.cpp` file and list it in `tests/CMakeLists.txt`.

## Architecture

The simulation stack has five layers:

### 1. Event Engine (`src/core/`)
Cycle-based min-heap priority queue. Time jumps to the next event; no per-cycle loop. Units register with `register_unit()` (which assigns a `UnitId` and optionally a buffer capacity). Events carry `(cycle, seq, type, target, instr_id, label, payload)`. Same-cycle events are ordered by insertion sequence (`seq`) for determinism. `find_unit_pool(name_prefix)` returns all registered units matching a logical name (e.g., `"vector_core"` → `["vector_core_0", "vector_core_1"]`). `reserve_unit_pool(candidates, latency)` picks the earliest-free unit with available buffer capacity.

### 2. Config (`src/config/`)
`ArchConfig` loads `configs/default.yaml` and exposes a typed field hierarchy: `systolic` (rows, cols, precision, bidirectional, d_head), `vector_core` (simd_width, exp_latency), `access_core` (bandwidth), `sram` (ibuf_kb, obuf_kb, banking_factor, private_vector_kb), `hbm` (bandwidth_tb_s, latency_cycles), `dma` (channels), plus pool counts (systolic_units, vector_cores, access_cores). Config is read once at startup. `hbm_bytes_per_cycle()` is a derived helper.

### 3. Schedule & Scheduler (`src/schedule/`)
A YAML schedule is a sequence of `Instruction` records: `{id, op, unit, params, depends_on[], label}`. `Schedule::validate()` runs Kahn's algorithm to reject cycles or unknown dep ids. `Scheduler` dispatches instructions as their dependencies complete; it calls the appropriate `OpRegistry` handler, which is responsible for scheduling events and eventually calling `scheduler.notify_done(instr_id)` to unblock dependents.

`Tiler` decomposes oversized logical GEMMs into hardware-sized sub-tiles (M≤SA_rows, N≤SA_cols) and rewires the dependency graph. `Tiler::expand_gemm_subtiles()` rewrites a full schedule in-place. `build_llama_schedule()` generates full prefill/decode/attention schedules programmatically from a `LlamaScheduleConfig`.

### 4. Op Handlers (`src/schedule/op_handlers.h/cpp`)
Handlers are registered with `OpRegistry` and invoked with an `IssueCtx {engine, scheduler, inst}`. Built-in ops:

| Op | Unit | Latency formula |
|---|---|---|
| `dma_load` / `dma_store` | dma | `hbm_latency_cycles + ceil(bytes / hbm_bytes_per_cycle)` |
| `dma_stage` | dma | `ceil(bytes / banking_factor)` |
| `gemm` | systolic | `K + fill_latency` (fill = (r-1)+(c-1) uni; half that bidir) |
| `transpose` / `sram_copy` | access_core | `ceil(elements / bandwidth)` |
| `init_fill` | access_core | `ceil(elements / bandwidth)` |
| `scale`, `rowmax`, `exp_shift`, `normalize`, etc. | vector_core | `ceil(elements / simd_width) * passes + exp_ops * exp_latency * groups` |
| `kv_stage_release` | access_core | 0 cycles (dependency marker only) |

### 5. Hardware Units (`src/units/`)
All derive from `Unit` and implement `handle(const Event&, EventEngine&)`. Units are timing-only: they compute a latency, schedule an `OP_DONE` event, and call `scheduler.notify_done()`. `DelayUnit` is the reference stub. `SystolicUnit`, `DmaUnit`, `VectorUnit`, `AccessUnit` are the real models.

## Extending the Simulator

**Add a new hardware unit:**
1. Create `src/units/my_unit.h/.cpp` deriving from `Unit`; implement `handle()` — on `OP_START` compute latency and schedule `OP_DONE`; on `OP_DONE` call `scheduler_->notify_done(e.instr)`.
2. Add `units/my_unit.cpp` to `SIM_CORE_SOURCES` in `src/CMakeLists.txt`.
3. Instantiate and `register_unit()` in `apps/sim_main.cpp`.

**Add a new op handler:**
```cpp
registry.register_op("my_op", [&cfg](const IssueCtx& ctx) {
    auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
    auto res = ctx.scheduler.reserve_unit_pool(targets, latency);
    Event e; e.type = OP_START; e.target = res.id;
    e.cycle = res.start; e.instr = ctx.inst.id; e.label = ctx.inst.label;
    e.payload = MyPayload{...};
    ctx.engine.schedule(e);
});
```
The unit's `handle()` on `OP_DONE` must call `notify_done`.

**Add a new op to a YAML schedule:**
```yaml
- id: 5
  op: my_op
  unit: my_unit
  params: { bytes: 65536 }
  depends_on: [3, 4]
  label: "my description"
```

## Key Files

| File | Role |
|---|---|
| `src/core/event_engine.h/cpp` | Core discrete-event loop, unit pool reservation |
| `src/core/event.h` | `Event` struct and `EventType` enum |
| `src/core/unit.h` | Base class for all hardware units |
| `src/schedule/scheduler.h/cpp` | DAG-respecting instruction dispatcher |
| `src/schedule/op_registry.h/cpp` | Dynamic op-handler lookup |
| `src/schedule/op_handlers.h/cpp` | All built-in op implementations |
| `src/schedule/tiler.h/cpp` | GEMM subtiling and schedule rewriting |
| `src/schedule/llama_schedule.h/cpp` | Programmatic LLaMA workload generation |
| `apps/sim_main.cpp` | CLI entry point |
| `configs/default.yaml` | Architecture parameters (edit without rebuild) |
| `schedules/dummy_example.yaml` | Sample 320-cycle DMA→GEMM→softmax schedule |
| `schedules/fa2_single_tile.yaml` | 22-instruction FlashAttention-2 single-tile schedule |
