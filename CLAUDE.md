# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sP3** is a C++ cycle-level, event-driven simulator for heterogeneous LLM accelerators (systolic array + vector/access cores + DMA), targeting LLaMA-3-8B inference. It is parametric and cycle-accurate, driven by YAML architecture configs and instruction schedules.

## Build & Run

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run with defaults (configs/default.yaml + schedules/dummy_example.yaml)
./build/apps/sim_main

# Custom config/schedule
./build/apps/sim_main --config <file> --schedule <file>

# Suppress per-event trace (summary only)
./build/apps/sim_main --no-trace
```

Dependencies (yaml-cpp 0.8.0, doctest v2.4.11) are auto-fetched by CMake — no manual install needed.

## Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run the test binary directly
./build/tests/unit_tests          # Linux/macOS
build\tests\Debug\unit_tests.exe  # Windows (MSVC)
```

Tests use the doctest framework (`TEST_CASE` macros). Test files live in `tests/`.

## Architecture

The simulation stack has four layers:

1. **Event Engine** (`src/core/`) — Cycle-based priority queue. Time jumps to the next event; no per-cycle loop. Units register with the engine and receive events via a virtual `handle()` method. Same-cycle events are ordered by insertion sequence for determinism.

2. **Schedule & Scheduler** (`src/schedule/`) — A YAML schedule defines instructions (op, target unit, params, dependency list). The `Schedule` loader validates the dependency DAG; the `Scheduler` dispatches instructions to units as their dependencies complete. Op handlers are looked up dynamically via `OpRegistry`.

3. **Hardware Units** (`src/units/`) — Concrete units derive from `Unit` (`src/core/unit.h`) and implement `handle()`. `DelayUnit` is the current stub: it fires `OP_START`, waits `latency_cycles`, then fires `OP_DONE`. Real hardware models (systolic, DMA, etc.) should follow this same pattern.

4. **Config** (`src/config/`) — `ArchConfig` loads `configs/default.yaml` and exposes typed fields for the systolic array, vector/access cores, SRAM, HBM, DMA, and clock frequency. Config is read once at startup and passed to units.

## Extending the Simulator

**Add a new hardware unit:**
1. Create `src/units/my_unit.h/.cpp` deriving from `Unit`; implement `handle(const Event&)`
2. Register op handlers in `OpRegistry` (or within the unit constructor)
3. Instantiate and register with the engine in `apps/sim_main.cpp`
4. Add a `TEST_CASE` in `tests/`

**Add a new op:**
- Define the op string in `src/schedule/op_registry.h` and register its handler in `op_registry.cpp`
- Add the op to a schedule YAML under `instructions:`

**Add a new test:**
- Add a `.cpp` file to `tests/` and list it in `tests/CMakeLists.txt`

## Key Files

| File | Role |
|---|---|
| `src/core/event_engine.h/cpp` | Core discrete-event loop |
| `src/core/event.h` | `Event` struct (cycle, type, unit, payload) |
| `src/core/unit.h` | Base class for all hardware units |
| `src/schedule/scheduler.h/cpp` | DAG-respecting instruction dispatcher |
| `src/schedule/op_registry.h/cpp` | Dynamic op-handler lookup |
| `apps/sim_main.cpp` | CLI entry point |
| `configs/default.yaml` | Architecture parameters (edit without rebuild) |
| `schedules/dummy_example.yaml` | Sample 320-cycle DMA→GEMM→softmax schedule |
