# Documentation

Developer guides for the **sP3** cycle-level, event-driven TPU/accelerator
simulator. Start with whichever matches your task; each guide links into the
others where they overlap.

| Guide | Read it when you want to… |
|---|---|
| [01 — Adding a New Hardware Unit](01-adding-a-new-unit.md) | Model a new piece of hardware (a core, a buffer, a DMA variant): the `Unit` contract, the `OP_START`→`OP_DONE`→`notify_done()` protocol, registering + wiring it into `sim_main`, and testing it. |
| [02 — Adding a New Callable Operation](02-adding-a-new-operation.md) | Make a new op name callable from any schedule: writing/registering an `OpHandler`, reading params, reserving a unit, building a typed payload, and accumulating roofline metrics. |
| [03 — How the Simulator Engine Works](03-simulator-engine.md) | Understand the whole machine: events, the min-heap, units as latency models, how the Scheduler turns an instruction DAG into events (Kahn's algorithm), how `llama_schedule.cpp`/`tiler.cpp` generate instructions, and the performance/memory engineering (10+ min → <4 min, ~9 GB → ~7 GB). |
| [04 — Creating Schedules (incl. ONNX models)](04-creating-schedules.md) | Author workloads for models *other than* LLaMA: the instruction format, the op vocabulary by latency class, hand-written YAML, a C++ builder, and a Python generator that turns an [ONNX](https://github.com/onnx/models) graph into a runnable schedule. |
| [05 — Writing a C++ Schedule Generator](05-cpp-schedule-generator.md) | Build a programmatic generator like `llama_schedule.cpp` step by step: the `Builder`, `append_*` leaf helpers, the config struct + YAML loader, wiring it into `sim_main`, the RAM/perf habits, and testing — with a complete worked MLP-stack generator. |

For the project-level overview, build/run/test commands, and architecture config
knobs, see the repository [CLAUDE.md](../CLAUDE.md) and [README.md](../README.md).
For the validation report and RAM/timing tables referenced in Guide 3, see
[VALIDATION.md](../VALIDATION.md).
