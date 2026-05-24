# sP3 — Cycle-Level Event-Driven Simulator

C++ event-driven cycle-accurate simulator for heterogeneous LLM accelerators
(systolic array + vector cores + access cores + DMA). Targets LLaMA-3-8B inference.

---

## Directory layout

```
p3-tpu-event-model/
├── CMakeLists.txt          # root build
├── cmake/
│   └── FetchDeps.cmake     # yaml-cpp + doctest via FetchContent (auto-downloaded)
├── configs/
│   └── default.yaml        # parametric architecture config (edit freely)
├── schedules/
│   └── dummy_example.yaml  # sample schedule: DMA -> transpose -> GEMM -> softmax
├── src/
│   ├── CMakeLists.txt      # add new .cpp files here — nothing else to touch
│   ├── core/
│   │   ├── types.h         # Cycle, UnitId, EventId typedefs
│   │   ├── event.h         # Event struct + EventType enum
│   │   ├── unit.h          # Unit base class
│   │   ├── event_engine.h/cpp
│   │   └── logger.h/cpp    # ConsoleLogger trace hook
│   ├── config/
│   │   └── arch_config.h/cpp
│   ├── schedule/
│   │   ├── instruction.h   # Instruction + ParamMap + pget_* helpers
│   │   ├── schedule.h/cpp  # Schedule (YAML loader + DAG validation)
│   │   ├── op_registry.h/cpp
│   │   └── scheduler.h/cpp
│   └── units/
│       ├── printing_unit.h/cpp   # prints every event (smoke-test)
│       └── delay_unit.h/cpp      # fixed-latency stub (template for real units)
├── apps/
│   └── sim_main.cpp        # CLI driver
└── tests/
    ├── test_event_engine.cpp
    ├── test_config.cpp
    ├── test_schedule.cpp
    └── test_dummy_units.cpp
```

---

## Build

**Requirements:** CMake ≥ 3.20, C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+).  
Dependencies (yaml-cpp, doctest) are fetched automatically on first build.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

---

## Run tests

```bash
ctest --test-dir build --output-on-failure

# Or run directly for full doctest output:
./build/tests/unit_tests          # Linux/macOS
build\tests\Debug\unit_tests.exe  # Windows
```

---

## Run the simulator

```bash
# Default: configs/default.yaml + schedules/dummy_example.yaml
./build/apps/sim_main

# Custom files
./build/apps/sim_main --config configs/default.yaml --schedule schedules/dummy_example.yaml

# Suppress per-event trace (summary only)
./build/apps/sim_main --no-trace
```

Expected output for `dummy_example.yaml` (320-cycle serial chain):

```
== simulation start  instructions=4 ==
[cycle        0 | 0.000 ns]  OP_START     -> dma_0              "DMA load K_tile from HBM"
  [dma_0]  START  instr=0  @cycle=0  lat=50  "DMA load K_tile from HBM"
  [dma_0]  DONE   instr=0  @cycle=50  "DMA load K_tile from HBM"
...
== simulation done  cycle=320  (320.000 ns)  outstanding=0 ==
```

---

## How to add a new hardware unit

1. Copy `src/units/delay_unit.h/.cpp` → `src/units/my_unit.h/.cpp`, rename class, implement `handle()`.
2. Add `units/my_unit.cpp` to the `SIM_CORE_SOURCES` list in `src/CMakeLists.txt`.
3. Register physical instances in `apps/sim_main.cpp`, e.g. `my_unit_0`, `my_unit_1`.
4. Reference the logical pool name in YAML, e.g. `unit: my_unit`; the scheduler picks the earliest-free physical instance.

## How to add a new op (any granularity)

1. Write: `void my_op(const sim::IssueCtx& ctx) { ... }`
2. Register: `registry.register_op("my_op", my_op);`
3. Use `op: my_op` in your YAML schedule.

**Coarse ops** (e.g. `flash_attn2`) fire events on multiple units at once.  
**Fine ops** (e.g. `dma_load`, `gemm`) fire one event on one unit.  
Both use the same schedule format and the same engine — only the handler differs.

## How to add a new test

1. Create `tests/test_my_thing.cpp`:
```cpp
#include <doctest/doctest.h>
TEST_CASE("my test") { REQUIRE(1 + 1 == 2); }
```
2. Add the filename to `tests/CMakeLists.txt`.
3. `cmake --build build && ctest --test-dir build`.

---

## Architecture config parameters

| Field | Default | Description |
|---|---|---|
| `clock_ghz` | 1.0 | Clock frequency. `cycles / clock_ghz = ns`. |
| `systolic.rows/cols` | 128×128 | Systolic array dimensions |
| `systolic.precision` | BF16 | FP8 / FP16 / BF16 / FP32 |
| `vector_cores` | 3 | Number of vector cores |
| `access_cores` | 1 | Number of Access Cores (transpose, scatter-gather) |
| `sram.ibuf_kb` | 4096 | Shared input buffer |
| `sram.obuf_kb` | 4096 | Shared output buffer |
| `sram.banking_factor` | 8 | Concurrent r/w ports per cycle |
| `sram.private_vector_kb` | 512 | Per-vector-core private SRAM |
| `hbm.bandwidth_tb_s` | 2.0 | HBM bandwidth (TB/s) |
| `hbm.latency_cycles` | 200 | HBM round-trip latency in cycles |
| `dma.channels` | 1 | DMA channels |
