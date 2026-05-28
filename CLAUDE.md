# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**sP3** is a C++ cycle-level, event-driven simulator for TPU-style LLM accelerators,
targeting FlashAttention-2 (FA2) inference on LLaMA-3-8B class models.

It is parametric and cycle-accurate. The architecture is defined in a YAML config
(`configs/default.yaml`). Workloads are expressed either as hand-written instruction
schedules (`schedules/`) or as matrix-size workload files (`workloads/`) that the
Tiler automatically decomposes into per-tile instructions.

The main reference workload is a single-head FA2 forward pass implemented in
`schedules/fa2_single_tile.yaml`. Every instruction uses a real named op — no
`delay` stubs. Every hardware unit performs actual float computation at `OP_DONE`
in addition to modelling cycle-accurate latency.

---

## Build and run

```bash
# Configure and build (dependencies auto-fetched by CMake)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run the FA2 schedule (primary use case)
./build/apps/sim_main --schedule schedules/fa2_single_tile.yaml

# Summary only (no per-event trace)
./build/apps/sim_main --schedule schedules/fa2_single_tile.yaml --no-trace

# Auto-tiled workload (Tiler generates instructions from matrix sizes)
./build/apps/sim_main --workload workloads/fa2_qkt.yaml --no-trace

# Custom config
./build/apps/sim_main --config configs/default.yaml --schedule schedules/fa2_single_tile.yaml

# Run tests
./build/tests/unit_tests
```

Flags: `--config FILE`, `--schedule FILE`, `--workload FILE`, `--no-trace`.
`--schedule` and `--workload` are mutually exclusive.

On Windows (PowerShell):
```powershell
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
.\build\apps\Release\sim_main.exe --schedule schedules\fa2_single_tile.yaml --no-trace
```

---

## Repository layout

```
apps/
  sim_main.cpp            CLI entry point — unit instantiation, op registration,
                          TensorStore seeding, --workload / --schedule dispatch

configs/
  default.yaml            Architecture parameters (edit without rebuild)

schedules/
  fa2_single_tile.yaml    FA2 forward pass, single head, one Q tile × one KV tile
  dummy_example.yaml      Minimal smoke-test schedule

workloads/
  fa2_qkt.yaml            FA2 Q@K^T workload (Br=256, d_head=128, Bc=256)
  gemm_512.yaml           Plain 512×512×512 GEMM

src/
  config/
    arch_config.h/.cpp    ArchConfig — all hardware parameters, YAML load/save

  core/
    types.h               Cycle, UnitId, InstructionId typedefs
    event.h               Event struct (cycle, type, target, payload, label)
    unit.h                Base Unit class — virtual handle(Event, EventEngine)
    event_engine.h/.cpp   Discrete-event loop — min-heap, unit registry,
                          unit-pool reservation (available_at per unit)
    logger.h/.cpp         ConsoleLogger — formats events to stdout
    tensor_store.h        Named float buffer store — shared across all units,
                          slice_rows / slice_cols / place_tile / init helpers

  schedule/
    instruction.h         Instruction struct — id, op, unit, params, depends_on
    schedule.h/.cpp       Schedule loader — parses YAML, handles op:[placeholder]
    op_registry.h/.cpp    Dynamic op-name → handler lookup (IssueCtx)
    scheduler.h/.cpp      DAG dispatcher — remaining_deps, successors, reserve_unit_pool
    tiler.h/.cpp          Tiler — decomposes WorkloadGemm × ArchConfig into
                          STAGE + GEMM instructions (Q-stationary loop order)

  units/
    delay_unit.h/.cpp     Generic fixed-latency stub (backward compat)
    systolic_unit.h/.cpp  Systolic array — timing + tiled float GEMM at OP_DONE
    dma_unit.h/.cpp       DMA — dma_load / dma_store / dma_stage latency models,
                          buffer copy in TensorStore at OP_DONE
    vector_unit.h/.cpp    Vector/tandem core — scale, rowmax, update_rowmax,
                          exp_shift, update_rowsum, accumulate, normalize, logsumexp
    access_unit.h/.cpp    Access core — init_fill, transpose
    buffer_unit.h/.cpp    Double-buffered banked SRAM — sram_read / sram_write ops,
                          banking contention model, double-buffer ping/pong halves

tests/
  test_event_engine.cpp     EventEngine unit tests
  test_config.cpp           ArchConfig YAML round-trip tests
  test_schedule.cpp         Schedule parser tests
  test_dummy_units.cpp      DelayUnit / PrintingUnit tests
  test_fa2_schedule.cpp     FA2 schedule parse + DAG + simulation tests
  test_fa2_correctness.cpp  End-to-end FA2 numerical correctness tests
```

---

## Architecture — four layers

### 1. Event engine (`src/core/event_engine.h`)

Discrete-event simulation. The engine holds a min-heap of `Event` structs ordered
by `cycle`. `engine.run()` pops events one at a time and dispatches each to its
target unit via `unit.handle(event, engine)`. Time jumps directly to the next
event — there is no per-cycle loop.

Units are registered by name. `find_unit(name)` returns a `UnitId`.
`find_unit_pool(prefix)` returns all units whose name starts with `prefix`
(e.g. `"vector_core"` matches `"vector_core_0"`, `"vector_core_1"`, `"vector_core_2"`).

`reserve_unit_pool(targets, duration)` picks the first available unit from a pool
and books it:
```cpp
Cycle start = max(now, available_at[uid]);
available_at[uid] = start + duration;
return {uid, start};
```
This is the entire "reservation station" — one integer per unit. No queue.

### 2. Scheduler (`src/schedule/scheduler.h`)

Reads a `Schedule` (vector of `Instruction`). Tracks `remaining_deps[id]` for each
instruction. When `remaining_deps[id]` hits zero, `try_issue()` is called immediately,
which looks up the op handler in `OpRegistry` and calls it.

The op handler builds a typed payload (e.g. `GemmShape`, `DmaTransfer`, `VectorOp`)
and calls `engine.schedule(OP_START event)`. The event sits in the heap until its
cycle arrives, then fires. The unit handles `OP_START`, schedules `OP_DONE` at
`start + latency`, and calls `sched.notify_done(id)` from `OP_DONE`.
`notify_done` decrements `remaining_deps` of all successors.

### 3. Hardware units (`src/units/`)

Every unit derives from `Unit` and implements `handle(const Event&, EventEngine&)`.
Two event types are used:
- `OP_START` — unit receives payload, computes latency, schedules `OP_DONE`
- `OP_DONE`  — unit executes compute (writes TensorStore), calls `notify_done`

All units accept an `int64_t` payload as fallback (backward compat with `op: delay`).

Unit latency models:

| Unit | Op | Latency |
|---|---|---|
| SystolicUnit | gemm | `tiles_m × tiles_n × (K + fill_latency)` |
| SystolicUnit | — | `fill_latency = (rows-1)+(cols-1)` unidir, `÷2` bidir |
| DmaUnit | dma_load / dma_store | `hbm_latency + ceil(bytes / hbm_bw)` |
| DmaUnit | dma_stage | `ceil(bytes / banking_factor)` (on-chip only) |
| VectorUnit | all ops | `passes × ceil(elems/simd_width) + exp_ops × exp_latency × groups` |
| AccessUnit | init_fill / transpose | `ceil(elements / access_core.bandwidth)` |

### 4. TensorStore (`src/core/tensor_store.h`)

Flat `string → vector<float>` map. Buffer keys follow a naming convention:
```
"shared_ibuf.*"              on-chip SRAM input buffer
"shared_obuf.*"              on-chip SRAM output buffer
"systolic_array.Q_operand"   array PE input register (staged by dma_stage)
"systolic_array.P_operand"   array PE input register
"vector_scratch.*"           vector core scratch space
```

DmaUnit copies `src_buf → dst_buf` at `OP_DONE`. SystolicUnit writes `dst_c`
with the tiled GEMM result at `OP_DONE`. VectorUnit and AccessUnit write their
output buffers at `OP_DONE`.

---

## FA2 ops and units

The full op set used by `schedules/fa2_single_tile.yaml`:

| Op | Unit | What it computes |
|---|---|---|
| `dma_load` | dma | copies HBM key into `shared_ibuf`, charges HBM latency |
| `dma_store` | dma | copies `shared_obuf` key to HBM, charges HBM latency |
| `dma_stage` | dma | copies IBUF key → array register, charges SRAM read latency |
| `sram_read` | shared_ibuf / shared_obuf | read through BufferUnit with banking contention + double-buffer |
| `sram_write` | shared_ibuf / shared_obuf | write through BufferUnit with banking contention + double-buffer |
| `init_fill` | access_core | fills buffer with 0 / −∞ in TensorStore |
| `transpose` | access_core | row-major matrix transpose in TensorStore |
| `gemm` | systolic | C = A × B tiled float GEMM, writes result to TensorStore |
| `scale` | vector_core | element-wise multiply (scalar or row-vector broadcast) |
| `rowmax` | vector_core | row-wise max reduction → length-Br vector |
| `update_rowmax` | vector_core | m = max(m, r); correction = exp(m_old − m_new) |
| `exp_shift` | vector_core | P = exp(S − m) with row broadcast |
| `update_rowsum` | vector_core | l = correction × l_old + rowsum(P) |
| `accumulate` | vector_core | O_acc += Temp element-wise |
| `normalize` | vector_core | O_tile = O_acc / l row-wise |
| `logsumexp` | vector_core | L = m + log(l) element-wise |

---

## Tiler (`src/schedule/tiler.h`)

Decomposes a `WorkloadGemm` (full matrix sizes) + `ArchConfig` (array size) into
explicit `dma_stage` + `gemm` instructions using Q-stationary loop order:

```
outer i: Q row sub-tiles  (Q stays in IBUF, reused for all j)
  STAGE: shared_ibuf.Q_sub_r{i} → systolic_array.Q_operand

  inner j: KT col sub-tiles
    GEMM: Q_operand × shared_ibuf.KT_sub_c{j} → shared_obuf.S_sub_r{i}_c{j}
```

K (d_head) is never tiled — it streams fully through each array execution.
After simulation, `Tiler::assemble_output()` places sub-tile results back into
the full output buffer.

Activated via `--workload FILE` instead of `--schedule FILE`.

---

## Arch config (`configs/default.yaml`)

All parameters editable without rebuild:

```yaml
clock_ghz: 1.0

systolic:
  rows:          128
  cols:          128
  precision:     BF16      # FP8 | FP16 | BF16 | FP32
  bidirectional: false     # true halves fill_latency
  d_head:        128       # K streaming dimension

vector_cores: 3
access_cores: 1

sram:
  ibuf_kb:           4096   # shared input buffer
  obuf_kb:           4096   # shared output buffer
  banking_factor:    8      # parallel SRAM ports (affects dma_stage and sram_read/write latency)
  private_vector_kb: 512    # per-vector-core private SRAM

hbm:
  bandwidth_tb_s: 0.9       # TPUv3-like; TPUv2 ~0.7, H100 ~3.35
  latency_cycles: 100       # ~100 ns at 1 GHz

dma:
  channels: 1

vector_core:
  simd_width:  128          # TPUv2: 128 vector lanes (Norrie 2021 p.4)
  exp_latency: 10           # transcendental pipeline depth ~10-20 cycles

access_core:
  bandwidth: 256            # elements/cycle for transpose / init_fill (TPUv2 Norrie 2021 p.5)
```

Bidirectional fill latency: `ceil((rows-1)/2) + ceil((cols-1)/2)` instead of
`(rows-1) + (cols-1)`. Benefits small-K workloads most (e.g. attention with
d_head=64 on a large array).

---

## Schedule YAML format

```yaml
schedule:
  - id: 0
    op: dma_load           # op name — must be registered in sim_main.cpp
    unit: dma              # unit pool prefix
    label: "human label"   # optional, shown in trace
    params:
      source:      "HBM.Q_tile"
      destination: "shared_ibuf.Q_tile"
      rows: "Br"           # symbolic — resolved to arch.systolic.rows
      cols: "d_k"          # symbolic — resolved to arch.systolic.d_head

  - id: 1
    op: gemm
    unit: systolic
    depends_on: [0]        # won't issue until id=0 calls notify_done
    params:
      source_a:    "systolic_array.Q_operand"
      source_b:    "shared_ibuf.K_tile_T"
      destination: "shared_obuf.S_tile"
      M: "Br"
      K: "d_k"
      N: "Bc"
```

Symbolic dimension values resolved in `sim_main.cpp::resolve()`:
- `"Br"`, `"Bc"` → `arch.systolic.rows` / `.cols`
- `"d_k"`, `"d_head"` → `arch.systolic.d_head`

`op: [placeholder]` (YAML sequence) is also accepted — parsed as the string
`"placeholder"` for backward compatibility with older schedule files.

---

## Adding a new hardware unit

1. Create `src/units/my_unit.h/.cpp` deriving from `Unit`:
   ```cpp
   class MyUnit : public Unit {
   public:
       void handle(const Event& e, EventEngine& engine) override;
       void set_scheduler(Scheduler* s) { sched_ = s; }
       void set_tensor_store(TensorStore* ts) { ts_ = ts; }
   };
   ```
2. Add to `src/CMakeLists.txt` under `SIM_CORE_SOURCES`.
3. Register new op handlers in `apps/sim_main.cpp::register_all_ops()`.
4. Instantiate and register with the engine in `main()`.
5. Add `dynamic_cast` wire-up in `wire_units()`.
6. Add tests in `tests/`.

## Adding a new op

1. Add a `reg.register_op("my_op", ...)` handler in `register_all_ops()`.
2. The handler reads params from `ctx.inst.params` via `pget_int / pget_str / pget_bool / pget_dbl`.
3. Build a typed payload struct, call `reserve_unit_pool` for the target pool, schedule `OP_START`.
4. The unit's `handle(OP_DONE)` must call `sched_->notify_done(e.instr)`.

## Adding a new schedule

Create a YAML file under `schedules/` following the format above. Run with:
```bash
./build/apps/sim_main --schedule schedules/my_schedule.yaml
```

## Adding a new workload (for Tiler)

Create a YAML file under `workloads/`:
```yaml
workload:
  Br:     256          # or M:
  d_head: 128          # or K:
  Bc:     256          # or N:
  src_a:  "shared_ibuf.Q_tile"
  src_b:  "shared_ibuf.KT_tile"
  dst_c:  "shared_obuf.S_tile"
  fill:   random       # random | zeros | ones
```

Run with:
```bash
./build/apps/sim_main --workload workloads/my_workload.yaml
```

---

## What is and is not modeled

**Modeled (cycle-accurate):**
- Systolic array GEMM latency with tiling and fill-pipeline delay
- Bidirectional wavefront (halved fill cost)
- HBM load/store latency (fixed + bandwidth)
- On-chip SRAM stage latency (banking factor)
- SRAM banking contention between concurrent accesses (BufferUnit)
- Double-buffer IBUF/OBUF — producer (DMA) and consumer (systolic/vector) overlap via ping/pong halves
- Weight-stationary pre-loading cost
- Vector core SIMD latency + transcendental (exp/log) overhead
- Access core transpose / init-fill latency
- Unit structural hazards (one array — all GEMMs serialized)
- Instruction-level parallelism across different units (DMA prefetch overlaps compute)
- Actual float values for all operations end-to-end

**Not yet modeled:**
- Multi-head attention / GQA outer loops (single head only)
- Multi-core / multi-chip parallelism
- Sparse attention / paged KV cache / scatter-gather
- K-split (tiling the K/d_head dimension across multiple array executions)
- Weight-stationary pre-loading of subsequent tiles during compute (weights assumed pre-staged before first tile)
