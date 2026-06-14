# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sP3** is a C++17 cycle-level, event-driven simulator for heterogeneous LLM accelerators (systolic array + vector/access cores + DMA), targeting LLaMA-3-8B inference. It exists to drive design-space sweeps — array size (edge 64² ↔ datacenter 256²+), unit counts, HBM bandwidth, SRAM capacity, dataflow — and answer "diminishing returns" / "where's the bottleneck (compute vs memory)" questions.

**This branch is timing-only.** Units compute a latency, schedule an `OP_DONE`, and call `notify_done()`; they do **not** perform real float math. There is no numerical-result data bus — treat the "real computation / numerical verification" story as aspirational, not current. This matters when extending: e.g. structural K-tiling can split GEMMs purely for timing/traffic without worrying about numerical correctness. The simulator is parametric via YAML architecture configs and supports hand-written or programmatically-generated instruction schedules.

## Project Status & Workflow

`PLAN.md` is the living source of truth for project status — current verified build/test/runtime state, completed work, changed files, and remaining issues. Per `AGENTS.md`, read `PLAN.md` before editing code (identify the current incomplete step, verify existing code state, avoid redoing completed work), and update it after finishing (mark completed items, record changed files, note tests run and remaining issues).

## Build & Run

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel                     # Linux/macOS
cmake --build build --config Release --parallel     # Windows (MSVC ignores CMAKE_BUILD_TYPE)

# Run with defaults (configs/default.yaml + schedules/dummy_example.yaml)
./build/apps/sim_main

# Custom config + YAML schedule
./build/apps/sim_main --config <cfg.yaml> --schedule <sched.yaml>

# Workload YAML (Tiler decomposes GEMM tiles automatically)
./build/apps/sim_main --config <cfg.yaml> --workload <wl.yaml>

# Programmatic LLaMA workload (generates schedule internally)
./build/apps/sim_main --config <cfg.yaml> --llama-workload <llama.yaml>

# Suppress per-event trace (summary only)
./build/apps/sim_main --no-trace
```

Every run ends with a `== metrics ==` block (P0.2): per-pool utilization, total MACs, off-chip HBM bytes, a compute-vs-memory roofline bound + efficiency, SRAM peak/spills (when `model_sram` is on), and TTFT/throughput (LLaMA workloads). Use `--no-trace` for sweeps — the per-event log dominates wall-clock on large schedules.

`--no-trace` does more than silence logging: for LLaMA workloads it builds the schedule in **`minimal` metadata mode** (drops human-readable labels + symbolic buffer-name string params, keeping only numeric params the timing model actually reads + `init_value`). This is timing-neutral (verified bit-identical `cycle`/`MACs`/`HBM_bytes`) but cuts host RAM substantially on huge schedules — the full LLaMA-3-8B prefill (11.6 M instructions) drops from ~9.25 GB to ~7.0 GB and from ~400 s to ~115 s. See `VALIDATION.md` for the full validation report and RAM/timing tables.

Dependencies (yaml-cpp 0.8.0, doctest v2.4.11) are auto-fetched by CMake — no manual install needed.

On Windows (MSVC), Visual Studio is a multi-config generator and **ignores** `-DCMAKE_BUILD_TYPE` — `cmake --build build` alone yields a Debug build under `build\apps\Debug\`. With `--config Release` (above) the binary lands at `build\apps\Release\sim_main.exe` (use that path in place of `./build/apps/sim_main`).

## Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure             # Linux/macOS
ctest --test-dir build -C Release --output-on-failure   # Windows (MSVC — must pass -C <config>)

# Run the test binary directly
./build/tests/unit_tests                   # Linux/macOS
build\tests\Release\unit_tests.exe         # Windows (MSVC; use Debug\ if built without --config Release)

# Run a single named test case (doctest filter)
./build/tests/unit_tests -tc="My test name"
```

Test files live in `tests/`. To add a test: create a `.cpp` file and list it in `tests/CMakeLists.txt`.

## Design-Space Sweeps & Comparison Scripts

The project's headline deliverable is sweep data answering "where's the bottleneck" questions across hardware configurations. Three Python scripts drive `sim_main` in batches with `--no-trace` and write CSVs. `compare.py` is at the repo root; `sweep.py` / `sweep_safe.py` live in `sweep/scripts/`. **Run all three from the repo root** — they resolve `configs/…`, `workloads/…`, and the `sim_main` binary by relative path. On Windows (MSVC), pass `--binary build/apps/Release/sim_main.exe` for real runs (the scripts default to the Linux path `./build/apps/sim_main`); `--dry-run` previews need no binary.

- **`sweep/scripts/sweep.py`** — single-axis sweeps on top of a base config (default `configs/default.yaml`, no PyYAML dependency). Groups: `1a`-`1f` compute (array size, systolic unit count, bidirectional, vector cores, SIMD width × exp_latency, access bandwidth), `2a`-`2e` memory (HBM bandwidth, HBM latency, DMA channels, `stage_double_buffer`, SRAM pressure via `model_sram`), `3a`-`3e` software (prompt length, tile size, KV cache on/off, head_dim × hidden_dim, max_seq_len/KV footprint), `4a`/`4b` GQA group size (8B/70B), `5` array-size × HBM-bw Pareto grid, `6` calibration against published roofline efficiency (target ±20%).
  ```bash
  python3 sweep/scripts/sweep.py --dry-run --group 1b           # preview a group's config table
  python3 sweep/scripts/sweep.py --model 8b --group 3e --out sweep_3e.csv
  python3 sweep/scripts/sweep.py --model both                   # run every group on 8B and 70B (slow)
  ```
- **`sweep/scripts/sweep_safe.py`** — memory-constrained variant (tuned for ~16 GB / 4-CPU pods, ~2.4 GB/run): caps `prompt_len`/`max_seq_len` (`SAFE_PLEN=512`, `SAFE_MAXSEQ=4096`), treats subprocess exit 137 as `OOM_KILLED` and other non-zero exits as `EXIT_N` (writes a `FAILED` row and continues), and adds `--focused` (high-signal groups `1b,1c,2a,2d,3c,4a` only, ~60 configs/~35 min) and `--no-modes` (run `prefill_decode` only, skipping the separate `decode` pass).
- **`compare.py`** — fixed cross-product: `configs/datacenter.yaml` × `configs/edge_dev.yaml` × `workloads/llama_prefill_decode_{1B,8B,70B}.yaml` × `{prefill_decode, decode}` modes (12 runs), emitting KPI columns compatible with the sweep CSVs (TTFT for `prefill_decode`, decode tok/s for `decode`). Marks runs `MEM_ERR` (vs. a generic failure) when sim output matches known OOM signal strings.

`configs/edge_dev.yaml` (64×64 array, 1 systolic unit, ~0.1 TB/s LPDDR5X-class HBM — modeled on Apple M3 / Snapdragon NPU) and `configs/datacenter.yaml` (256×256 array, 8 systolic units, 3.35 TB/s HBM3 — modeled on H100 SXM5) are the two preset HW profiles these scripts compare against `configs/default.yaml`. Pre-existing `sweep_*.csv` / `*_results.csv` / `edge_dev*.csv` / `datacenter.csv` files at repo root are prior sweep outputs — useful as a schema/value-range reference.

## Architecture

The simulation stack has these layers:

### 1. Event Engine (`src/core/`)
Cycle-based min-heap priority queue. Time jumps to the next event; no per-cycle loop. Units register with `register_unit()` (which assigns a `UnitId` and optionally a buffer capacity). Events carry `(cycle, seq, type, target, instr_id, label, payload)`. Same-cycle events are ordered by insertion sequence (`seq`) for determinism. `find_unit_pool(name_prefix)` returns all registered units matching a logical name (e.g., `"vector_core"` → `["vector_core_0", "vector_core_1"]`). `reserve_unit_pool(candidates, latency)` picks the earliest-free unit with available buffer capacity.

### 2. Config (`src/config/`)
`ArchConfig` loads `configs/default.yaml` and exposes a typed field hierarchy: `systolic` (rows, cols, precision, bidirectional, d_head), `vector_core` (simd_width, exp_latency), `access_core` (bandwidth), `sram` (ibuf_kb, obuf_kb, banking_factor, private_vector_kb), `hbm` (bandwidth_tb_s, latency_cycles), `dma` (channels), plus pool counts (systolic_units, vector_cores, access_cores). Config is read once at startup. `hbm_bytes_per_cycle()` is a derived helper.

**Modeling knobs added for sweeps** (all parse from YAML and round-trip through `to_yaml_string`):
- `systolic.dataflow`: `weight_stationary` (default) | `output_stationary` — selects the GEMM latency model (see Op Handlers).
- `systolic.weight_load_cycles`: per-K-block weight-load cost; `0` ⇒ auto (= `rows`).
- `systolic.weight_double_buffer` (default `true`, **P1.4**): hide weight-load(i+1) behind stream(i).
- `hbm.pipelined` (default `true`, **P1.5**): channel occupancy = bandwidth term only; latency overlaps adjacent transfers (so streams are bandwidth-bound, not latency-bound).
- `structural_k_tiling` (default `false`, **P1.2**): Tiler emits explicit per-K-block GEMMs + `accumulate` ops (real partial-sum OBUF traffic) instead of folding K into the analytical latency.
- `model_sram` (default `false`, **P1.2**): bind IBUF+OBUF capacity; over-capacity GEMM working sets spill (HBM penalty).
- `stage_double_buffer` (default `false`, **P1.2/S2**): ping-pong operand staging banks so staging overlaps compute.

`LlamaScheduleConfig.causal_block_skip` (default `true`, **P1.3**) lives with the schedule config, not `ArchConfig`.

### 3. Schedule & Scheduler (`src/schedule/`)
A YAML schedule is a sequence of `Instruction` records: `{id, op, unit, params, depends_on[], label}`. `params` is a `ParamMap` — a flat vector-backed `string→ParamVal` map (not `unordered_map`) chosen because instructions carry only 3–8 params; linear search beats hashing at this size and avoids millions of tiny heap allocations in large schedules. Use `pget_int/pget_dbl/pget_str/pget_bool` helpers to read params. `Schedule::validate()` runs Kahn's algorithm to reject cycles or unknown dep ids. `Scheduler` dispatches instructions as their dependencies complete using an O(1) `by_id_` lookup map; it calls the appropriate `OpRegistry` handler, which is responsible for scheduling events and eventually calling `scheduler.notify_done(instr_id)` to unblock dependents.

`Tiler` decomposes oversized logical GEMMs into hardware-sized sub-tiles (M≤SA_rows, N≤SA_cols) and rewires the dependency graph. `Tiler::expand_gemm_subtiles()` takes the `Schedule` **by value** (so callers `std::move` into it) and short-circuits to a no-op `return sched;` — moved, not copied — when no GEMM exceeds the array and `structural_k_tiling` is off. This is the common case for every LLaMA path (the builder already emits hardware-sized tiles), so the move/fast-path avoids deep-copying a multi-million-instruction schedule. **K is left whole** on the gemm instruction — the weight-stationary latency model fragments K analytically. When `structural_k_tiling` is set, `expand_gemm_subtiles()` chains `expand_k_subtiles()` to split K>rows GEMMs into explicit per-K-block sub-GEMMs + `accumulate` ops (modeling partial-sum traffic). `decompose()` also honors `stage_double_buffer` (S2) by ping-ponging two operand banks.

`build_llama_schedule(cfg, minimal=false)` is the top-level entry point; internally it calls `build_attention_schedule()` (mode `attention`), `build_transformer_layer_schedule()` (modes `layer` | `prefill` | `decode`), or `build_prefill_decode_schedule()` (mode `prefill_decode`) depending on `LlamaScheduleConfig::mode`. The `minimal` flag (set to `!trace` by `sim_main`) drops trace-only metadata as described under Build & Run. The config also controls KV-cache policy (`kv_cache_enabled`, `kv_cache_location`: `sram`|`hbm`, `kv_prefetch`: `none`|`double_buffer`, `kv_cache_eviction_policy`: `fail`|`spill_to_hbm`), tiling granularity (`tile_rows/cols` for attention, `linear_tile_rows/cols` for linear layers), and **`causal_block_skip`** (P1.3): in `append_attention`'s `kt` loop, KV tiles fully above the diagonal (`kv_first > q_last` for every Q-tile) are skipped entirely and `causal_mask` ops are emitted only on diagonal-straddling tiles. Position-based, so decode (which attends to all past keys) never skips.

### 4. Op Handlers (`src/schedule/op_handlers.h/cpp`)
Handlers are registered with `OpRegistry` and invoked with an `IssueCtx {engine, scheduler, inst}`. Built-in ops:

| Op | Unit | Latency formula |
|---|---|---|
| `dma_load` / `dma_store` | dma | data ready after `hbm_latency_cycles + ceil(bytes / (hbm_bytes_per_cycle·channels))`. With `hbm.pipelined` (P1.5) the channel is *occupied* only for the bandwidth term `ceil(bytes/bw)` (latency overlaps neighbors); otherwise occupied for the full latency+bandwidth. |
| `dma_stage` | dma | `ceil(elements / systolic.rows)` — models IBUF→systolic array ingest over the wide operand bus (one column of `rows` PEs per cycle); NOT bounded by the narrower SRAM `banking_factor` |
| `gemm` | systolic | **weight-stationary** (P0.1): `⌈N/cols⌉·(⌈K/rows⌉·max(weight_load, M) + weight_load + fill)` with double-buffer (P1.4), or `⌈N/cols⌉·(⌈K/rows⌉·(weight_load + M) + fill)` without. `output_stationary` mode = legacy `⌈M/rows⌉·⌈N/cols⌉·(K + fill)`. Computed by the shared free function `systolic_gemm_latency()` (declared in `units/systolic_unit.h`, used by both the handler and the unit so reservation and `OP_DONE` can't desync). The handler also accumulates `M·K·N` MACs; `dma_load/store` accumulate HBM bytes — both feed the P0.2 roofline. |
| `transpose` / `sram_copy` | access_core | `ceil(elements / bandwidth)` |
| `init_fill` | access_core | `ceil(elements / bandwidth)` |
| `scale`, `rowmax`, `exp_shift`, `normalize`, etc. | vector_core | `ceil(elements / simd_width) * passes + exp_ops * exp_latency * groups` |
| `kv_stage_release` | access_core | 0 cycles (dependency marker only) |

### 5. Metrics (`src/core/event_engine.cpp`, `apps/sim_main.cpp`)
P0.2 accounting lives on `EventEngine`: per-unit `busy_cycles` (summed in `reserve_unit_pool`), global `total_macs_`/`total_hbm_bytes_` (added by handlers), and a shared SRAM working-set tracker (`sram_acquire/release`, peak + spill counters; capacity 0 = unlimited). `sim_main` reads these after `run()` to print the `== metrics ==` block. Everything is timing-only; there is no numerical-result data bus.

### 6. Hardware Units (`src/units/`)
All derive from `Unit` and implement `handle(const Event&, EventEngine&)`. Units are timing-only: on `OP_START` they compute a latency, schedule an `OP_DONE` event; on `OP_DONE` they call `scheduler.notify_done()`. `DelayUnit` is the reference stub. `SystolicUnit`, `DmaUnit`, `VectorUnit`, `AccessUnit` are the models. `SystolicUnit` additionally **acquires its SRAM working set at `OP_START` and releases at `OP_DONE`** (P1.2) — acquiring at issue time would over-count, since the scheduler issues handlers far ahead of execution.

## Extending the Simulator

**Add a new hardware unit:**
1. Create `src/units/my_unit.h/.cpp` deriving from `Unit`; implement `handle()` — on `OP_START` compute latency and schedule `OP_DONE`; on `OP_DONE` call `scheduler_->notify_done(e.instr)`.
2. Add `units/my_unit.cpp` to `SIM_CORE_SOURCES` in `src/CMakeLists.txt`.
3. Instantiate and `register_unit()` in `apps/sim_main.cpp`.
4. Add a `dynamic_cast<MyUnit*>` branch in `wire_units()` in `apps/sim_main.cpp` to inject the `Scheduler*` (via `set_scheduler`) — without this the unit can't call `notify_done()`.

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
| `src/schedule/instruction.h` | `Instruction` struct, `ParamMap` (flat vector-backed map), `pget_*` helpers |
| `src/schedule/scheduler.h/cpp` | DAG-respecting instruction dispatcher |
| `src/schedule/op_registry.h/cpp` | Dynamic op-handler lookup |
| `src/schedule/op_handlers.h/cpp` | All built-in op implementations |
| `src/schedule/tiler.h/cpp` | GEMM subtiling and schedule rewriting |
| `src/schedule/llama_schedule.h/cpp` | Programmatic LLaMA workload generation |
| `apps/sim_main.cpp` | CLI entry point |
| `configs/default.yaml` | Architecture parameters (edit without rebuild) |
| `configs/edge_dev.yaml` / `configs/datacenter.yaml` | Preset HW profiles (edge SoC vs. H100-class) for sweeps/`compare.py` |
| `schedules/dummy_example.yaml` | Minimal sample schedule (4 `delay` ops across units; the default if no schedule is given) |
| `schedules/fa2_single_tile.yaml` | 22-instruction FlashAttention-2 single-tile schedule |
| `workloads/val_*.yaml` | Validation workloads (real LLaMA-3-8B dims): single head, layer, full 8B, GQA vs MHA, causal vs non-causal |
| `workloads/llama_prefill_decode_{1B,8B,70B}.yaml` | Model-size workloads driving `sweep/scripts/sweep.py`/`sweep_safe.py`/`compare.py` |
| `sweep/scripts/sweep.py` / `sweep/scripts/sweep_safe.py` / `compare.py` | Design-space sweep and HW×workload comparison drivers (see Design-Space Sweeps) |
| `PLAN.md` / `AGENTS.md` | Live project status + agent update workflow (see Project Status & Workflow) |
| `VALIDATION.md` | Validation report: schedule/TPU-timing fidelity, MAC/cycle self-consistency, RAM optimization tables |
| `src/core/logger.h/cpp` | `ConsoleLogger` — attach to engine with `engine.set_trace([&](const Event& e){ logger(e); })` |
