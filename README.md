# sP3 — Cycle-Level Event-Driven Accelerator Simulator

C++17 **discrete-event, cycle-level** simulator for heterogeneous LLM accelerators
(systolic array + vector cores + access cores + DMA). It targets **LLaMA-3-8B**
inference and exists to drive **design-space sweeps** — array size (edge 64² ↔
datacenter 256²+), unit counts, HBM bandwidth, SRAM capacity, dataflow — and answer
"diminishing returns" / "where's the bottleneck: compute or memory?" questions.

**Timing-only.** Each hardware unit is a *latency model*: it receives an
`OP_START`, computes how many cycles the operation takes, schedules an `OP_DONE`,
and signals completion. No real float math is performed — the deliverable is
*cycles, HBM bytes, and utilization*, which are independent of the numerical
values. The engine is event-driven (a min-heap of events; time *jumps* to the next
event), so idle gaps cost zero wall-time and the full 11.6 M-instruction LLaMA-3-8B
prefill runs in minutes.

---

## Quick Start

```bash
# 1. Configure + build (deps yaml-cpp + doctest are auto-fetched by CMake)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel                     # Linux/macOS
cmake --build build --config Release --parallel    # Windows (MSVC ignores CMAKE_BUILD_TYPE — pass --config)

# 2. Run the default smoke test (configs/default.yaml + schedules/dummy_example.yaml)
./build/apps/sim_main                     # Linux/macOS
build\apps\Release\sim_main.exe           # Windows (MSVC)

# 3. Run a real LLaMA-3-8B workload, summary only (no per-event trace)
./build/apps/sim_main \
  --config configs/default.yaml \
  --llama-workload workloads/llama_prefill_decode_8B.yaml \
  --no-trace

# 4. Run the tests
ctest --test-dir build --output-on-failure             # Linux/macOS
ctest --test-dir build -C Release --output-on-failure  # Windows (MSVC — must pass -C <config>)
```

Every run ends with a `== metrics ==` block: per-pool utilization, total MACs,
off-chip HBM bytes, a compute-vs-memory **roofline** bound + efficiency, SRAM
peak/spills (when `model_sram` is on), and TTFT/throughput (LLaMA workloads).

**Four ways to feed the simulator** (pick one):

| Flag | Input | Use it for |
|---|---|---|
| `--schedule FILE` | hand-written YAML instruction list | small, exact, hand-tuned schedules |
| `--workload FILE` | one logical GEMM; the **Tiler** decomposes it | studying a single matmul's tiling |
| `--llama-workload FILE` | a LLaMA workload spec; the schedule is generated in C++ | full transformer prefill/decode |
| *(none)* | defaults to `schedules/dummy_example.yaml` | first-run smoke test |

Always pass `--no-trace` for sweeps and large LLaMA runs — the per-event log
dominates wall-clock (and for LLaMA workloads `--no-trace` also builds the schedule
in a RAM-lean "minimal" mode; see [Running the simulator](#running-the-simulator)).

> New to the codebase? Read [`documentation/`](documentation/) — three guides on
> adding a unit, adding an op, and how the engine works.

---

## Directory layout

```
p3-tpu-event-model/
├── CMakeLists.txt          # root build
├── cmake/FetchDeps.cmake   # yaml-cpp + doctest via FetchContent (auto-downloaded)
├── configs/                # parametric architecture configs (edit, no rebuild)
│   ├── default.yaml        #   256² bidirectional baseline
│   ├── datacenter.yaml     #   H100-class: 256², 8 systolic units, 3.35 TB/s HBM3
│   └── edge_dev.yaml       #   edge SoC:   64², 1 systolic unit, ~0.1 TB/s LPDDR5X
├── schedules/              # hand-written YAML schedules
│   ├── dummy_example.yaml  #   4-op linear chain (default smoke test)
│   ├── fa2_single_tile.yaml#   22-instruction FlashAttention-2 single tile
│   └── gqa_fa2_*.yaml       #   GQA + FA2 single vs overlapped dataflows
├── workloads/              # GEMM (--workload) and LLaMA (--llama-workload) specs
│   ├── gemm_512.yaml       #   plain 512³ GEMM for the Tiler
│   ├── llama_prefill_decode_{1B,8B,70B}.yaml
│   └── val_*.yaml          #   validation workloads (real LLaMA-3-8B dims)
├── src/
│   ├── CMakeLists.txt      # add new .cpp files here — nothing else to touch
│   ├── core/               # event engine, Event, Unit base, types, logger
│   ├── config/             # ArchConfig (YAML load + typed fields + round-trip)
│   ├── schedule/           # Instruction, Schedule, Scheduler, OpRegistry,
│   │                       #   op_handlers, Tiler, llama_schedule (builder)
│   └── units/              # systolic, dma, vector, access, delay, printing
├── apps/sim_main.cpp       # CLI driver
├── tests/                  # doctest unit tests (test_*.cpp)
├── compare.py              # HW×workload comparison driver (repo root)
├── sweep/scripts/          # sweep.py / sweep_safe.py design-space sweep drivers
├── documentation/          # developer guides (unit / op / engine internals)
├── CLAUDE.md               # architecture deep-dive + modeling knobs
├── VALIDATION.md           # fidelity + RAM/timing validation report
└── PLAN.md / AGENTS.md     # live worklog + agent update workflow
```

---

## Build

**Requirements:** CMake ≥ 3.20, a C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+).
Dependencies — **yaml-cpp 0.8.0** and **doctest v2.4.11** — are fetched
automatically by CMake on first configure; nothing to install manually.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel                     # Linux/macOS
cmake --build build --config Release --parallel    # Windows (MSVC)
```

> **Windows / MSVC note:** Visual Studio is a *multi-config* generator, so it
> **ignores** `-DCMAKE_BUILD_TYPE=Release` at configure time — `cmake --build build`
> alone produces a **Debug** build under `build\apps\Debug\`. Pass
> `--config Release` (as above) to get the Release binary at the path below.

| Platform | `sim_main` binary | test binary |
|---|---|---|
| Linux / macOS | `build/apps/sim_main` | `build/tests/unit_tests` |
| Windows (MSVC, `--config Release`) | `build\apps\Release\sim_main.exe` | `build\tests\Release\unit_tests.exe` |
| Windows (MSVC, default Debug) | `build\apps\Debug\sim_main.exe` | `build\tests\Debug\unit_tests.exe` |

Substitute the platform path wherever this README writes `./build/apps/sim_main`.

---

## Run tests

```bash
# All tests via CTest
ctest --test-dir build --output-on-failure             # Linux/macOS
ctest --test-dir build -C Release --output-on-failure  # Windows (MSVC — must pass -C <config>)

# Or run the binary directly for full doctest output
./build/tests/unit_tests                   # Linux/macOS
build\tests\Release\unit_tests.exe         # Windows (use Debug\ if built without --config Release)

# A single test case by name (doctest filter)
./build/tests/unit_tests -tc="serial chain: total cycles = sum of individual latencies"
```

> **Windows / MSVC note:** `ctest --test-dir build` without `-C <config>` fails with
> *"Test not available without configuration. (Missing `-C <config>`?)"* — pass
> `-C Release` (or `-C Debug`, matching how you built) as shown above.

Test files live in [`tests/`](tests). To add one: create `tests/test_<thing>.cpp`
and append the filename to `SIM_TEST_SOURCES` in
[`tests/CMakeLists.txt`](tests/CMakeLists.txt) — that's the only wiring needed.
(`SIM_PROJECT_ROOT` is baked into the test binary so tests can load real YAML files
regardless of the working directory.)

---

## Running the simulator

```bash
# Default: configs/default.yaml + schedules/dummy_example.yaml
./build/apps/sim_main

# Hand-written schedule
./build/apps/sim_main --config configs/default.yaml --schedule schedules/fa2_single_tile.yaml

# Workload GEMM (Tiler decomposes one logical GEMM into HW-sized sub-tiles)
./build/apps/sim_main --config configs/default.yaml --workload workloads/gemm_512.yaml

# Programmatic LLaMA workload (schedule generated in C++)
./build/apps/sim_main --config configs/default.yaml --llama-workload workloads/llama_prefill_decode_8B.yaml

# Summary only — silence the per-event trace (always use this for big runs/sweeps)
./build/apps/sim_main --llama-workload workloads/val_full8b.yaml --no-trace
```

`--config` defaults to `configs/default.yaml`. If no input flag is given,
`schedules/dummy_example.yaml` is used.

Expected output for `dummy_example.yaml` (an 888-cycle serial chain
`DMA 50 → transpose 30 → GEMM 768 → softmax 40`; the GEMM cost comes from the
weight-stationary latency model on the 256² array, not a literal `200`):

```
== simulation start  instructions=4 ==
[cycle        0 | 0.000 ns]  OP_START     -> dma_0     "DMA load K_tile from HBM"
  [dma_0]  DMA_START  instr=0  @cycle=0  lat=50  "DMA load K_tile from HBM"
  [dma_0]  DMA_DONE   instr=0  @cycle=50  "DMA load K_tile from HBM"
...
== simulation done  cycle=888  (888.000 ns)  outstanding=0 ==

== metrics ==
  ...per-pool utilization, MACs, HBM bytes, roofline, (SRAM, TTFT/throughput)...
```

### What `--no-trace` does (beyond silencing logs)

For **LLaMA** workloads, `--no-trace` builds the schedule in **`minimal` metadata
mode**: it drops human-readable labels and the symbolic buffer-name string params,
keeping only the numeric params the timing model actually reads. This is
**timing-neutral** (verified bit-identical `cycle` / `MACs` / `HBM_bytes`) but cuts
host RAM substantially on huge schedules — the full LLaMA-3-8B prefill (11.6 M
instructions) drops from **~9.25 GB → ~7.0 GB** and from **~400 s → ~115 s**. It
also gates off the per-unit `OP_START`/`OP_DONE` prints, whose *formatting*
dominates wall-clock on large runs. See [VALIDATION.md](VALIDATION.md) for the full
RAM/timing tables.

---

## Architecture configs

Architecture parameters live in `configs/*.yaml` and are read **once at startup** —
edit a value and re-run, no rebuild. All fields are optional; missing fields keep
the `ArchConfig` defaults. Three presets ship:

| Preset | Models | Array | Clock | Systolic units | HBM BW | DMA ch |
|---|---|---|---|---|---|---|
| `default.yaml` | bidirectional baseline | 256² | 1.0 GHz | 2 | 2.0 TB/s | 1 |
| `datacenter.yaml` | H100 SXM5-class | 256² | 2.0 GHz | 8 | 3.35 TB/s (HBM3) | 4 |
| `edge_dev.yaml` | Apple M-/Snapdragon-NPU-class | 64² | 1.5 GHz | 1 | 0.10 TB/s (LPDDR5X) | 1 |

### Config fields

Values below are the `configs/default.yaml` settings. Knobs tagged **P1.x** are
modeling features added for sweeps (all parse from YAML and round-trip through
`ArchConfig::to_yaml_string`).

| Field | Default | Description |
|---|---|---|
| `clock_ghz` | 1.0 | Clock frequency. `cycles / clock_ghz = ns`. |
| `systolic.rows` / `cols` | 256 / 256 | Systolic array dimensions. |
| `systolic.precision` | BF16 | FP8 / FP16 / BF16 / FP32 (sets `dtype_bytes`). |
| `systolic.bidirectional` | true | Feed from both edges → ~½ pipeline-fill latency. |
| `systolic.d_head` | 128 | Attention head dim (= K streaming axis). |
| `systolic.dataflow` | weight_stationary | `weight_stationary` \| `output_stationary` — selects the GEMM latency model. |
| `systolic.weight_load_cycles` | 0 | Per-K-block weight-load cost; `0` ⇒ auto (= `rows`). |
| `systolic.weight_double_buffer` | true | **P1.4**: hide weight-load(i+1) behind stream(i). |
| `structural_k_tiling` | false | **P1.2**: emit explicit per-K-block GEMMs + `accumulate` ops (real partial-sum OBUF traffic). |
| `model_sram` | true | **P1.2**: bind IBUF+OBUF capacity; over-capacity working sets spill (HBM penalty). |
| `stage_double_buffer` | true | **P1.2/S2**: ping-pong operand staging so it overlaps compute. |
| `systolic_units` | 2 | Number of systolic/MXU arrays in the pool. |
| `vector_cores` | 3 | Number of vector/VPU cores. |
| `access_cores` | 1 | Number of access cores (transpose, scatter-gather, copy). |
| `sram.ibuf_kb` / `obuf_kb` | 4096 / 4096 | Shared input / output buffer. |
| `sram.banking_factor` | 8 | Concurrent SRAM r/w ports per cycle. |
| `sram.private_vector_kb` | 512 | Per-vector-core private SRAM. |
| `hbm.bandwidth_tb_s` | 2.0 | HBM bandwidth (H100 ~3.35; TPUv4 ~1.2). |
| `hbm.latency_cycles` | 200 | HBM round-trip latency. |
| `hbm.pipelined` | true | **P1.5**: channel occupancy = bandwidth term only (latency overlaps neighbors → streams are bandwidth-bound). |
| `dma.channels` | 1 | DMA channels (multiplies effective HBM bandwidth). |
| `vector_core.simd_width` | 64 | Lanes per vector core. |
| `vector_core.exp_latency` | 4 | Cycles per transcendental (exp/rsqrt) group. |
| `access_core.bandwidth` | 64 | Elements per cycle. |

See [CLAUDE.md](CLAUDE.md) for the exact latency formulas each knob feeds.

---

## Generated LLaMA workloads (`--llama-workload`)

Instead of a hand-written schedule, the simulator can **generate** a LLaMA-style
schedule in C++ from a small workload spec. The `Tiler` is run afterward but is a
no-op for these paths (the builder already emits hardware-sized tiles).

```bash
./build/apps/sim_main --config configs/default.yaml \
  --llama-workload workloads/llama_prefill_decode_8B.yaml --no-trace
```

### Modes (`llama.mode`)

| Mode | Meaning |
|---|---|
| `attention` | Build only the attention block (FlashAttention-2 inner loop). |
| `layer` | Build one full transformer layer (attention + MLP + residuals). |
| `prefill` | Process a prompt sequence and optionally populate the KV cache. |
| `decode` | Process one generated token using existing prompt/KV context. |
| `prefill_decode` | Run prefill once, then `generation_steps` decode iterations. |

### Sequence fields

| Field | Meaning |
|---|---|
| `prompt_len` | Input prompt tokens processed during prefill. |
| `generation_steps` | New tokens generated after prefill in `prefill_decode`. |
| `seq_len` | Sequence length for standalone `attention`/`layer` paths. |
| `max_seq_len` | KV-cache capacity / planning length. If omitted: `max(seq_len, prompt_len + generation_steps)`. |

For `mode: prefill_decode`, prefill is driven by `prompt_len`; each decode step
processes one token and attends over the prompt plus previously generated tokens.
With `prompt_len: 4` and `generation_steps: 2`, the schedule prefills 4 tokens, then
decodes positions 4 and 5 (zero-based), with KV lengths 5 and 6. In a
`prefill_decode`-only YAML, `seq_len` may be omitted if `max_seq_len` is set; keep
`seq_len` if you reuse the same YAML for standalone `attention`/`layer`/`prefill`.

### Model + attention fields

`num_layers`, `num_q_heads`, `num_kv_heads` (GQA — Q heads share KV heads),
`gqa_group_size`, `head_dim`, `hidden_dim`, `intermediate_dim`, `vocab_size`,
`dtype_bytes`, attention tile shape (`tile_rows`/`tile_cols`), linear-layer tile
shape (`linear_tile_rows`/`linear_tile_cols`), and KV-cache policy
(`kv_cache_enabled`, `kv_cache_location: sram|hbm`, `kv_prefetch`,
`kv_cache_eviction_policy`, `causal_block_skip`). `head_dim` and `gqa_group_size`
are **derived** when omitted (`head_dim = hidden_dim / num_q_heads`,
`gqa_group_size = num_q_heads / num_kv_heads`) and validated for consistency.

### Validation workloads (real LLaMA-3-8B dims)

[`workloads/val_*.yaml`](workloads) cover a single head, one layer, the full 8B
model, GQA vs MHA, and causal vs non-causal attention — used by
[VALIDATION.md](VALIDATION.md). The `llama_prefill_decode_{1B,8B,70B}.yaml` files
drive the sweep scripts below.

---

## Workload GEMM mode (`--workload`)

Describe a single logical GEMM; the `Tiler` decomposes it into hardware-sized
STAGE + GEMM sub-tiles (M ≤ array rows, N ≤ array cols; K streams and is never
tiled here) and rewires the dependency graph:

```yaml
# workloads/gemm_512.yaml
workload:
  M: 512
  K: 512
  N: 512
  src_a: "A"
  src_b: "B"
  dst_c: "C"
  fill: random
```

```bash
./build/apps/sim_main --config configs/default.yaml --workload workloads/gemm_512.yaml
```

The run prints the tile decomposition (how many array executions the GEMM became)
alongside the usual metrics.

---

## Design-space sweeps

Three Python drivers run `sim_main` in batches (always with `--no-trace`) and write
CSVs. They need only a working `sim_main` and Python 3 (no PyYAML required by
`sweep.py`). `compare.py` lives at the repo root; `sweep.py` / `sweep_safe.py` live
in [`sweep/scripts/`](sweep/scripts). **Run all three from the repo root** — they
resolve `configs/…`, `workloads/…`, and the `sim_main` binary by relative path.

| Script | What it does |
|---|---|
| [`sweep/scripts/sweep.py`](sweep/scripts/sweep.py) | Varies **one axis at a time** on top of a base config. Groups `1a–1f` (compute: array size, systolic count, bidirectional, vector cores, SIMD width, access BW), `2a–2e` (memory: HBM BW/latency, DMA channels, `stage_double_buffer`, SRAM pressure), `3a–3e` (software: prompt length, tile size, KV cache, head/hidden dim, KV footprint), `4a/4b` (GQA group size), `5` (array × HBM-BW Pareto grid), `6` (calibration vs published roofline efficiency). |
| [`sweep/scripts/sweep_safe.py`](sweep/scripts/sweep_safe.py) | Memory-constrained variant (~16 GB / 4-CPU pods): caps `prompt_len`/`max_seq_len`, treats OOM (exit 137) and other failures as recoverable `FAILED` rows, and adds `--focused` (high-signal groups only) and `--no-modes`. |
| [`compare.py`](compare.py) | Fixed cross-product `configs/datacenter.yaml × configs/edge_dev.yaml × {1B,8B,70B} × {prefill_decode, decode}` (12 runs), emitting KPI columns compatible with the sweep CSVs. |

```bash
python3 sweep/scripts/sweep.py --dry-run --group 1b           # preview a group's config table
python3 sweep/scripts/sweep.py --model 8b --group 3e --out sweep_3e.csv
python3 sweep/scripts/sweep_safe.py --focused --out sweep_focused.csv
python3 compare.py --dry-run
```

> **Windows note:** `sweep.py`/`sweep_safe.py` default to the Linux binary path
> (`./build/apps/sim_main`). For a real sweep on Windows (MSVC), point them at the
> built binary: `--binary build/apps/Release/sim_main.exe`. `--dry-run` previews
> need no binary. `compare.py` accepts the same `--binary` override.

Key KPIs: TTFT (prefill latency), decode tok/s, `hbm_util_pct` (calibration
target), `roofline_eff_pct`, `mem_compute_ratio` (>1 = memory-bound),
`arith_intensity`, and `bytes_per_token`. See each script's module docstring for
the full group/KPI reference.

---

## Extending the simulator

The full step-by-step guides live in [`documentation/`](documentation):

- **[Adding a new hardware unit](documentation/01-adding-a-new-unit.md)** — the
  `Unit` contract, the `OP_START → OP_DONE → notify_done()` protocol, registering
  it in `sim_main.cpp`, and testing it.
- **[Adding a new operation](documentation/02-adding-a-new-operation.md)** — writing
  and registering an `OpHandler`, reading params, reserving a unit, building a typed
  payload, and feeding the roofline metrics.
- **[How the simulator engine works](documentation/03-simulator-engine.md)** —
  events, the min-heap, units as latency models, the Scheduler/DAG (Kahn's
  algorithm), the LLaMA builder + Tiler, and the performance/memory engineering.
- **[Creating schedules (incl. ONNX models)](documentation/04-creating-schedules.md)**
  — authoring workloads for models other than LLaMA: the instruction format, the op
  vocabulary, hand-written YAML, a C++ builder, and a Python ONNX-graph → schedule
  generator.
- **[Writing a C++ schedule generator](documentation/05-cpp-schedule-generator.md)** —
  build a programmatic generator like `llama_schedule.cpp` from scratch: the
  `Builder`, `append_*` leaves, config + YAML loader, `sim_main` wiring, and the
  RAM/perf habits, with a complete worked MLP generator.

The short version:

**Add a hardware unit** — copy `src/units/delay_unit.{h,cpp}`, implement `handle()`
(on `OP_START` compute latency + schedule `OP_DONE`; on `OP_DONE` call
`notify_done()`); add the `.cpp` to `SIM_CORE_SOURCES` in
[`src/CMakeLists.txt`](src/CMakeLists.txt); instantiate + `register_unit()` it and
add a `dynamic_cast` branch in `wire_units()` in
[`apps/sim_main.cpp`](apps/sim_main.cpp).

**Add an op** — write `void my_op(const IssueCtx&)` and `reg.register_op("my_op",
my_op)` inside `register_builtin_ops()`
([`op_handlers.cpp`](src/schedule/op_handlers.cpp)); reference `op: my_op` in any
schedule.

**Add a YAML schedule op:**
```yaml
- id: 5
  op: my_op
  unit: my_unit
  params: { bytes: 65536 }
  depends_on: [3, 4]
  label: "my description"
```

**Add a test** — create `tests/test_<thing>.cpp` with a `TEST_CASE`, list it in
[`tests/CMakeLists.txt`](tests/CMakeLists.txt), then `cmake --build build &&
ctest --test-dir build`.

---

## Documentation & further reading

| Doc | Contents |
|---|---|
| [`documentation/`](documentation) | Developer guides: adding a unit, adding an op, engine internals. |
| [CLAUDE.md](CLAUDE.md) | Architecture deep-dive: layer-by-layer design, every latency formula, modeling knobs, key-files map. |
| [VALIDATION.md](VALIDATION.md) | Schedule/TPU-timing fidelity, MAC/cycle self-consistency, RAM/timing optimization tables. |
| [PLAN.md](PLAN.md) / [AGENTS.md](AGENTS.md) | Live worklog (verified state, completed work, remaining issues) + the agent update workflow. |