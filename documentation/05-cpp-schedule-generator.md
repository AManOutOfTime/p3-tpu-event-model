# Guide 5 — Writing a C++ Schedule Generator (like `llama_schedule.cpp`)

[Guide 4](04-creating-schedules.md) showed three ways to author a workload. This
guide drills into the most powerful one: a **programmatic C++ generator** — a `.cpp`
that builds a `Schedule` in nested loops from a small typed config. This is what
[llama_schedule.cpp](../src/schedule/llama_schedule.cpp) does, and it's the right
tool when you have a *family* of models (vary layers / heads / dims / tile sizes)
rather than one fixed graph.

We build a complete, working example from scratch — an **MLP-stack generator**
(`mlp_schedule.cpp` exposing `build_mlp_schedule`) — then show how to scale the same
pattern up to transformer-style workloads. Every piece mirrors the structure of the
LLaMA generator so you can read that file fluently afterward.

Prerequisites: [Guide 2](02-adding-a-new-operation.md) (ops/params) and
[Guide 3 §6](03-simulator-engine.md) (how the builder fits the engine).

---

## 1. The anatomy of a generator

Every generator in this codebase has the same five parts. Learn them once and
`llama_schedule.cpp` (and your own files) become obvious:

| Part | Role | In `llama_schedule.cpp` |
|---|---|---|
| **Config struct** | typed knobs for the model family | `LlamaScheduleConfig` ([header](../src/schedule/llama_schedule.h)) |
| **`Builder`** | accumulates instructions, hands out sequential ids, holds `minimal` | `struct Builder` (anon namespace) |
| **`append_*` leaf helpers** | each emits a few instructions for one sub-operation and **returns the id to depend on next** | `append_detailed_tiled_gemm`, `append_detailed_rmsnorm`, `append_attention`, … |
| **Top-level `build_*` fn** | normalizes the config, reserves the vector, calls leaves in loops, returns `finish(b)` | `build_llama_schedule` → `build_*_schedule` |
| **YAML loader** | turns a workload YAML into the config struct | `llama_config_from_yaml_file` |

Plus the wiring in [sim_main.cpp](../apps/sim_main.cpp) (a CLI flag + a branch in
`preprocess_schedule`) that calls your `build_*` and runs the result.

The **golden rule** that makes the whole thing compose: *every `append_*` helper
returns the `InstructionId` of its last instruction, and callers thread that id into
the `deps` of whatever comes next.* That single convention is how the dependency DAG
gets built edge-by-edge with no separate graph-construction pass.

---

## 2. The `Builder`

The `Builder` is ~30 lines and you can copy it almost verbatim from
[llama_schedule.cpp](../src/schedule/llama_schedule.cpp). It does three things:
assigns sequential ids, supports **minimal mode** (drop trace-only strings to save
RAM under `--no-trace`), and shrinks each instruction's params to fit.

```cpp
struct Builder {
    std::vector<Instruction> out;
    InstructionId next = 0;
    bool minimal = false;   // set from !trace; drops labels + string params

    InstructionId add(std::string op, std::string unit, std::string label,
                      ParamMap params = {}, std::vector<InstructionId> deps = {}) {
        Instruction inst;
        inst.id   = next++;                 // sequential ids → flat-vector scheduler
        inst.op   = std::move(op);
        inst.unit = std::move(unit);
        if (minimal) {
            // Keep numeric params (+ init_value); drop buffer-name strings & label.
            for (auto& kv : params)
                if (!kv.second.is_string() || kv.first == "init_value")
                    inst.params[kv.first] = std::move(kv.second);
            inst.params.shrink();
        } else {
            inst.label  = std::move(label);
            inst.params = std::move(params);
            inst.params.shrink();
        }
        inst.depends_on = std::move(deps);
        out.push_back(std::move(inst));
        return out.back().id;               // <-- the id callers thread into later deps
    }
};
```

Why minimal mode matters: timing reads only **numeric** params, so dropping the
symbolic `source`/`destination` strings is bit-identical for cycles/MACs/bytes but
saves gigabytes on huge schedules (see [Guide 3 §7.4](03-simulator-engine.md)).
Build it in if your generator can ever produce large schedules; it's free.

---

## 3. Leaf helpers — `append_*`

A leaf helper emits the instructions for one logical sub-operation and returns the
id to chain from. Keep them small and single-purpose. Two building blocks you'll
reuse constantly:

**A staged linear layer** `Y[m,n] = X[m,k] · W[k,n]` — weight load → stage → gemm →
bias add. This is the §3 pattern from [Guide 4](04-creating-schedules.md), wrapped
in a helper:

```cpp
// Returns the id of the final instruction (the bias add).
InstructionId append_linear(Builder& b, const MlpScheduleConfig& cfg,
                            const std::string& tag,
                            const std::string& source_x,   // activations (already on-chip)
                            const std::string& hbm_weight, // weights in HBM
                            const std::string& dst,
                            uint32_t m, uint32_t k, uint32_t n,
                            std::vector<InstructionId> deps) {
    // 1. weights HBM → IBUF  (counts HBM bytes)
    ParamMap wl;
    wl["rows"] = (int64_t)k; wl["cols"] = (int64_t)n;
    wl["source"] = hbm_weight; wl["destination"] = "ibuf." + tag + ".W";
    InstructionId w_load = b.add("dma_load", "dma", tag + " weight load", wl, deps);

    // 2. IBUF → systolic operand bus (on-chip stage)
    ParamMap st;
    st["rows"] = (int64_t)k; st["cols"] = (int64_t)n;
    st["source"] = "ibuf." + tag + ".W"; st["destination"] = "array." + tag + ".W";
    InstructionId w_stage = b.add("dma_stage", "dma", tag + " weight stage", st, {w_load});

    // 3. the matmul (counts M·K·N MACs; targets the systolic pool)
    ParamMap g;
    g["M"] = (int64_t)m; g["K"] = (int64_t)k; g["N"] = (int64_t)n;
    g["source_a"] = source_x; g["source_b"] = "array." + tag + ".W"; g["destination"] = dst;
    std::vector<InstructionId> gemm_deps = deps;     // needs X ready ...
    gemm_deps.push_back(w_stage);                    // ... and W staged
    InstructionId gemm = b.add("gemm", "systolic", tag + " gemm", g, gemm_deps);

    // 4. + bias (vector elementwise)
    ParamMap ba;
    ba["rows"] = (int64_t)m; ba["cols"] = (int64_t)n;
    ba["source_a"] = dst; ba["source_b"] = hbm_weight + ".bias"; ba["destination"] = dst;
    return b.add("residual_add", "vector_core", tag + " bias add", ba, {gemm});
}
```

**An elementwise activation** over `m·n` elements:

```cpp
InstructionId append_activation(Builder& b, const std::string& op,  // "scale" | "silu" | ...
                                const std::string& tag,
                                const std::string& buf,
                                uint32_t m, uint32_t n,
                                std::vector<InstructionId> deps) {
    ParamMap p;
    p["rows"] = (int64_t)m; p["cols"] = (int64_t)n;
    p["source"] = buf; p["destination"] = buf;
    return b.add(op, "vector_core", tag + " act", p, std::move(deps));
}
```

Note the param mechanics: build a `ParamMap`, assign numeric dims as `int64_t`
(cast explicitly — the param value is a tagged union; an unannotated literal could
pick the wrong type), and symbolic names as strings. The leaf returns one id.

> **Tiling choice.** Above, `append_linear` emits *one logical* `gemm`. On the
> `--schedule`/generator path the **Tiler** (`Tiler::expand_gemm_subtiles`) will
> subtile any GEMM whose `M`/`N` exceeds the array and rewire dependents — so you
> can emit logical sizes and let the simulator fragment them. If you'd rather emit
> hardware-sized tiles yourself (like the LLaMA builder's
> `append_detailed_tiled_gemm`, which loops `row_tiles × col_tiles`), do that and
> the Tiler becomes a no-op move-through. Either is correct; the explicit form gives
> you per-tile control of staging/placement.

---

## 4. The config struct + a `normalize`/validate step

Define a plain struct of knobs with sensible defaults. Validate and derive fields
once, up front, so the leaves can assume a clean config (mirrors
`normalize_cfg` in the LLaMA file):

```cpp
// mlp_schedule.h
#pragma once
#include "schedule/schedule.h"
#include <string>
#include <vector>

namespace sim {

struct MlpScheduleConfig {
    uint32_t batch       = 64;
    std::vector<uint32_t> layer_dims = {784, 256, 256, 10};  // [in, h1, h2, ..., out]
    std::string activation = "scale";   // "scale" (relu-class) | "silu" (gelu-class)
    bool        final_softmax = true;
    uint32_t    dtype_bytes = 2;
};

Schedule build_mlp_schedule(const MlpScheduleConfig& cfg, bool minimal = false);

MlpScheduleConfig mlp_config_from_yaml_file(const std::string& path);

}  // namespace sim
```

```cpp
// inside mlp_schedule.cpp, anon namespace
MlpScheduleConfig normalize_cfg(MlpScheduleConfig cfg) {
    if (cfg.layer_dims.size() < 2)
        throw std::runtime_error("MlpScheduleConfig: need at least [in, out]");
    if (!cfg.batch)        throw std::runtime_error("MlpScheduleConfig: batch must be > 0");
    if (!cfg.dtype_bytes)  cfg.dtype_bytes = 2;
    if (cfg.activation != "scale" && cfg.activation != "silu")
        throw std::runtime_error("MlpScheduleConfig: activation must be scale|silu");
    return cfg;
}
```

Validating here means the leaves never have to defend against zero dims or bad enum
values — they just emit instructions.

---

## 5. The top-level `build_*` function

This is where it comes together: normalize, **reserve** the vector to its final
size (the key RAM optimization), then loop the layers threading ids, and `finish`.

```cpp
// finish(): move the instructions into a Schedule. Skip validate() — a generator
// that assigns sequential ids and wires deps by construction can't produce
// duplicate ids / unknown deps / cycles, and the Scheduler rebuilds the same
// structures anyway. (YAML-loaded schedules ARE validated; generated ones need not.)
Schedule finish(Builder& b) {
    Schedule s;
    s.instructions = std::move(b.out);
    return s;
}

// Rough upper bound so the vector never re-doubles mid-build (see §6).
static size_t estimate_instruction_count(const MlpScheduleConfig& cfg) {
    // ~4 instructions per linear (load, stage, gemm, bias) + 1 activation.
    return cfg.layer_dims.size() * 6 + 8;
}

Schedule build_mlp_schedule(const MlpScheduleConfig& input_cfg, bool minimal) {
    const MlpScheduleConfig cfg = normalize_cfg(input_cfg);
    Builder b;
    b.minimal = minimal;
    b.out.reserve(estimate_instruction_count(cfg));   // ONE allocation, no doublings

    std::string act_buf = "x";                 // current activation buffer name
    std::vector<InstructionId> prev = {};      // ids the next layer depends on

    const size_t L = cfg.layer_dims.size() - 1;  // number of linear layers
    for (size_t i = 0; i < L; i++) {
        const uint32_t k = cfg.layer_dims[i];
        const uint32_t n = cfg.layer_dims[i + 1];
        const std::string tag = "fc" + std::to_string(i);
        const std::string dst = "h" + std::to_string(i);

        InstructionId lin = append_linear(
            b, cfg, tag, act_buf, "HBM.W" + std::to_string(i), dst,
            cfg.batch, k, n, prev);

        const bool last = (i == L - 1);
        if (!last) {
            InstructionId a = append_activation(b, cfg.activation, tag, dst,
                                                cfg.batch, n, {lin});
            prev = {a};
        } else if (cfg.final_softmax) {
            InstructionId sm = append_activation(b, "softmax", tag, dst,
                                                 cfg.batch, n, {lin});
            prev = {sm};
        } else {
            prev = {lin};
        }
        act_buf = dst;
    }
    return finish(b);
}
```

That's a complete generator. `build_mlp_schedule({.batch=64, .layer_dims={784,256,256,10}})`
produces the same workload as the hand-written YAML in
[Guide 4 §4](04-creating-schedules.md#4-worked-example-a--a-tiny-mlp-by-hand) — but
parameterized, so you can sweep batch size or depth from one call.

---

## 6. RAM & performance — the four things `llama_schedule.cpp` does

At millions of instructions these stop being micro-optimizations and become the
difference between minutes and never-finishing. Bake them in from the start; they
cost nothing on small schedules:

1. **`out.reserve(estimate_instruction_count(cfg))`.** A `vector` growing to N
   elements doubles ~log₂N times, and glibc never returns the freed intermediate
   buffers to the OS (they're interleaved with live param data). One upfront
   reservation = one allocation, zero doublings, zero heap holes. Run the estimate
   ~10% hot so you never under-reserve and fall back to doubling.
2. **Minimal mode** (`b.minimal = !trace`). Drops labels + string params under
   `--no-trace`. Timing-neutral, large RAM win.
3. **`params.shrink()`** after each `add` (already in the `Builder`). Each
   `ParamMap::operator[]` can double capacity, leaving up to 50% slack that would
   persist for the whole run.
4. **Skip `validate()` in `finish()`.** A correct-by-construction generator can't
   create the errors `validate()` checks for, and the Scheduler rebuilds equivalent
   structures anyway — validating an 11 M-node graph twice is wasted minutes. (YAML
   schedules still get validated, because humans make mistakes.)

For the full accounting — flat-vector scheduler, `SmallStr`/`CompactParamVal`
struct shrink, etc. — see [Guide 3 §7](03-simulator-engine.md#7-performance--memory-engineering-10-min--4-min-9-gb--7-gb). Those live in the engine; the four
above are what *your generator* controls.

---

## 7. The YAML loader

Map a workload YAML to your config struct with yaml-cpp. A tiny `read_scalar`
helper keeps it terse (the LLaMA loader uses the same shape):

```cpp
#include <yaml-cpp/yaml.h>

namespace {
template <typename T>
void read_scalar(const YAML::Node& n, const char* key, T& dst) {
    if (n[key]) dst = n[key].as<T>();
}
}  // namespace

MlpScheduleConfig mlp_config_from_yaml_file(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    const YAML::Node n = root["mlp"] ? root["mlp"] : root;   // accept a top-level 'mlp:' block
    MlpScheduleConfig cfg;
    read_scalar(n, "batch", cfg.batch);
    read_scalar(n, "activation", cfg.activation);
    read_scalar(n, "final_softmax", cfg.final_softmax);
    read_scalar(n, "dtype_bytes", cfg.dtype_bytes);
    if (n["layer_dims"]) {
        cfg.layer_dims.clear();
        for (auto d : n["layer_dims"]) cfg.layer_dims.push_back(d.as<uint32_t>());
    }
    return cfg;
}
```

Example workload `workloads/mlp_demo.yaml`:

```yaml
mlp:
  batch: 64
  layer_dims: [784, 256, 256, 10]
  activation: scale       # relu-class
  final_softmax: true
  dtype_bytes: 2
```

---

## 8. Wiring it into `sim_main`

Three edits in [apps/sim_main.cpp](../apps/sim_main.cpp), parallel to the existing
`--llama-workload` path.

**(a) Parse a CLI flag** in the `main()` arg loop:

```cpp
std::string mlp_path = "";
...
else if (a == "--mlp-workload" && i+1 < argc) mlp_path = argv[++i];
```

**(b) A branch in `preprocess_schedule`** (add a param + an `if`):

```cpp
if (!mlp_path.empty()) {
    MlpScheduleConfig cfg = mlp_config_from_yaml_file(mlp_path);
    cfg.dtype_bytes = precision_bytes(arch.systolic.precision);  // match the HW precision
    Schedule raw = build_mlp_schedule(cfg, /*minimal=*/!trace);
    r.schedule = Tiler::expand_gemm_subtiles(std::move(raw), arch);  // subtile big GEMMs
    r.used_llama = false;  // (or add a r.used_mlp flag for the metrics footer)
} else if (!llama_path.empty()) {
    ...
```

Include your header (`#include "schedule/mlp_schedule.h"`) at the top and add
`schedule/mlp_schedule.cpp` to `SIM_CORE_SOURCES` in
[src/CMakeLists.txt](../src/CMakeLists.txt) — the only build wiring needed.

**(c)** Optionally extend the metrics footer (TTFT/throughput) for your token/sample
notion, like the LLaMA path does. For an MLP, MACs + HBM bytes + utilization already
tell the whole roofline story, so this is optional.

Run it:

```bash
./build/apps/sim_main --config configs/default.yaml \
    --mlp-workload workloads/mlp_demo.yaml --no-trace
```

> **Don't want to touch `sim_main`?** Generators are just library functions. A test
> (or a throwaway `main`) can call `build_mlp_schedule(cfg)` directly and feed the
> result to a `Scheduler` — see §9. The `sim_main` wiring is only for driving it
> from the CLI.

---

## 9. Testing the generator

Generators are tested two ways, both in the doctest style of
[tests/test_llama_schedule.cpp](../tests/test_llama_schedule.cpp): **structural**
assertions on the emitted `Schedule`, and an **end-to-end** run through real units.

Reusable structural helpers (copy from `test_llama_schedule.cpp`):

```cpp
int count_op(const Schedule& s, const std::string& op) {
    return (int)std::count_if(s.instructions.begin(), s.instructions.end(),
        [&](const Instruction& i){ return i.op == op; });
}
```

```cpp
#include <doctest/doctest.h>
#include "schedule/mlp_schedule.h"
#include "schedule/op_handlers.h"
#include "schedule/scheduler.h"
#include "core/event_engine.h"
#include "units/systolic_unit.h"   // + dma/vector/access units
using namespace sim;

TEST_CASE("MLP generator: one gemm per linear layer") {
    MlpScheduleConfig cfg;
    cfg.layer_dims = {784, 256, 256, 10};   // 3 linear layers
    Schedule s = build_mlp_schedule(cfg);
    REQUIRE(count_op(s, "gemm") == 3);
    REQUIRE(count_op(s, "dma_load") == 3);  // one weight load each
    REQUIRE(count_op(s, "softmax") == 1);   // final
}

TEST_CASE("MLP generator: schedule runs to completion") {
    ArchConfig arch = ArchConfig::from_yaml_string("systolic: { rows: 256, cols: 256 }");
    EventEngine engine(arch.clock_ghz);
    // register systolic_/dma_/vector_core_/access_core_ pools (see test_llama_schedule.cpp)
    register_real_units(engine, arch);

    OpRegistry reg; register_builtin_ops(reg, arch);
    MlpScheduleConfig cfg; cfg.batch = 8; cfg.layer_dims = {64, 64, 16};
    Scheduler sched(engine, reg, build_mlp_schedule(cfg));
    wire_units(engine, sched);              // inject Scheduler* into every unit

    sched.launch();
    engine.run();
    REQUIRE(sched.all_done());              // outstanding == 0 → no broken deps
}
```

Add `test_mlp_schedule.cpp` to `SIM_TEST_SOURCES` in
[tests/CMakeLists.txt](../tests/CMakeLists.txt). The first test guards the
*structure* (right op counts, dims, dependency edges); the second guards that the
DAG is *runnable* (every instruction's deps eventually complete). Both are fast and
catch the common breakages.

---

## 10. Scaling the pattern up (toward `llama_schedule.cpp`)

The MLP generator is the whole skeleton; a transformer generator is the same
skeleton with richer leaves. To grow it:

- **Add an attention leaf.** `append_attention(b, cfg, ...)` emits Q/K/V projections
  (`append_linear`), then per Q/KV tile the score `gemm` → `softmax`
  (+ `causal_mask` only if autoregressive) → context `gemm`, then the output
  projection. The LLaMA `append_attention` is the reference; a BERT encoder just
  drops the causal pieces (see [Guide 4 §6](04-creating-schedules.md)).
- **Add a normalization leaf.** `append_layernorm` → `square` → `row_reduce_sum` →
  `add_epsilon` → `rsqrt` → `norm_scale` (the LLaMA `append_detailed_rmsnorm`
  pattern), or collapse to a single `rmsnorm` op when you don't need the breakdown.
- **Compose layers.** An `append_encoder_layer` calls attention → residual_add →
  layernorm → FFN (`append_linear` → `silu` → `append_linear`) → residual_add →
  layernorm, threading ids through. `build_*` loops it `num_layers` times.
- **Pick your tiling granularity** per leaf (logical-GEMM-let-the-Tiler-split vs.
  emit-hardware-tiles), exactly as in §3.

The structure never changes: **config → Builder → leaves that return ids → top-level
loop → finish.** Once an MLP generator compiles and runs, a transformer generator is
just more (and bigger) `append_*` helpers.

---

## Pitfalls & checklist

**Pitfalls**
- *Forgetting to thread the returned id into the next `deps`* → instructions run
  out of order / concurrently when they shouldn't (or the run still completes but
  the timing is wrong). The returned id is the contract — always chain it.
- *Unannotated numeric literals in `ParamMap`* → cast dims to `int64_t` explicitly;
  the param value is a tagged union and an `int` vs `double` mismatch changes how
  `pget_int`/`resolve_dim` read it.
- *Reserving too small* → the vector re-doubles and you lose the RAM win. Estimate
  hot.
- *Calling `validate()` on a generated schedule* → wasted time at scale; rely on
  correct-by-construction + the Scheduler's rebuild.
- *Latency depending on a string param* → breaks minimal mode (strings are dropped).
  Keep all timing in numeric params.

**Checklist**
- [ ] Config struct with defaults + a `normalize_cfg` that validates/derives once.
- [ ] `Builder` with sequential ids, `minimal` mode, and `params.shrink()`.
- [ ] Leaf `append_*` helpers, each returning the id to chain from.
- [ ] Top-level `build_*` that `reserve()`s, loops, threads ids, and `finish()`es.
- [ ] YAML loader (`*_config_from_yaml_file`).
- [ ] `src/CMakeLists.txt` updated; `sim_main` flag + `preprocess_schedule` branch
      (or call it from a test).
- [ ] `Tiler::expand_gemm_subtiles` applied so oversized GEMMs subtile.
- [ ] Structural + end-to-end tests; `all_done()` true.

---

## Summary

A C++ schedule generator is five composable parts — **a config struct, a `Builder`,
`append_*` leaves that each return the id to depend on next, a top-level loop, and a
YAML loader** — wired into `sim_main` with one flag and one branch. The MLP generator
above is a complete, runnable instance of that template; `llama_schedule.cpp` is the
same template with transformer-sized leaves. Bake in the four RAM/perf habits
(`reserve`, minimal mode, `shrink`, skip `validate`) from day one and your generator
scales from a 10-instruction MLP to an 11-million-instruction LLaMA prefill without
changing shape. For how the resulting schedule becomes a cycle count, see
[Guide 3](03-simulator-engine.md); for the op vocabulary the leaves emit, see
[Guide 2](02-adding-a-new-operation.md).
