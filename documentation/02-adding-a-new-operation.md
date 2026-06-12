# Guide 2 — Adding a New Callable Operation

An **operation** ("op") is the bridge between a *schedule* (a list of
instructions) and the *hardware units* that consume time. When the scheduler
decides an instruction is ready to run, it looks up that instruction's `op` name
in the `OpRegistry` and calls the registered **handler**. The handler's job is to
translate the instruction's open-ended `params` into a concrete, typed event
targeting a unit.

This guide explains the op pipeline end-to-end and shows how to add a new op that
any schedule (hand-written YAML, workload, or the programmatic LLaMA builder) can
call by name.

> Prerequisite: an op fires events at a **unit**. If your op needs a unit that
> doesn't exist yet, read [01-adding-a-new-unit.md](01-adding-a-new-unit.md)
> first. Many ops, though, reuse the existing systolic / dma / vector / access
> units.

---

## The op pipeline at a glance

```
Schedule (instructions)            OpRegistry                Unit
┌─────────────────────┐   op name  ┌──────────────┐  event   ┌──────────────┐
│ {id, op, unit,      │──────────▶ │ handler(ctx) │ ───────▶ │  handle()    │
│  params, deps...}   │            │  - read params         │  OP_START →  │
└─────────────────────┘            │  - reserve unit         │  OP_DONE     │
        ▲                          │  - build payload        └──────┬───────┘
        │ notify_done()            │  - engine.schedule()           │
        └────────────────────────── Scheduler ◀──────────────────────┘
```

1. **`Scheduler::try_issue(id)`** ([scheduler.cpp](../src/schedule/scheduler.cpp))
   fires when an instruction's dependencies are all satisfied. It calls
   `registry_.get(inst->op.code())(IssueCtx{engine, *this, *inst})`.
2. **Your handler** receives an `IssueCtx` and does four things:
   resolve the target unit pool, compute a latency, **reserve** a unit for that
   duration, and **schedule** an `OP_START` event carrying a typed payload.
3. The **unit's `handle()`** later schedules `OP_DONE` and calls
   `notify_done()`, which unblocks the instruction's dependents.

The handler runs at **issue time** (potentially far ahead of the simulated
cycle). The reservation it makes is what assigns the op its actual start cycle.

---

## Key types you'll touch

### `IssueCtx` — what your handler receives
From [op_registry.h](../src/schedule/op_registry.h):

```cpp
struct IssueCtx {
    EventEngine&       engine;     // schedule events, accumulate metrics, find units
    Scheduler&         scheduler;  // reserve_unit_pool()
    const Instruction& inst;       // id, op, unit, params, (label)
};
using OpHandler = std::function<void(const IssueCtx&)>;
```

### `Instruction` and `ParamMap` — your input
From [instruction.h](../src/schedule/instruction.h). `params` is a `ParamMap`, a
flat vector-backed map of `KeyStr → ParamVal`. You read it with the free helper
functions — **never** index it directly:

```cpp
int64_t     pget_int (const ParamMap&, "key", int64_t def = 0);
double      pget_dbl (const ParamMap&, "key", double  def = 0.0);
std::string pget_str (const ParamMap&, "key", "" );
bool        pget_bool(const ParamMap&, "key", false);
```

These tolerate missing keys (return the default) and do int↔double coercion, so
they're robust against YAML that writes `4` vs `4.0`.

> **Minimal-mode caveat.** When `--no-trace` is on, the LLaMA builder drops
> *string* params (buffer names) and labels to save RAM (see
> [Guide 3](03-simulator-engine.md)). So a handler must derive its **timing**
> purely from **numeric** params (`M`, `K`, `N`, `rows`, `cols`, `length`, …).
> String params like `source`/`destination` are for the trace log and the
> (currently dead) data path — never let latency depend on them.

### `resolve_dim()` — symbolic-or-numeric dimensions
From [op_handlers.cpp](../src/schedule/op_handlers.cpp). Many ops accept a
dimension that may be a literal int *or* a symbolic name resolved against the arch
config:

```cpp
uint32_t resolve_dim(const ParamMap& p, const std::string& key,
                     const ArchConfig& arch, uint32_t def = 0);
// "Br"/"Bc" → systolic.rows ; "d_k"/"d_head" → systolic.d_head ;
// "hidden_dim" → rows*d_head ; a non-negative int → that int ; else def.
```

### Reservation — picking the unit and the start cycle
`ctx.scheduler.reserve_unit_pool(targets, duration, buffer_bytes = 0)` returns a
`UnitReservation { UnitId id; Cycle start; }`. It walks the candidate units and
picks the one that is **least-loaded on buffer, then earliest-free** (full policy
in [event_engine.cpp](../src/core/event_engine.cpp) `reserve_unit_pool`), marks it
busy for `duration` cycles, and accumulates `busy_cycles` for the P0.2 utilization
metric. **The reservation `duration` must equal the latency the unit will use for
`OP_DONE`** — otherwise utilization and completion desync.

---

## Anatomy of a handler (annotated)

Here is the built-in `init_fill` handler, line-by-line — it's the cleanest
complete template:

```cpp
reg.register_op("init_fill", [arch](const IssueCtx& ctx) {
    // 1. Resolve the target unit pool from the instruction's `unit` field.
    auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
    if (targets.empty()) throw std::runtime_error("init_fill: unknown unit");

    // 2. Read params (numeric → timing; strings → symbolic).
    const auto& p = ctx.inst.params;
    uint32_t rows = resolve_dim(p, "rows", arch);
    uint32_t cols = resolve_dim(p, "cols", arch);
    uint32_t len  = resolve_dim(p, "length", arch);

    // 3. Build the typed payload the unit understands.
    AccessOp op;
    op.kind     = "init_fill";
    op.elements = (rows && cols) ? uint64_t(rows) * cols : uint64_t(len);
    op.dst      = pget_str(p, "destination");
    std::string iv = pget_str(p, "init_value");
    op.fill_value = (iv == "-inf") ? -INFINITY
                                   : float(pget_dbl(p, "init_value", 0.0));

    // 4. Compute latency = ceil(elements / bandwidth).
    Cycle lat = Cycle(std::ceil(double(op.elements) / arch.access_core.bandwidth));

    // 5. Reserve the earliest-free unit for `lat` cycles.
    auto res = ctx.scheduler.reserve_unit_pool(targets, lat);

    // 6. Emit OP_START at the reserved start cycle, carrying the payload.
    Event e;
    e.type    = EventType::OP_START;
    e.target  = res.id;       // chosen physical unit
    e.cycle   = res.start;    // when it can begin
    e.instr   = ctx.inst.id;  // so OP_DONE → notify_done(this id)
    e.label   = ctx.inst.label;
    e.payload = op;           // std::any — the unit any_casts it back
    ctx.engine.schedule(std::move(e));
});
```

Every handler follows this shape: **resolve unit → read params → build payload →
latency → reserve → schedule `OP_START`.** Where they differ is the latency
formula and the payload type.

---

## Worked example: adding a `bias_add` op

Goal: a new vector-core op that adds a bias vector across a matrix, costing one
pass over the elements.

### 1. Pick the unit and payload
We reuse the existing `VectorUnit`, whose payload is `VectorOp`
([vector_unit.h](../src/units/vector_unit.h)) and whose latency model is:

```
groups  = ceil(elements / simd_width)
latency = passes * groups + exp_ops * exp_latency * groups
```

A bias-add is one pass, no transcendental ops, so `passes = 1`, `exp_ops = 0`.

### 2. Register the handler
The vector ops in [op_handlers.cpp](../src/schedule/op_handlers.cpp) share a
`vector_matrix` lambda that already does all the param plumbing. Adding a new
elementwise vector op is literally one line inside `register_builtin_ops()`:

```cpp
reg.register_op("bias_add", [vector_matrix](const IssueCtx& ctx) {
    vector_matrix(ctx, "bias_add", /*passes=*/1, /*exp_ops=*/0);
});
```

That's the whole integration — `vector_matrix` reads `rows`/`cols`/`length` and
the symbolic `source`/`destination` params, builds the `VectorOp`, computes the
latency via the shared `vector_latency()`, reserves a vector core, and schedules
`OP_START`. `VectorUnit::handle()` already knows how to time any `VectorOp`.

### 3. If your op needs a *bespoke* latency or payload
When the shared lambda doesn't fit (e.g. you need a different unit, a custom
formula, or extra payload fields), write a full handler like `init_fill` above.
For example a hypothetical `bias_add` that runs on a **new** scalar unit:

```cpp
reg.register_op("bias_add", [arch](const IssueCtx& ctx) {
    auto targets = ctx.engine.find_unit_pool(ctx.inst.unit);
    if (targets.empty()) throw std::runtime_error("bias_add: unknown unit");
    const auto& p = ctx.inst.params;

    uint32_t rows = resolve_dim(p, "rows", arch);
    uint32_t cols = resolve_dim(p, "cols", arch);
    uint64_t elements = uint64_t(rows) * cols;

    ScalarOp op;                      // your unit's payload type
    op.kind = "bias_add";
    op.elements = elements;
    op.src = pget_str(p, "source");
    op.dst = pget_str(p, "destination");

    Cycle lat = Cycle(std::ceil(double(elements) / arch.access_core.bandwidth));
    auto res = ctx.scheduler.reserve_unit_pool(targets, lat);

    Event e;
    e.type = EventType::OP_START; e.target = res.id; e.cycle = res.start;
    e.instr = ctx.inst.id; e.label = ctx.inst.label; e.payload = op;
    ctx.engine.schedule(std::move(e));
});
```

### 4. Use it in a schedule

Hand-written YAML ([schedule format](../schedules/dummy_example.yaml)):

```yaml
schedule:
  - id: 7
    op: bias_add
    unit: vector_core
    params: { rows: 128, cols: 128, source: "obuf.S", destination: "obuf.S" }
    depends_on: [6]
    label: "add attention bias"
```

Or emit it from the programmatic builder
([llama_schedule.cpp](../src/schedule/llama_schedule.cpp)) via `Builder::add`:

```cpp
ParamMap p;
p["rows"] = int64_t(rows);
p["cols"] = int64_t(cols);
p["source"] = src;
p["destination"] = dst;
InstructionId bias = b.add("bias_add", "vector_core",
                           "add attention bias", std::move(p), {prev_id});
```

---

## How an op accumulates metrics (the roofline)

Handlers are also where global counters for the P0.2 roofline get incremented —
at **issue time**, because the MAC/byte totals are independent of *when* the op
runs:

- **`gemm`** calls `ctx.engine.add_macs(M*K*N)`
  ([op_handlers.cpp](../src/schedule/op_handlers.cpp)). The count is exact and
  independent of tiling.
- **`dma_load` / `dma_store`** call `ctx.engine.add_hbm_bytes(bytes)` — this is the
  off-chip traffic that feeds the memory side of the roofline.

If your new op moves off-chip bytes or does MACs that should count toward the
roofline, add the corresponding `add_hbm_bytes()` / `add_macs()` call in the
handler. If it's an on-chip elementwise op (most vector/access ops), it
contributes only unit-busy cycles (utilization), which `reserve_unit_pool`
accounts for automatically.

---

## SRAM-aware ops (optional, P1.2)

The `gemm` handler shows how to model SRAM pressure. When `arch.model_sram` is on,
it tags the payload with a working-set size and a spill penalty:

```cpp
if (arch.model_sram) {
    const uint64_t db = dtype_bytes(arch.systolic.precision);
    s.buffer_bytes   = (uint64_t(M)*K + uint64_t(K)*N + uint64_t(M)*N) * db;
    s.spill_penalty  = arch.hbm.latency_cycles;
}
```

The bytes are **not** acquired in the handler. The `SystolicUnit` calls
`engine.sram_acquire()` at `OP_START` and `engine.sram_release()` at `OP_DONE`, so
peak SRAM reflects *concurrent execution*, not the scheduler's issue-ahead. If your
op holds SRAM during execution, follow the same split: tag in the handler, acquire/
release in the unit.

---

## Registering at startup

For an op to be available to `sim_main`, it must be registered inside
`register_builtin_ops()` ([op_handlers.cpp](../src/schedule/op_handlers.cpp)),
which `sim_main` calls once:

```cpp
OpRegistry reg;
register_builtin_ops(reg, arch);   // <-- your reg.register_op(...) lives in here
```

Under the hood, `register_op` stores the handler under **both** the string name and
a 1-byte interned code ([op_registry.cpp](../src/schedule/op_registry.cpp)). The
scheduler's hot loop uses the integer-code path
(`registry_.get(inst->op.code())`) to avoid a string hash per instruction — this
matters at 11M instructions. You don't have to think about it; just know that op
names are interned `OpStr`s, so there's a soft ceiling of 255 distinct op names
(see `SmallStr` in [instruction.h](../src/schedule/instruction.h)).

For **tests** you can register ops à la carte on a local `OpRegistry` (as
[test_dummy_units.cpp](../tests/test_dummy_units.cpp) does with `delay`) — you
don't have to go through `register_builtin_ops()`.

---

## Built-in op reference

These all live in [op_handlers.cpp](../src/schedule/op_handlers.cpp). Use them as
templates and to avoid duplicating an existing capability.

| Op(s) | Unit | Payload | Latency |
|---|---|---|---|
| `delay` | any | `int64_t` | the `latency_cycles` param verbatim |
| `dma_load`, `dma_store`, `embedding_lookup` | dma | `DmaTransfer` (off-chip) | `ceil(bytes/(bw·channels))`; +`hbm.latency` occupancy when not pipelined. Adds HBM bytes. |
| `dma_stage` | dma | `DmaTransfer` (on-chip) | `ceil(elements / systolic.rows)` |
| `gemm` | systolic | `GemmShape` | `systolic_gemm_latency(...)`; adds MACs; optional SRAM tag |
| `init_fill`, `transpose`, `sram_copy`, `select_last_token`, `gather_select` | access_core | `AccessOp` | `ceil(elements / access_core.bandwidth)` |
| `kv_stage_release` | access_core | `int64_t(0)` | 0 (dependency marker only) |
| `scale`, `rowmax`, `row_reduce_sum`, `square`, `add_epsilon`, `rsqrt`, `norm_scale`, `exp_shift`, `accumulate`, `normalize`, `causal_mask`, `rope*`, `rmsnorm`, `silu`, `elementwise_mul`, `silu_mul`, `residual_add`, `softmax`, `sample_token`, `sample_top1`, `token_feedback`, `attention_merge` | vector_core | `VectorOp` | `passes·groups + exp_ops·exp_latency·groups`, `groups=ceil(elements/simd_width)` |
| `update_rowmax`, `update_rowsum`, `logsumexp` | vector_core | `VectorOp` | as above, with op-specific `passes`/`exp_ops` |

The `passes`/`exp_ops` constants per vector op are exactly the second/third
arguments to `vector_matrix(ctx, kind, passes, exp_ops)` in the registration
block — e.g. `rope` is 2 passes, `silu` is 1 pass + 1 exp op, `softmax` is 3
passes + 1 exp op.

---

## Checklist & pitfalls

**Checklist**
- [ ] `reg.register_op("name", handler)` added inside `register_builtin_ops()`.
- [ ] Handler resolves `find_unit_pool(ctx.inst.unit)` and throws on empty.
- [ ] Latency derived from **numeric** params only (minimal-mode safe).
- [ ] `reserve_unit_pool(targets, lat)` duration == the unit's `OP_DONE` latency.
- [ ] `OP_START` event sets `target`, `cycle`, `instr`, `payload` (and `label`).
- [ ] If it moves HBM bytes / does MACs: `add_hbm_bytes` / `add_macs` called.
- [ ] The target unit's `handle()` `any_cast`s the same payload type and calls
      `notify_done()` on `OP_DONE`.

**Pitfalls**
- *Reservation/latency mismatch* → utilization wrong, `OP_DONE` lands on the wrong
  cycle. Compute latency once, pass the same value to `reserve_unit_pool` and the
  payload.
- *Latency depends on a string param* → breaks under `--no-trace` (strings
  dropped). Use numeric params for timing.
- *Op name collides or you exceed 255 names* → `OpStr` interning caps at 255;
  reuse/rename rather than inventing near-duplicates.
- *Handler reserves a pool but the unit's `handle()` doesn't `notify_done`* →
  silent deadlock (`outstanding > 0`). The op and unit are two halves of one
  contract.
- *`std::any_cast<WrongType>`* in the unit → `nullptr`, latency falls to 0 / wrong
  branch. Keep payload types in sync between handler and unit.

---

## Summary

An op is a **registered function that turns one instruction into one (or more)
unit events**. The handler runs at issue time: it reads numeric params, computes a
latency, reserves a unit for that exact duration, and schedules an `OP_START`
carrying a typed payload. The unit completes it and calls `notify_done()` to walk
the DAG forward. Add ops inside `register_builtin_ops()`; reuse the `vector_matrix`
lambda for elementwise vector ops and the `init_fill`/`gemm` shape for everything
else. The scheduling machinery that calls your handler is detailed in
[Guide 3](03-simulator-engine.md).
