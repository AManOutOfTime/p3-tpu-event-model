# Guide 4 — Creating Schedules (and mapping real models, e.g. from ONNX)

A **schedule** is the workload the simulator runs: a list of instructions wired
into a dependency DAG. The built-in `--llama-workload` path generates one for
LLaMA-style transformers, but the simulator is not LLaMA-specific — any model whose
work is dominated by **matmuls, elementwise ops, reductions, and data movement**
can be expressed as a schedule. This guide shows how, ending with a concrete
workflow for turning an **ONNX model** (from
[github.com/onnx/models](https://github.com/onnx/models)) into a runnable schedule.

Prerequisites: skim [02-adding-a-new-operation.md](02-adding-a-new-operation.md)
(the op vocabulary) and [03-simulator-engine.md](03-simulator-engine.md) (how a
schedule becomes events). This guide builds on both.

---

## 0. What "creating a schedule" actually means here

The simulator is **timing-only** — it never multiplies real matrices. So porting a
model is *not* about reproducing its numbers; it's about reproducing **where the
time and traffic go**:

- **Compute** — every matmul/conv becomes a `gemm` whose `M·K·N` is the real MAC
  count. This feeds the compute side of the roofline and the systolic array's
  occupancy.
- **Memory** — every weight/activation transfer from HBM becomes a `dma_load` /
  `dma_store` whose byte count is real. This feeds the memory side of the roofline.
- **Everything else** — activations, normalizations, pooling, softmax — becomes a
  vector or access op whose element count and *latency class* (how many passes,
  whether it uses transcendentals) match the real operator.

Get those three right and the cycle count, utilization, and
compute-vs-memory verdict are meaningful — which is the entire point of the tool.

### Three ways to author a schedule

| Approach | Best for | Mechanism |
|---|---|---|
| **Hand-written YAML** (`--schedule`) | small / exact / didactic workloads | write `schedule:` instructions directly |
| **Programmatic C++ builder** | parametric model *families* (like LLaMA) | a `Builder` that emits instructions in loops ([llama_schedule.cpp](../src/schedule/llama_schedule.cpp)) |
| **External generator → YAML** | arbitrary real models (ONNX, etc.) | a Python script walks the model graph and emits a `schedule:` YAML |

We cover all three, in increasing order of automation.

---

## 1. The instruction format (precise)

A schedule is a YAML document with one top-level `schedule:` sequence
([schedule.cpp](../src/schedule/schedule.cpp) parses it). Each item:

```yaml
- id: 7                      # optional; auto-increments from the previous id if omitted
  op: gemm                   # REQUIRED — must be a registered op name
  unit: systolic             # logical pool name (systolic | dma | vector_core | access_core)
  params: { M: 512, K: 512, N: 512, source_a: "A", source_b: "B", destination: "C" }
  depends_on: [5, 6]         # ids that must complete first (the DAG edges)
  label: "fc1 matmul"        # optional, trace-only
```

Rules enforced by `Schedule::validate()`:

- **ids must be unique**, every `depends_on` id must exist, and the graph must be
  **acyclic** (checked with Kahn's algorithm).
- Instructions with no unmet deps start immediately and concurrently; the only
  serialization beyond `depends_on` is hardware contention (one systolic array
  serializes its GEMMs even if they're independent).

**Param conventions** (read with `pget_*` / `resolve_dim` in
[op_handlers.cpp](../src/schedule/op_handlers.cpp)):

- Dimensions (`M`, `K`, `N`, `rows`, `cols`, `length`, `input_rows`, `input_cols`)
  may be integers **or** symbolic names resolved against the arch config: `Br`/`Bc`
  → `systolic.rows`; `d_k`/`d_head` → `systolic.d_head`; `hidden_dim` →
  `rows·d_head`.
- `source*` / `destination` are **symbolic buffer names** — used only for the trace
  log and dependency readability. Latency never depends on them, so keep all timing
  in the numeric params (this is also why `--no-trace` can drop them).

---

## 2. The op vocabulary, by what it models

Pick the sim op whose **unit + latency class** matches the model operator. Full
list and formulas live in [Guide 2 §"Built-in op reference"](02-adding-a-new-operation.md#built-in-op-reference); here it is organized by *what you're porting*:

| Model operator | Sim op | Unit | Key params |
|---|---|---|---|
| MatMul / Gemm / Linear / fully-connected | `gemm` | systolic | `M`, `K`, `N` |
| Conv (via im2col — see §5) | `gemm` (+ an access op for im2col) | systolic | `M=batch·H'·W'`, `K=Cin·Kh·Kw`, `N=Cout` |
| Load weights / activations from HBM | `dma_load` | dma | `rows`,`cols` (or `length`) |
| Store results to HBM | `dma_store` | dma | `rows`,`cols` (or `length`) |
| Stage operand into the array | `dma_stage` | dma | `rows`,`cols` |
| Relu / Add / Mul / Clip / LeakyRelu (1 pass, no exp) | `scale` or `residual_add` | vector_core | `rows`,`cols` (or `length`) |
| Sigmoid / Tanh / GELU / Erf / SiLU (uses exp) | `silu` | vector_core | `rows`,`cols` |
| Softmax | `softmax` | vector_core | `rows`,`cols` |
| LayerNorm / BatchNorm / RMSNorm | `rmsnorm` | vector_core | `rows`,`cols` |
| MaxPool / AvgPool / GlobalAvgPool / ReduceMean | `row_reduce_sum` | vector_core | `length` = elements reduced |
| Transpose / im2col / layout shuffle | `transpose` | access_core | `input_rows`,`input_cols` |
| Reshape / Flatten / Concat / Squeeze (layout copy) | `sram_copy` | access_core | `rows`,`cols` (or `length`) |
| Constant init / mask fill | `init_fill` | access_core | `rows`,`cols`, `init_value` |
| Attention (QK·softmax·PV) | the FA2 op chain | mixed | see §6 |

> Two important auto-behaviors: **`gemm` always targets the `systolic` pool**
> (the handler hardcodes `find_unit_pool("systolic")`), regardless of the `unit:`
> field — but write `unit: systolic` for clarity. And on the `--schedule` path the
> **Tiler** automatically subtiles any `gemm` whose `M` or `N` exceeds the array,
> rewiring dependents — so you may write *logical* GEMM sizes and let the simulator
> fragment them.

If a model operator has no good match, either approximate it with the nearest
latency class **or** register a new op (one `reg.register_op(...)` line — see
[Guide 2](02-adding-a-new-operation.md)).

---

## 3. The canonical "one layer" pattern

Most layers reduce to: **bring weights on-chip → matmul → activation**. The fully
faithful version (what the LLaMA builder emits) for a linear layer
`Y[M,N] = X[M,K] · W[K,N]`:

```yaml
- id: 0
  op: dma_load            # weights HBM → IBUF   (counts HBM bytes)
  unit: dma
  params: { rows: 1024, cols: 4096, source: "HBM.W", destination: "ibuf.W" }
- id: 1
  op: dma_stage           # IBUF → systolic operand bus  (on-chip)
  unit: dma
  params: { rows: 1024, cols: 4096, source: "ibuf.W", destination: "array.W" }
  depends_on: [0]
- id: 2
  op: gemm                # Y = X · W            (counts M·K·N MACs)
  unit: systolic
  params: { M: 512, K: 1024, N: 4096, source_a: "ibuf.X", source_b: "array.W", destination: "obuf.Y" }
  depends_on: [1]
- id: 3
  op: residual_add        # + bias / skip connection (vector)
  unit: vector_core
  params: { rows: 512, cols: 4096, source_a: "obuf.Y", source_b: "HBM.bias", destination: "obuf.Y" }
  depends_on: [2]
```

The **minimal faithful** version, if you only care about the roofline (MACs +
bytes) and not the staging detail, collapses to two instructions —
`dma_load(weights)` then `gemm` — because `dma_load` is what counts HBM bytes and
`gemm` is what counts MACs. Start minimal; add staging/activation detail when you
need the extra fidelity.

---

## 4. Worked example A — a tiny MLP by hand

A 3-layer MLP: `784 → 256 → 256 → 10`, batch 64, ReLU between layers. Save as
`schedules/mlp.yaml`:

```yaml
# 3-layer MLP, batch=64. GEMM dims: M=batch, K=in, N=out.
schedule:
  # ---- layer 1: 784 -> 256 ----
  - { id: 0, op: dma_load, unit: dma,
      params: { rows: 784, cols: 256, source: "HBM.W1", destination: "ibuf.W1" },
      label: "load W1" }
  - { id: 1, op: gemm, unit: systolic,
      params: { M: 64, K: 784, N: 256, source_a: "x", source_b: "ibuf.W1", destination: "h1" },
      depends_on: [0], label: "fc1" }
  - { id: 2, op: scale, unit: vector_core,
      params: { rows: 64, cols: 256, source: "h1", destination: "h1" },
      depends_on: [1], label: "relu1" }       # ReLU ~ 1 vector pass, no exp

  # ---- layer 2: 256 -> 256 ----
  - { id: 3, op: dma_load, unit: dma,
      params: { rows: 256, cols: 256, source: "HBM.W2", destination: "ibuf.W2" },
      label: "load W2" }
  - { id: 4, op: gemm, unit: systolic,
      params: { M: 64, K: 256, N: 256, source_a: "h1", source_b: "ibuf.W2", destination: "h2" },
      depends_on: [2, 3], label: "fc2" }
  - { id: 5, op: scale, unit: vector_core,
      params: { rows: 64, cols: 256, source: "h2", destination: "h2" },
      depends_on: [4], label: "relu2" }

  # ---- layer 3: 256 -> 10 ----
  - { id: 6, op: dma_load, unit: dma,
      params: { rows: 256, cols: 10, source: "HBM.W3", destination: "ibuf.W3" },
      label: "load W3" }
  - { id: 7, op: gemm, unit: systolic,
      params: { M: 64, K: 256, N: 10, source_a: "h2", source_b: "ibuf.W3", destination: "logits" },
      depends_on: [5, 6], label: "fc3" }
  - { id: 8, op: softmax, unit: vector_core,
      params: { rows: 64, cols: 10, source: "logits", destination: "probs" },
      depends_on: [7], label: "softmax" }
```

Run it and read the metrics:

```bash
./build/apps/sim_main --config configs/default.yaml --schedule schedules/mlp.yaml
```

Sanity-check the roofline by hand: total MACs = `64·784·256 + 64·256·256 + 64·256·10`
= 12.85M + 4.19M + 0.16M ≈ **17.2 M MACs**, which should match the `MACs=` line in
`== metrics ==`. The `depends_on` edges make the three layers serial; the weight
loads can overlap the previous layer's compute because they only depend on nothing
(W1) or are free to be reserved early.

---

## 5. Worked example B — a Conv layer as a GEMM (im2col)

The simulator has no `conv` op, and that's fine: in real accelerators convolution
*is* a GEMM after **im2col** (unfolding each output pixel's receptive field into a
row). For `Conv(N=1, Cin=64, H=W=56, kernel 3×3, stride 1, pad 1, Cout=64)`:

- Output spatial size `H'=W'=56`.
- GEMM dims: `M = N·H'·W' = 1·56·56 = 3136`, `K = Cin·Kh·Kw = 64·9 = 576`,
  `N_out = Cout = 64`.

```yaml
schedule:
  - { id: 0, op: dma_load, unit: dma,
      params: { rows: 576, cols: 64, source: "HBM.conv_W", destination: "ibuf.W" },
      label: "load conv weights [K=576 x Cout=64]" }
  - { id: 1, op: transpose, unit: access_core,
      params: { input_rows: 3136, input_cols: 576, source: "HBM.act", destination: "ibuf.im2col" },
      label: "im2col unfold [M=3136 x K=576]" }   # gather cost ~ M*K elements
  - { id: 2, op: gemm, unit: systolic,
      params: { M: 3136, K: 576, N: 64, source_a: "ibuf.im2col", source_b: "ibuf.W", destination: "obuf.Y" },
      depends_on: [0, 1], label: "conv as GEMM" }
  - { id: 3, op: silu, unit: vector_core,
      params: { rows: 3136, cols: 64, source: "obuf.Y", destination: "obuf.Y" },
      depends_on: [2], label: "activation (GELU-class)" }
```

`M=3136 > array rows (256)`, so the Tiler will fragment this GEMM into
`ceil(3136/256)=13` row-tiles automatically on the `--schedule` path — you wrote the
logical conv, the simulator handles hardware tiling. The im2col is modeled as an
access-core gather over `M·K` elements; if your activations are already resident
you can drop instruction 1 and have the GEMM depend only on the weight load.

> **Pooling, BN, etc.** A following `MaxPool 2×2` → `row_reduce_sum` over the
> pooled elements; `BatchNormalization` → `rmsnorm` over `H'·W'·Cout`. The exact op
> name matters less than the unit and element count.

---

## 6. Non-LLaMA attention (e.g. a BERT encoder)

A BERT-style **encoder** layer is structurally close to LLaMA's decoder layer but
**non-causal** (every token attends to every token — no causal mask, no KV cache)
and uses standard MHA (no GQA). Per attention head with sequence length `S` and head
dim `d`:

1. `gemm` Q·Kᵀ → scores `[S,S]` (`M=S, K=d, N=S`)
2. `softmax` over `[S,S]` (no `causal_mask` op — that's the only structural
   difference from the LLaMA chain)
3. `gemm` scores·V → `[S,d]` (`M=S, K=S, N=d`)

Then the FFN is two linears with a GELU between (the §3 pattern, with `silu` for
GELU). You can hand-write one encoder layer in YAML by following §3 + the three
steps above, or — if you want a *parametric* BERT family (vary layers/heads/seq) —
add a `build_bert_schedule()` next to `build_llama_schedule()` in
[llama_schedule.cpp](../src/schedule/llama_schedule.cpp) reusing the same `Builder`
and `append_*` helpers, minus the causal/KV-cache pieces. The LLaMA builder's
`append_attention` is the template; dropping `causal_block_skip` and the
`causal_mask` emission gives you encoder attention.

---

## 7. Mapping an ONNX model → schedule (the principle)

[ONNX](https://github.com/onnx/models) distributes hundreds of real models
(ResNet, MobileNet, BERT, GPT-2, …) as `.onnx` graphs. An ONNX graph is a list of
**nodes**, each with an `op_type` (`Conv`, `MatMul`, `Gemm`, `Relu`, `Add`,
`Softmax`, …), input/output tensor names, and attributes. With **shape inference**
you also get every tensor's shape. That is exactly the information a schedule needs:

```
ONNX node  →  (op_type, input shapes, output shape, attributes)
            →  one or more sim instructions with matching dims + deps
```

The mapping is a function from `op_type` + shapes to sim ops, using the table in §2.
Dependencies come for free: an ONNX node depends on the nodes that produced its
input tensors, so you track `producer_id[tensor]` and set each instruction's
`depends_on` to the producers of its inputs.

**What to preserve:** MACs (from GEMM/Conv dims), HBM bytes (from weight/activation
loads), and the elementwise/reduction element counts. **What to drop:** numerical
values, exotic ops you can approximate, and dynamic shapes (fix the batch size /
run shape inference so every dim is concrete).

---

## 8. Worked example C — an ONNX → schedule generator (Python)

Below is a self-contained starting point. It loads an ONNX model, runs shape
inference, walks the graph in topological order, maps each node to sim
instruction(s), and writes a `schedule:` YAML the simulator can run. It depends only
on the `onnx` package (`pip install onnx`), not onnxruntime.

```python
#!/usr/bin/env python3
"""onnx_to_schedule.py — turn an ONNX model into a sim schedule YAML.

Usage:
    python3 onnx_to_schedule.py resnet18.onnx schedules/resnet18.yaml --batch 1

Then:
    ./build/apps/sim_main --config configs/default.yaml \
        --schedule schedules/resnet18.yaml --no-trace
"""
import argparse, math, sys
import onnx
from onnx import shape_inference

# --- shape bookkeeping ------------------------------------------------------
def collect_shapes(graph):
    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = [d.dim_value or 0 for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    for init in graph.initializer:          # weights / constants
        shapes[init.name] = list(init.dims)
    return shapes

def numel(shape):
    n = 1
    for d in shape:
        n *= max(int(d), 1)
    return n

# --- the schedule emitter ---------------------------------------------------
class Sched:
    def __init__(self):
        self.items = []
        self.next_id = 0
        self.producer = {}          # tensor name -> instruction id that wrote it

    def add(self, op, unit, params, inputs, label=""):
        deps = sorted({self.producer[t] for t in inputs if t in self.producer})
        iid = self.next_id; self.next_id += 1
        self.items.append((iid, op, unit, params, deps, label))
        return iid

    def emit(self, outputs, *args, **kw):
        iid = self.add(*args, **kw)
        for t in outputs:           # this instr now produces these tensors
            self.producer[t] = iid
        return iid

    def to_yaml(self):
        lines = ["schedule:"]
        for iid, op, unit, params, deps, label in self.items:
            p = ", ".join(f"{k}: {v}" for k, v in params.items())
            lines.append(f"  - {{ id: {iid}, op: {op}, unit: {unit},")
            lines.append(f"      params: {{ {p} }},")
            if deps:  lines.append(f"      depends_on: {deps},")
            lines.append(f'      label: "{label}" }}')
        return "\n".join(lines) + "\n"

# --- per-op-type mapping ----------------------------------------------------
def map_node(node, shapes, sched, batch):
    t = node.op_type
    outs = list(node.output)
    out_shape = shapes.get(outs[0], [])
    elems = numel(out_shape)

    if t in ("Gemm", "MatMul"):
        a, b = node.input[0], node.input[1]
        A, B = shapes.get(a, []), shapes.get(b, [])
        # A: [.., M, K]   B: [K, N]  (Gemm) or [.., K, N] (MatMul)
        M = numel(A[:-1]) if len(A) >= 2 else batch
        K = A[-1] if A else (B[-2] if len(B) >= 2 else 1)
        N = B[-1] if B else (out_shape[-1] if out_shape else 1)
        sched.emit([], "dma_load", "dma",
                   {"rows": K, "cols": N, "source": f"HBM.{b}", "destination": f"ibuf.{b}"},
                   [b], f"{t} weights {b}")
        sched.emit(outs, "gemm", "systolic",
                   {"M": M, "K": K, "N": N, "source_a": a, "source_b": b, "destination": outs[0]},
                   [a, b], f"{t} {node.name}")

    elif t == "Conv":
        # im2col -> GEMM. X:[N,Cin,H,W]  W:[Cout,Cin,Kh,Kw]  Y:[N,Cout,H',W']
        x, w = node.input[0], node.input[1]
        W = shapes.get(w, [1, 1, 1, 1]); Y = out_shape or [batch, W[0], 1, 1]
        Cout = W[0]; K = numel(W[1:])              # Cin*Kh*Kw
        M = numel(Y[:1] + Y[2:]) or batch          # N*H'*W'
        sched.emit([], "dma_load", "dma",
                   {"rows": K, "cols": Cout, "source": f"HBM.{w}", "destination": f"ibuf.{w}"},
                   [w], f"conv weights {w}")
        sched.emit([], "transpose", "access_core",
                   {"input_rows": M, "input_cols": K, "source": x, "destination": f"ibuf.{x}.im2col"},
                   [x], f"im2col {node.name}")
        sched.emit(outs, "gemm", "systolic",
                   {"M": M, "K": K, "N": Cout, "source_a": f"ibuf.{x}.im2col",
                    "source_b": f"ibuf.{w}", "destination": outs[0]},
                   [x, w], f"conv {node.name}")

    elif t in ("Relu", "Add", "Mul", "Sub", "Clip", "LeakyRelu"):
        op = "residual_add" if t == "Add" else "scale"
        sched.emit(outs, op, "vector_core",
                   {"length": elems, "source": node.input[0], "destination": outs[0]},
                   list(node.input), f"{t} {node.name}")

    elif t in ("Sigmoid", "Tanh", "Gelu", "Erf", "Softplus", "HardSigmoid"):
        sched.emit(outs, "silu", "vector_core",       # has exp -> exp_latency class
                   {"length": elems, "source": node.input[0], "destination": outs[0]},
                   list(node.input), f"{t} {node.name}")

    elif t == "Softmax":
        sched.emit(outs, "softmax", "vector_core",
                   {"length": elems, "source": node.input[0], "destination": outs[0]},
                   list(node.input), f"softmax {node.name}")

    elif t in ("BatchNormalization", "LayerNormalization", "InstanceNormalization"):
        sched.emit(outs, "rmsnorm", "vector_core",
                   {"length": elems, "source": node.input[0], "destination": outs[0]},
                   list(node.input), f"{t} {node.name}")

    elif t in ("MaxPool", "AveragePool", "GlobalAveragePool", "ReduceMean", "ReduceSum"):
        in_elems = numel(shapes.get(node.input[0], out_shape))
        sched.emit(outs, "row_reduce_sum", "vector_core",
                   {"length": in_elems, "source": node.input[0], "destination": outs[0]},
                   list(node.input), f"{t} {node.name}")

    elif t in ("Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose",
               "Concat", "Identity", "Cast", "Dropout"):
        # Layout / no-op: pass the dependency through as a cheap access copy.
        sched.emit(outs, "sram_copy", "access_core",
                   {"length": max(elems, 1), "source": node.input[0] if node.input else "x",
                    "destination": outs[0]},
                   list(node.input), f"{t} {node.name}")

    else:
        # Unknown op: approximate as a one-pass vector op over its output.
        # (Better: register a dedicated op — see documentation/02.)
        sys.stderr.write(f"warn: approximating unsupported op '{t}' ({node.name})\n")
        sched.emit(outs, "scale", "vector_core",
                   {"length": max(elems, 1), "source": node.input[0] if node.input else "x",
                    "destination": outs[0]},
                   list(node.input), f"approx {t} {node.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    model = shape_inference.infer_shapes(onnx.load(args.model))
    graph = model.graph
    shapes = collect_shapes(graph)
    sched = Sched()
    for node in graph.node:            # ONNX nodes are already topologically ordered
        map_node(node, shapes, sched, args.batch)
    with open(args.out, "w") as f:
        f.write(sched.to_yaml())
    print(f"wrote {sched.next_id} instructions to {args.out}")

if __name__ == "__main__":
    main()
```

Workflow end-to-end:

```bash
pip install onnx
# grab any model from github.com/onnx/models, e.g. ResNet-18 or BERT
python3 onnx_to_schedule.py resnet18.onnx schedules/resnet18.yaml --batch 1
./build/apps/sim_main --config configs/datacenter.yaml \
    --schedule schedules/resnet18.yaml --no-trace
```

The `== metrics ==` block now reports ResNet-18's MACs, HBM bytes, systolic
utilization, and the roofline verdict on the datacenter config — and you can swap to
`configs/edge_dev.yaml` to see the same model on an edge SoC, or feed the YAML into a
sweep (see the [README](../README.md#design-space-sweeps)).

> This script is a **starting point**, not a turnkey importer. Treat its op-mapping
> table as the thing you tune for your model family: refine Conv stride/pad handling,
> split fused ops, or register dedicated ops for anything it currently approximates.

---

## 9. Validating a schedule you created

1. **It runs to completion.** The footer must read `outstanding=0`. A non-zero
   value means a dependency was never satisfied — usually a `depends_on` pointing at
   an id that no op ever completes, or an op whose unit isn't wired (see
   [Guide 1](01-adding-a-new-unit.md) pitfalls).
2. **MACs match a hand calc.** Sum `M·K·N` over your GEMMs and compare to the
   `MACs=` line. A mismatch means a wrong dimension somewhere.
3. **HBM bytes are sane.** `HBM_bytes=` should ≈ Σ(weight elements)·`dtype_bytes`
   plus any activation loads. If it's ~0, you forgot the `dma_load`s and the memory
   roofline will be meaningless.
4. **The bottleneck verdict is plausible.** Big-matmul models on a small array →
   compute-bound; bandwidth-starved configs or tiny GEMMs → memory-bound. If the
   verdict is surprising, re-check the dims that drive whichever side dominates.

The `--schedule` path runs `Schedule::validate()` first, so structural errors
(duplicate ids, unknown deps, cycles) are caught before simulation with a clear
error.

---

## 10. Limitations & when to add a new op

- **No native `conv`, `pool`, or normalization ops** — they are *modeled* via GEMM /
  reductions / vector passes (§5). That's deliberate (the hardware does conv as
  GEMM), but it means stride/padding/dilation only affect timing through the dims
  *you* compute, not through any conv semantics in the sim.
- **No real numerics** — anything data-dependent (dynamic shapes, control flow,
  `If`/`Loop`/`NonMaxSuppression`) must be flattened to a fixed shape before
  emitting a schedule.
- **Latency-class approximation** — when you map ten different activation functions
  onto `scale`/`silu`, you're asserting they share a cost class. If a specific op is
  performance-critical and *doesn't* fit, register a dedicated op with the right
  `passes`/`exp_ops` in one line ([Guide 2](02-adding-a-new-operation.md)).
- **Batch / sequence dims** live entirely in the `M` you compute — there is no
  implicit batching. Fold batch into `M` for GEMMs and into element counts for
  vector ops.

---

## Checklist

- [ ] Every matmul/conv → a `gemm` with correct `M`, `K`, `N` (conv via im2col dims).
- [ ] Every weight/activation brought from HBM → a `dma_load` (so HBM bytes count).
- [ ] Activations/norms/pools → the vector/access op of the matching latency class.
- [ ] `depends_on` reflects real data dependencies; ids unique; no cycles.
- [ ] `gemm` uses `unit: systolic`; dma → `dma`; vector → `vector_core`; access →
      `access_core`.
- [ ] Ran it: `outstanding=0`, and `MACs` / `HBM_bytes` match a hand calculation.
- [ ] For a parametric model family, consider a C++ `build_*` builder; for arbitrary
      real models, a graph-walking generator like §8.

---

## Summary

Creating a schedule is **mapping a model's work onto the simulator's compute /
memory / elementwise op vocabulary** so that MACs, HBM bytes, and per-op element
counts are faithful — numbers, not numerics. Hand-write YAML for small or exact
workloads, write a C++ `Builder` for parametric families (the LLaMA path is the
reference), and for real models from [ONNX](https://github.com/onnx/models) walk the
graph and emit YAML with a script like §8. Once the schedule runs with
`outstanding=0` and its MACs/bytes check out, every config, preset, and sweep in the
simulator applies to *your* model. See [Guide 3](03-simulator-engine.md) for how
that schedule then becomes a cycle count.
