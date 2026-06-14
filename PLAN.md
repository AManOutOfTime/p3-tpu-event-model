# LLaMA Pipeline Project Plan

## Current Verified State

- Build status: passes after rebuilding.
- Test status: `ctest --test-dir build --output-on-failure` passes.
- Runtime smoke status: `./build/apps/sim_main --config configs/default.yaml --schedule schedules/fa2_single_tile.yaml --no-trace` completes with no outstanding events.
- LLaMA smoke status: `./build/apps/sim_main --config configs/default.yaml --llama-workload /private/tmp/llama_event_only_smoke.yaml --no-trace` completed with 545 instructions and no outstanding events; the temporary YAML was removed after the run.
- Existing hand-written FA2 schedule remains available at `schedules/fa2_single_tile.yaml`.
- Generated LLaMA schedules now model TPU-style dataflow explicitly: weights move HBM -> shared SRAM/input buffer -> systolic operands, MXU outputs land in shared output buffers, and KV cache movement is explicit HBM DMA or SRAM copy depending on config.
- LLaMA schedule configs now derive `head_dim = hidden_dim / num_q_heads` and `gqa_group_size = num_q_heads / num_kv_heads` when omitted, and reject inconsistent attention dimensions.
- Generated LLaMA schedules now default to detailed granularity, expanding linear tiling, RMSNorm, RoPE, SwiGLU, logits softmax, sampling phases, and KV staging-slot release events.
- Oversized GEMM instructions are expanded through the original `Tiler::decompose()` subtile path before simulation; runtime GEMM no longer collapses subtiles into one multiplied-latency event.
- Generated LLaMA attention now follows the single-tile FA2 event body for each Q/KV tile pair: QK, scale/mask, rowmax, online `m/l/O_acc` update, PV, accumulate, then post-KV-loop normalize and logsumexp per query head tile.
- Generated LLaMA schedules now support only the detailed path; `schedule_granularity: coarse` is rejected instead of generating a second schedule style.
- Generated LLaMA MLP now streams a tile-level kernel: gate/up tiles, tile-local SiLU and multiply, partial down GEMMs, accumulated down output tiles, and final output assembly.
- Runtime units are event/timing-only. Systolic, vector, DMA, and access units no longer compute or mutate TensorStore values at `OP_DONE`.
- Architecture config now supports `systolic_units`; `sim_main` registers a systolic/MXU pool and existing logical `systolic` GEMMs dispatch across that pool.

## Completed

- [x] Stabilized typed op registration.
  - Moved reusable typed handlers into `src/schedule/op_handlers.h/.cpp`.
  - `sim_main` now uses `register_builtin_ops()` instead of private app-local handlers.
  - GEMM now reserves the systolic unit through the scheduler, preserving single-array serialization.

- [x] Added typed FA2 event-runtime coverage.
  - Added test coverage that runs `schedules/fa2_single_tile.yaml` through real DMA/access/systolic/vector units as timing events.
  - The test verifies schedule completion and cycle progression without requiring TensorStore value seeding or numerical outputs.

- [x] Added programmatic LLaMA schedule generation.
  - Added `src/schedule/llama_schedule.h/.cpp`.
  - Supports `attention`, `layer`, `prefill`, `decode`, and combined `prefill_decode` dispatch.
  - Generates tiled attention schedules instead of requiring handwritten YAML.
  - Preserves the current single hardware tile/array behavior through scheduler reservations.

- [x] Added GQA-aware schedule structure.
  - Parameters: `num_q_heads`, `num_kv_heads`, `gqa_group_size`.
  - Loop order is KV head group -> K/V tile movement -> Q heads in the group.
  - K/V cache is keyed by KV head, so Q heads in a group reuse loaded K/V tiles.

- [x] Added KV cache schedule options.
  - Parameters: `kv_cache_enabled`, `kv_cache_location: sram|hbm`.
  - HBM cache movement uses `dma_load`/`dma_store`.
  - SRAM cache movement uses new `sram_copy` access-core op.

- [x] Added transformer-layer schedule pieces.
  - Attention block: Q/K/V projections, RoPE, causal mask, FA2-style softmax/PV, output projection.
  - Layer block: attention RMSNorm, attention residual, MLP RMSNorm, gate/up projections, SwiGLU, down projection, MLP residual.
  - Prefill/decode combined schedule writes KV during prefill and then runs repeated decode steps.
  - These are schedule events with modeled cycle costs; generated LLaMA workloads are not intended to compute model values on CPU.

- [x] Added full diagram-level pipeline events around the transformer stack.
  - Token embedding lookup is modeled as a DMA/HBM event with `[tokens x hidden_dim]` dimensions.
  - Generated schedules now support `num_layers > 1` by chaining an Nx layer stack.
  - Attention tile outputs feed an explicit `attention_merge` vector event before output projection.
  - Final RMSNorm, LM-head linear logits projection, logits softmax, sampled-token event, and sampled-token feedback into decode are explicit schedule events.
  - New events target TPU-style resources: systolic/MXU for matrix multiplies, vector/VPU for elementwise/reduction/softmax/sample/merge, DMA for HBM movement, and access/scatter-gather for layout/select movement.

- [x] Added CLI workload path for generated LLaMA schedules.
  - New option: `--llama-workload FILE`.
  - Added sample workload: `workloads/llama_prefill_decode.yaml`.

- [x] Added TPU-style staged GEMM schedule events.
  - Generated Q/K/V, attention output, MLP gate/up/down, and LM-head GEMMs no longer consume `HBM.*` operands directly.
  - Each generated linear projection now emits weight DMA load, activation staging, weight staging, then systolic GEMM.
  - This keeps generated LLaMA schedules event/cycle based; no LLaMA numerical execution is required.

- [x] Tightened KV cache schedule addressing.
  - K/V cache writes and reads now use consistent range-addressed names per layer, KV head, tensor kind, and token range.
  - Prefill writes prompt K/V cache ranges; decode appends the generated token range and reads cached ranges back into shared input buffers.

- [x] Modeled SRAM KV cache capacity checks.
  - Added `max_seq_len`, `dtype_bytes`, and `sram_kv_capacity_kb` schedule config fields.
  - SRAM KV cache checks use `num_layers * num_kv_heads * max_seq_len * head_dim * dtype_bytes * 2`.
  - `sim_main` defaults `sram_kv_capacity_kb` from configured shared SRAM (`ibuf_kb + obuf_kb`) when a LLaMA workload does not set it.

- [x] Added schedule-structure tests for TPU-style dataflow.
  - Tests assert generated GEMMs do not consume direct `HBM.*` operands.
  - Tests validate staged projection dimensions, cache range read/write naming, HBM vs SRAM cache movement, and SRAM cache over-capacity failure.

- [x] Refined `attention_merge` latency semantics.
  - Generated merge events now carry `q_tiles`, `kv_tiles`, `num_q_heads`, `head_dim`, `input_elements`, and `output_elements`.
  - Runtime latency now charges for reading all per-K/V-tile attention contributions plus writing the final contiguous attention-head output.
  - Added a handler-level cycle test for the new merge element accounting.

- [x] Added automatic attention dimension derivation and validation.
  - `head_dim` may be omitted or set to `0`; the builder derives it from `hidden_dim / num_q_heads`.
  - `gqa_group_size` may be omitted or set to `0`; the builder derives it from `num_q_heads / num_kv_heads`.
  - Schedule generation now rejects non-divisible `hidden_dim`, incorrect explicit `head_dim`, and incorrect explicit GQA group sizes.

- [x] Added FA2-YAML-style granularity for generated LLaMA schedules.
  - Added `schedule_granularity: detailed|coarse`, defaulting to `detailed`.
  - Added separate `linear_tile_rows` and `linear_tile_cols` so linear GEMMs tile at MXU-style granularity independently of attention K/V tile shape.
  - Detailed linear projections now emit activation tile load, weight tile DMA load, activation/weight staging, per-tile systolic GEMM, tile placement, and output assembly.
  - Detailed RMSNorm now emits input tile load, square, row reduction, epsilon add, rsqrt, RMS weight load, scale/write, and output assembly phases.
  - Detailed RoPE now emits sin/cos table load, input tile load, pair split, rotate/multiply-add, and write phases.
  - Detailed MLP/SwiGLU now emits separate SiLU and elementwise multiply phases between tiled gate/up and down projections.
  - Detailed logits now emit tiled LM-head projection, rowmax, exp, rowsum, normalize, and sample phases.
  - Coarse mode remains available for quick schedule generation.

- [x] Added double-buffered KV prefetch and cache residency modeling.
  - Added `kv_prefetch: none|double_buffer`, `kv_stage_buffers`, `kv_cache_block_tokens`, and `kv_cache_eviction_policy: fail|spill_to_hbm`.
  - K/V cache names now include page/block/range components so writes, reads, prefill ranges, and decode appends use one logical address scheme.
  - Decode/prefill attention now stages cached K/V tiles into alternating `shared_ibuf.kv_stage.*.slotN` buffers, with explicit `kv_stage_release` events before slot reuse.
  - K and V cache reads for the same tile are independently schedulable; the scheduler/resource model decides whether they serialize or overlap on available DMA/SRAM units.
  - `spill_to_hbm` models hot SRAM-resident KV pages and older HBM-spilled pages; default `fail` still rejects over-capacity fully-SRAM KV cache workloads.

- [x] Routed oversized GEMMs through the original subtiler.
  - Added `Tiler::expand_gemm_subtiles()` as dependency-preserving glue around the existing `Tiler::decompose()` implementation.
  - `sim_main` expands generated LLaMA schedules and hand-written schedules before launching the scheduler.
  - Runtime GEMM handling now models exactly one physical systolic execution and rejects oversized GEMMs that bypass the tiler.
  - Added regression coverage that verifies oversized generated GEMM tiles produce old tiler `STAGE Q_sub_r` and `S[r,c]` sub-events with hardware-sized GEMM shapes.

- [x] Made generated attention exact down to the single-tile FA2 schedule body.
  - Generated attention initializes one `O_acc`, `m`, and `l` state per KV head, query head, and query tile before iterating over KV tiles.
  - Each KV tile now emits the same FA2 inner sequence used by `schedules/fa2_single_tile.yaml`, with causal masking inserted before rowmax for autoregressive attention.
  - Online `m`, `l`, and `O_acc` dependencies carry across KV tiles instead of normalizing each KV tile independently.
  - `normalize` and `logsumexp` now run once after the KV loop for each finalized query head tile, and `attention_merge` consumes finalized head tiles instead of per-KV partial outputs.
  - Added regression coverage for the exact generated FA2 inner event sequence and carried state dependencies across two KV tiles.

- [x] Made generated MLP faithful down to a single-tile kernel schedule.
  - Removed the generated coarse schedule path; LLaMA generation now accepts only `schedule_granularity: detailed`.
  - Replaced the full-matrix gate/up -> full-matrix SwiGLU -> full-matrix down sequence with a streamed tile kernel.
  - For each row tile and intermediate tile, the schedule emits gate/up weight loads and stages, gate/up GEMMs, tile-local SiLU, tile-local multiply, down weight loads and stages, partial down GEMMs, and accumulated output tiles.
  - Down output accumulation now carries across intermediate tiles before placing and assembling the final MLP output tile.
  - Added regression coverage for tile-level MLP event counts, dimensions, and dependencies.

- [x] Removed numerical computation from runtime simulation.
  - Systolic GEMM runtime now only charges latency and emits start/done events; it no longer reads input buffers, computes MACs, writes outputs, or logs skipped compute.
  - Vector, DMA, and access units now model timing only and do not compute or mutate TensorStore contents.
  - `sim_main` no longer seeds random FA2/LLaMA buffers or prints numerical TensorStore outputs.
  - The tiler now produces symbolic subtile schedule events without seeding/slicing TensorStore data.

- [x] Documented generated LLaMA workload modes and sequence fields.
  - Added README guidance for `attention`, `layer`, `prefill`, `decode`, and `prefill_decode`.
  - Documented `prompt_len`, `seq_len`, `generation_steps`, and `max_seq_len` semantics.
  - Clarified when `seq_len` can be omitted from a `prefill_decode`-only workload.

- [x] Added configurable MXU count.
  - Added top-level `systolic_units` to `ArchConfig`, YAML parsing, serialization, and `configs/default.yaml`.
  - `sim_main` now registers `systolic_0..N-1`; existing GEMM instructions still target logical `systolic` and use the event-engine unit pool.
  - Added regression coverage showing two independent GEMMs complete in parallel with `systolic_units = 2`.

- [x] Removed `sweep.py` PyYAML dependency.
  - Replaced `yaml.safe_load` usage with a small reader for the repo's simple key/value YAML config shape.

## Changed Files

- `README.md`
- `apps/sim_main.cpp`
- `configs/default.yaml`
- `src/config/arch_config.h`
- `src/config/arch_config.cpp`
- `src/schedule/op_handlers.cpp`
- `src/schedule/llama_schedule.h`
- `src/schedule/llama_schedule.cpp`
- `src/schedule/tiler.h`
- `src/schedule/tiler.cpp`
- `src/units/systolic_unit.h`
- `src/units/systolic_unit.cpp`
- `tests/test_config.cpp`
- `tests/test_llama_schedule.cpp`
- `sweep.py`
- `PLAN.md`

## Tests Run

- `cmake --build build --parallel`
- `ctest --test-dir build --output-on-failure`
- `./build/apps/sim_main --config configs/default.yaml --schedule schedules/fa2_single_tile.yaml --no-trace`
- `./build/apps/sim_main --config configs/default.yaml --llama-workload /private/tmp/llama_event_only_smoke.yaml --no-trace` (temporary YAML removed after run)
- `python3 -m py_compile sweep.py`
- `python3 sweep.py --model 1b --workload1b workloads/llama_prefill_decode_1B.yaml --group 1a --dry-run`

## Remaining Issues / Next Steps

- [ ] Add exact cycle golden tests for generated attention, GQA attention, KV cache movement, and full layer event ordering.
- [ ] Reduce unit output verbosity for generated schedules when `--no-trace` is used; units still print their own operation logs today.
- [ ] Re-run the full current LLaMA3-shaped workload after adding quieter generated-schedule execution; the tile-level MLP expansion makes the verbose full smoke much larger than the previous 202926-instruction run.
- [ ] Add hardware-specific presets for full LLaMA3 model sizes and TPU MXU tile dimensions instead of relying on workload-local linear tile defaults.
