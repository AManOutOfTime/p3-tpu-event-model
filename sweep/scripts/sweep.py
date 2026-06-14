#!/usr/bin/env python3
"""
sweep.py — TPU Simulator Parameter Sweep
=========================================
Varies ONE axis at a time on top of configs/default.yaml.

Usage:
  python3 scripts/sweep.py --dry-run [--group PREFIX]
  python3 scripts/sweep.py [--model 8b|70b|both] [--group PREFIX] [--out FILE]

--model 8b   (default) Run all groups on 8B workload only
--model 70b            Run all groups on 70B workload only
--model both           Run all groups on both workloads (doubles runtime)

Groups
------
  1a  Compute — array size          (tile = array → tiler passthrough)
  1b  Compute — systolic unit count
  1c  Compute — bidirectional
  1d  Compute — vector cores
  1e  Compute — vector SIMD width × exp_latency combos
  1f  Compute — access core bandwidth        ← NEW
  2a  Memory  — HBM bandwidth
  2b  Memory  — HBM latency
  2c  Memory  — DMA channels                 (2 systolic to show overlap)
  2d  Memory  — stage_double_buffer          (closes roofline gap)
  2e  Memory  — SRAM pressure                (model_sram=true, vary ibuf_kb)
  3a  SW      — prompt length                (FFN→attn regime annotated)
  3b  SW      — tile size                    (3 cases: sub / exact / over)
  3c  SW      — KV cache on/off
  3d  SW      — head_dim × hidden_dim        ← NEW  (num_q_heads=32 fixed)
  3e  SW      — max_seq_len / KV footprint   ← NEW  (gen_steps=32 for decode stats)
  4a  GQA     — 8B  MHA→MQA
  4b  GQA     — 70B MHA→MQA
  5   Pareto  — array-size × HBM-bw grid
  6   Calib   — real-chip HBM configs, compare hbm_util_pct to published

Key metrics (KPIs)
------------------
  TTFT (ns)          — time to process the full prompt (prefill latency)
  ttft_per_token_ns  — TTFT / prompt_len (prefill only; None for decode rows)
  decode_tps         — generated tokens / sec (meaningful only with gen_steps > 1)
  hbm_util_pct       — hbm_bw_achieved / hbm_bw_peak: calibration target ±20%
  roofline_eff_pct   — bound_cycles / actual_cycles: how close to the ceiling
  mem_compute_ratio  — memory_cyc / compute_cyc: >1 = memory-bound, <1 = compute-bound
  arith_intensity    — MACs / HBM_bytes: position on roofline x-axis
  systolic_imbalance — |util_0 − util_1|: work distribution across systolic units
  bytes_per_token    — prefill: HBM bytes / (prompt_len + gen_steps)
                       decode:  HBM bytes / gen_steps = GB per output token

Calibration note
----------------
  Comparing absolute TPS to H100/TPUv4 is meaningless — single 256×256 array
  ≈ 0.13 TFLOPS vs H100 989 TFLOPS. Correct comparison: hbm_util_pct.
  Real hardware bs=1 decode: 50–70% HBM utilisation.
  Sim with stage_double_buffer=ON should match within ±20%.

  AppleM3Max and RTX4090 are excluded from calibration: their published TPS
  numbers come from INT4 inference (~4–5 GB/token), not BF16 (18 GB/token),
  so BYTES_PER_TOKEN=18e9 would yield >100% utilisation and always fail.

FFN vs attention crossover (8B): S ≈ 3 × intermediate_dim / 2 ≈ 21K tokens
  S=2048: FFN 91% / attn 9%     S=8192: FFN 70% / attn 30%
"""

import argparse, csv, os, re, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

DEFAULT_BINARY  = "./build/apps/sim_main"
DEFAULT_OUTFILE = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
DEFAULT_HW      = "configs/default.yaml"
DEFAULT_WL1B    = "workloads/llama_prefill_decode_1B.yaml"
DEFAULT_WL8B    = "workloads/llama_prefill_decode_8B.yaml"
DEFAULT_WL70B   = "workloads/llama_prefill_decode_70B.yaml"
SIM_TIMEOUT_S   = 600   # 10 min hard cap per run

# ---------------------------------------------------------------------------
# CSV column order — fixed so plotting scripts can rely on position / name
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    # ── identification ────────────────────────────────────────────────────
    "group", "name", "model", "description",
    # ── arch params ───────────────────────────────────────────────────────
    "clock_ghz",
    "array_rows", "array_cols", "bidirectional",
    "systolic_units", "vector_cores", "access_cores",
    "vec_simd", "exp_lat", "access_bw",
    "dma_channels",
    "hbm_bw_tb_s", "hbm_lat_cycles",
    "ibuf_kb", "obuf_kb",
    "stage_double_buffer", "model_sram",
    # ── workload params ───────────────────────────────────────────────────
    "tile_rows", "tile_cols",
    "mode",
    "num_layers", "hidden_dim", "head_dim", "intermediate_dim",
    "num_q_heads", "num_kv_heads", "gqa_group",
    "prompt_len", "gen_steps", "max_seq_len", "kv_cache",
    # ── raw sim outputs ───────────────────────────────────────────────────
    "cycles", "MACs", "hbm_bytes",
    "systolic_util_pct",
    "systolic_0_util_pct", "systolic_1_util_pct",   # per-unit for imbalance
    "dma_util_pct", "vec_util_pct", "access_util_pct",
    "roofline_eff_pct", "roofline_bound",
    "compute_bound_cyc", "memory_bound_cyc",         # raw roofline ceilings
    "ttft_ns", "decode_tps",
    # ── derived / KPI metrics ─────────────────────────────────────────────
    "tflops_achieved", "hbm_bw_achieved_tb_s", "tflops_peak",
    "hbm_util_pct",
    "arith_intensity",
    "ttft_per_token_ns",    # prefill: TTFT/prompt_len; None for decode rows
    "bytes_per_token",      # prefill: HBM/(prompt+gen); decode: HBM/gen_steps
    "systolic_imbalance",   # |util_0 − util_1| — work distribution quality
    "mem_compute_ratio",    # memory_cyc / compute_cyc  — bound strength
    "ffn_mac_pct", "attn_mac_pct", "ffn_attn_crossover_tok",
    # ── calibration ───────────────────────────────────────────────────────
    "pub_hbm_util_pct", "pub_tps_bs1", "calib_hbm_err_pct",
    # ── run metadata ──────────────────────────────────────────────────────
    "wall_s", "peak_rss_mb", "status",
]


# ── YAML readers ─────────────────────────────────────────────────────────────
def parse_yaml_value(value):
    v = value.strip().strip('"').strip("'")
    if v.lower() == "true": return True
    if v.lower() == "false": return False
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v

def read_simple_yaml(path):
    root, stack = {}, [(-1, {})]
    stack[0] = (-1, root)
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip(): continue
            indent = len(line) - len(line.lstrip())
            key, sep, value = line.strip().partition(":")
            if not sep: continue
            while indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value.strip():
                parent[key] = parse_yaml_value(value)
            else:
                parent[key] = {}
                stack.append((indent, parent[key]))
    return root

def read_arch(path):
    d = read_simple_yaml(path)
    s  = d.get("systolic", {})
    h  = d.get("hbm", {})
    vc = d.get("vector_core", {})
    ac = d.get("access_core", {})
    sr = d.get("sram", {})
    return dict(
        clock_ghz           = d.get("clock_ghz",          1.0),
        array_rows          = s.get("rows",                256),
        array_cols          = s.get("cols",                256),
        bidirectional       = s.get("bidirectional",      False),
        systolic_units      = d.get("systolic_units",       1),
        vector_cores        = d.get("vector_cores",         3),
        access_cores        = d.get("access_cores",         1),
        hbm_bw_tb_s         = h.get("bandwidth_tb_s",     2.0),
        hbm_lat_cycles      = h.get("latency_cycles",     200),
        dma_channels        = d.get("dma", {}).get("channels", 1),
        vec_simd            = vc.get("simd_width",          64),
        exp_lat             = vc.get("exp_latency",          4),
        access_bw           = ac.get("bandwidth",           64),
        ibuf_kb             = sr.get("ibuf_kb",           4096),
        obuf_kb             = sr.get("obuf_kb",           4096),
        banking_factor      = sr.get("banking_factor",       8),
        stage_double_buffer = d.get("stage_double_buffer", False),
        model_sram          = d.get("model_sram",          False),
    )

def read_workload(path):
    d = read_simple_yaml(path)
    ll = d.get("llama", {})
    return dict(
        mode             = ll.get("mode",          "prefill_decode"),
        tile_rows        = ll.get("tile_rows",           256),
        tile_cols        = ll.get("tile_cols",           256),
        num_layers       = ll.get("num_layers",           32),
        hidden_dim       = ll.get("hidden_dim",         4096),
        head_dim         = ll.get("head_dim",            128),
        intermediate_dim = ll.get("intermediate_dim",  14336),
        num_q_heads      = ll.get("num_q_heads",          32),
        num_kv_heads     = ll.get("num_kv_heads",          8),
        gqa_group        = ll.get("gqa_group_size",        4),
        vocab_size       = ll.get("vocab_size",       128256),
        prompt_len       = ll.get("prompt_len",         2048),
        gen_steps        = ll.get("generation_steps",      1),
        max_seq_len      = ll.get("max_seq_len",        8192),
        kv_cache         = ll.get("kv_cache_enabled",   True),
        kv_loc           = ll.get("kv_cache_location",  "hbm"),
    )


# ── YAML generators ───────────────────────────────────────────────────────────
def arch_yaml(clock_ghz=1.0, array_rows=256, array_cols=256, bidirectional=False,
              systolic_units=1, vector_cores=3, access_cores=1,
              hbm_bw_tb_s=2.0, hbm_lat_cycles=200, dma_channels=1,
              vec_simd=64, exp_lat=4, access_bw=64,
              ibuf_kb=4096, obuf_kb=4096, banking_factor=8,
              stage_double_buffer=False, model_sram=False, **_):
    def b(v): return "true" if v else "false"
    return (
        f"clock_ghz: {clock_ghz}\n"
        f"systolic:\n  rows: {array_rows}\n  cols: {array_cols}\n"
        f"  precision: BF16\n  bidirectional: {b(bidirectional)}\n  d_head: 128\n"
        f"  dataflow: weight_stationary\n  weight_load_cycles: 0\n"
        f"  weight_double_buffer: true\n"
        f"structural_k_tiling: {b(model_sram)}\n"
        f"model_sram: {b(model_sram)}\n"
        f"stage_double_buffer: {b(stage_double_buffer)}\n"
        f"systolic_units: {systolic_units}\n"
        f"vector_cores: {vector_cores}\n"
        f"access_cores: {access_cores}\n"
        f"sram:\n  ibuf_kb: {ibuf_kb}\n  obuf_kb: {obuf_kb}\n"
        f"  banking_factor: {banking_factor}\n  private_vector_kb: 512\n"
        f"hbm:\n  bandwidth_tb_s: {hbm_bw_tb_s}\n"
        f"  latency_cycles: {hbm_lat_cycles}\n  pipelined: true\n"
        f"dma:\n  channels: {dma_channels}\n"
        f"vector_core:\n  simd_width: {vec_simd}\n  exp_latency: {exp_lat}\n"
        f"access_core:\n  bandwidth: {access_bw}\n")

def workload_yaml(mode="prefill_decode",
                  num_layers=32, num_q_heads=32, num_kv_heads=8, gqa_group=4,
                  head_dim=128, hidden_dim=4096, intermediate_dim=14336,
                  vocab_size=128256, prompt_len=2048, gen_steps=1, max_seq_len=8192,
                  tile_rows=256, tile_cols=256, kv_cache=True, kv_loc="hbm", **_):
    return (
        f"llama:\n  mode: {mode}\n  prompt_len: {prompt_len}\n"
        f"  generation_steps: {gen_steps}\n  num_layers: {num_layers}\n"
        f"  num_q_heads: {num_q_heads}\n  num_kv_heads: {num_kv_heads}\n"
        f"  gqa_group_size: {gqa_group}\n  head_dim: {head_dim}\n"
        f"  hidden_dim: {hidden_dim}\n  intermediate_dim: {intermediate_dim}\n"
        f"  vocab_size: {vocab_size}\n  max_seq_len: {max_seq_len}\n"
        f"  dtype_bytes: 2\n"
        f"  tile_rows: {tile_rows}\n  tile_cols: {tile_cols}\n"
        f"  kv_cache_enabled: {'true' if kv_cache else 'false'}\n"
        f"  kv_cache_location: {kv_loc}\n")


# ── Regime (analytical FFN vs attn FLOP split) ────────────────────────────────
def regime(wl):
    S   = wl.get("prompt_len",        2048)
    H   = wl.get("hidden_dim",        4096)
    I   = wl.get("intermediate_dim", 14336)
    nq  = wl.get("num_q_heads",         32)
    nk  = wl.get("num_kv_heads",         8)
    dh  = wl.get("head_dim",           128)
    ffn  = S * (3 * H * I)
    attn = S * (nq*dh*H + nk*dh*H*2 + nq*dh*H) + nq*S*S*dh*2
    tot  = ffn + attn
    ffn_pct = round(ffn / tot * 100, 1)
    return dict(
        ffn_pct    = ffn_pct,
        attn_pct   = round(100 - ffn_pct, 1),
        crossover  = int(3 * I // 2),
    )


# ── Calibration reference ─────────────────────────────────────────────────────
# Metric: hbm_util_pct = hbm_bw_achieved / hbm_bw_peak  (NOT absolute TPS)
# Llama-3 8B, GQA: 8 KV heads, 32 layers, head_dim=128, BF16
#   weights : 8 B params × 2 B                             = 16.000 GB
#   KV cache: 2 × 32 × 8 × 128 × 2 B × 2 048 tok ≈ 0.268 GB
#   total                                                  = 16.268 GB
BYTES_PER_TOKEN = 16.27e9   # was 18e9 — old KV term (~2 GB) was ~8× too large

CALIB = {
    "H100_SXM5": dict(
        hbm_bw_tb_s=3.35,
        hbm_lat_cycles=280,     # 280 / 1.98 GHz ≈ 141 ns — same physical ns as A100 ✓
        dma_channels=2,
        clock_ghz=1.98,
        vector_cores=4,
        pub_tps_bs1=120,        # 16.27 GB / 3.35 TB/s → 58% BW util ✓ (unchanged)
    ),
    "A100_SXM4": dict(
        hbm_bw_tb_s=2.00,
        hbm_lat_cycles=200,     # 200 / 1.41 GHz ≈ 142 ns ✓ (unchanged)
        dma_channels=1,
        clock_ghz=1.41,
        vector_cores=3,
        pub_tps_bs1=70,         # 16.27 GB / 2.00 TB/s → 57% BW util ✓ (unchanged)
    ),
    "TPUv4_chip": dict(
        hbm_bw_tb_s=1.20,
        hbm_lat_cycles=140,     # 140 / 1.05 GHz ≈ 133 ns
        dma_channels=1,
        clock_ghz=1.05,
        vector_cores=2,
        pub_tps_bs1=43,         # was 60 → implied 90% util; 73.7 TPS max × 58% ≈ 43
        array_rows=128,
        array_cols=128,         # was 256; one MXU is 128×128  (→ YAML systolic.cols)
        systolic_units=4,       # new; 4 MXUs per chip          (→ YAML systolic_units)
    ),
}
for chip, p in CALIB.items():
    p["pub_hbm_util_pct"] = round(
        p["pub_tps_bs1"] * BYTES_PER_TOKEN / (p["hbm_bw_tb_s"] * 1e12) * 100, 1)


# ── Sweep builder ─────────────────────────────────────────────────────────────
def make_sweep(hw, wl1b, wl8b, wl70b, models):
    """
    Returns list of (group, name, desc, arch_dict, wl_dict, model_tag).
    Every run overrides ONE axis on top of the default hw/workload config.
    """
    runs = []

    def add(grp, name, desc, arch_ov=None, wl_ov=None, model="8b"):
        if model not in models: return
        base = wl1b if model == "1b" else (wl8b if model == "8b" else wl70b)
        runs.append((grp, name, desc,
                     {**hw,   **(arch_ov or {})},
                     {**base, **(wl_ov  or {})},
                     model))

    def addm(grp, name, desc, arch_ov=None, wl_ov=None,
             modes=("prefill_decode", "decode")):
        """Add config for every selected model × every run mode.

        modes=("prefill_decode","decode")  → two rows per config, suffixed _pd/_dec
        modes=("prefill_decode",)          → one row, no suffix  (calibration)
        modes=("decode",)                  → one row, no suffix  (decode-only groups)

        For prefill_decode: TTFT is the primary KPI.
        For decode:         decode_tps is the primary KPI.
                            prompt_len = context already in KV cache.
        """
        multi = len(modes) > 1
        for mode in modes:
            mode_wl = {**(wl_ov or {}), "mode": mode}
            mode_sfx = ("_pd" if mode == "prefill_decode" else "_dec") if multi else ""
            for m in sorted(models):
                m_sfx = f"_{m}" if len(models) > 1 else ""
                add(grp,
                    name + mode_sfx + m_sfx,
                    desc + (f" [mode={mode}]" if multi else ""),
                    arch_ov, mode_wl, model=m)

    arr = hw["array_rows"]

    # ── 1a. Array size ────────────────────────────────────────────────────────
    # tile = array so the tiler is a passthrough — isolates pure array-size
    # effect without sub-tiling artefacts confounding the result.
    for sz in [64, 128, 256, 512]:
        if   sz == arr: note = "EXACT FIT — tiler passthrough"
        elif sz <  arr: note = f"undersize {sz*sz*100//(arr*arr)}% util"
        else:           note = f"oversize {(sz//arr)**2} internal passes/instr"
        addm("1a_array_size", f"arr{sz}",
             f"array={sz}x{sz} tile={sz} [{note}]",
             arch_ov=dict(array_rows=sz, array_cols=sz),
             wl_ov  =dict(tile_rows=sz,  tile_cols=sz))

    # ── 1b. Systolic unit count ───────────────────────────────────────────────
    for n in [1, 2, 3, 4]:
        addm("1b_systolic_units", f"sys{n}x",
             f"systolic_units={n}",
             arch_ov=dict(systolic_units=n))

    # ── 1c. Bidirectional ─────────────────────────────────────────────────────
    for bd in [False, True]:
        addm("1c_bidirectional",
             "bidir" if bd else "unidir",
             f"bidirectional={bd}  (bidir=2 MACs/cell/cycle, unidir=1)",
             arch_ov=dict(bidirectional=bd))

    # ── 1d. Vector cores ──────────────────────────────────────────────────────
    # vc_0 handles ~37% of vector work (softmax/GeLU); vc_1,2 ~6% each.
    # Beyond 3 there are diminishing returns in typical LLM workloads.
    for n in [1, 2, 3, 6]:
        addm("1d_vector_cores", f"vc{n}",
             f"vector_cores={n}",
             arch_ov=dict(vector_cores=n))

    # ── 1e. Vector SIMD width × exp_latency ──────────────────────────────────
    # simd_width: elements/cycle for vector ops (softmax, layernorm, GeLU)
    # exp_latency: cycles to compute e^x — only matters at long seq where
    #              softmax dominates; negligible at short seq (FFN-dominated).
    for simd, exp, note in [
        (16,  4, "narrow SIMD baseline"),
        (32,  4, "half-width"),
        (64,  2, "default SIMD, fast exp"),
        (64,  4, "default SIMD, default exp"),
        (64,  8, "default SIMD, slow exp — shows attn sensitivity"),
        (128, 4, "wide SIMD — benefit peaks at long seq"),
    ]:
        addm("1e_simd_exp", f"simd{simd}_exp{exp}",
             f"simd={simd} exp_lat={exp}  [{note}]",
             arch_ov=dict(vec_simd=simd, exp_lat=exp))

    # ── 1f. Access core bandwidth ─────────────────────────────────────────────
    # access_core handles scalar/irregular memory ops (index lookups, gather).
    # At ~4% utilisation in the default run it barely matters for Llama, but
    # sweeping confirms this empirically rather than assuming it.
    for bw in [32, 64, 128]:
        addm("1f_access_bw", f"abw{bw}",
             f"access_core.bandwidth={bw}",
             arch_ov=dict(access_bw=bw))

    # ── 2a. HBM bandwidth ─────────────────────────────────────────────────────
    for bw, tag in [(0.5, "LPDDR5X"), (1.0, "TPUv4"),
                    (2.0, "A100/default"), (3.35, "H100")]:
        addm("2a_hbm_bw", f"bw{bw}TB",
             f"hbm_bw={bw} TB/s ({tag})",
             arch_ov=dict(hbm_bw_tb_s=bw))

    # ── 2b. HBM latency ───────────────────────────────────────────────────────
    for lat in [50, 100, 200, 400]:
        addm("2b_hbm_lat", f"lat{lat}",
             f"hbm_lat={lat} cycles",
             arch_ov=dict(hbm_lat_cycles=lat))

    # ── 2c. DMA channels ──────────────────────────────────────────────────────
    # systolic_units=2 so there is meaningful compute to overlap against DMA.
    # Bug 2 fix: dma>1 switches to multi-unit print format in the sim output.
    for ch in [1, 2, 4]:
        addm("2c_dma_channels", f"dma{ch}ch",
             f"dma_channels={ch}  systolic_units=2",
             arch_ov=dict(dma_channels=ch, systolic_units=2))

    # ── 2d. Stage double buffer ───────────────────────────────────────────────
    # OFF: DMA → compute → DMA (serialised) → ~15× roofline gap
    # ON:  DMA prefetches tile N+1 while systolic computes tile N → closes gap
    for sdb in [False, True]:
        addm("2d_stage_doublebuf",
             "sdb_ON" if sdb else "sdb_OFF",
             f"stage_double_buffer={sdb}",
             arch_ov=dict(stage_double_buffer=sdb))

    # ── 2e. SRAM ibuf pressure ────────────────────────────────────────────────
    # model_sram=false → varying ibuf_kb has NO effect (capacity not enforced).
    # model_sram=true  → DMA stalls when ibuf occupancy > ibuf_kb.
    for kb in [512, 1024, 2048, 4096]:
        addm("2e_sram_ibuf", f"ibuf{kb}KB",
             f"ibuf_kb={kb} model_sram=ON",
             arch_ov=dict(ibuf_kb=kb, model_sram=True))

    # ── 3a. Prompt length ─────────────────────────────────────────────────────
    # Sparse set covering key regimes: tiny / short / default / long-context.
    # Each entry annotates the FFN% vs attn% split so regime transition is
    # visible without needing to run the full dense sweep.
    plens_8b  = [128, 512, 2048, 8192]
    plens_70b = [128, 512, 2048]          # 70B is ~2× heavier per token
    for plen in plens_8b:
        r = regime({**wl8b, "prompt_len": plen})
        add("3a_prompt_len", f"p{plen}",
            f"prompt={plen}  FFN={r['ffn_pct']}% attn={r['attn_pct']}%"
            f"  xover@{r['crossover']}tok",
            wl_ov=dict(prompt_len=plen), model="8b")
    for plen in plens_70b:
        r = regime({**wl70b, "prompt_len": plen})
        add("3a_prompt_len", f"p{plen}",
            f"prompt={plen}  FFN={r['ffn_pct']}% attn={r['attn_pct']}%"
            f"  xover@{r['crossover']}tok",
            wl_ov=dict(prompt_len=plen), model="70b")

    # ── 3b. Tile size ─────────────────────────────────────────────────────────
    # Three meaningful cases only:
    #   sub-tile  → tiler actively decomposes, array underutilised
    #   exact fit → tiler is a passthrough (1 instr = 1 array execution)
    #   over-size → tiler emits N² passes per instruction
    for ts, note in [
        (128, f"sub-tile: tiler active, {128*128*100//(arr*arr)}% array util"),
        (256, "EXACT FIT: tiler passthrough"),
        (512, f"over-size: {(512//arr)**2 if arr <= 512 else 1} passes/instr"),
    ]:
        addm("3b_tile_size", f"tile{ts}",
             f"tile={ts}x{ts} array={arr}x{arr}  [{note}]",
             wl_ov=dict(tile_rows=ts, tile_cols=ts))

    # ── 3c. KV cache on/off ───────────────────────────────────────────────────
    for kv_en in [True, False]:
        addm("3c_kv_cache",
             "kv_on" if kv_en else "kv_off",
             f"kv_cache_enabled={kv_en}",
             wl_ov=dict(kv_cache=kv_en))

    # ── 3d. Head dim × hidden dim ─────────────────────────────────────────────
    # hidden_dim = num_q_heads × head_dim  (num_q_heads=32 held fixed).
    # Larger head_dim → heavier per-head attention → pushes attn toward
    # compute-bound faster. Shows impact of architectural head-size choices.
    for hd, hidd in [(64, 2048), (128, 4096), (256, 8192)]:
        r = regime({**wl8b, "head_dim": hd, "hidden_dim": hidd})
        addm("3d_head_dim", f"hd{hd}",
             f"head_dim={hd} hidden_dim={hidd}  (32 q_heads fixed)"
             f"  FFN={r['ffn_pct']}% attn={r['attn_pct']}%",
             wl_ov=dict(head_dim=hd, hidden_dim=hidd))

    # 3e is intentionally omitted here — it uses gen_steps=32 which makes each
    # run ~10-20% slower. It runs AFTER group 5 (Pareto) so all fast
    # TTFT-producing configs complete first. See below.

    # ── 4a. GQA sweep 8B ─────────────────────────────────────────────────────
    nq = wl8b["num_q_heads"]
    for kv, grp_sz, tag in [
        (nq,      1,  "MHA — full KV"),
        (nq//4,   4,  "GQA-4 — default Llama-3-8B"),
        (nq//8,   8,  "GQA-8"),
        (nq//16, 16,  "GQA-16"),
        (1,      nq,  "MQA — minimum KV bandwidth"),
    ]:
        if kv < 1: continue
        for mode in ("prefill_decode", "decode"):
            mode_sfx = "_pd" if mode == "prefill_decode" else "_dec"
            add("4a_gqa_8b", f"kv{kv}_{tag.split()[0]}{mode_sfx}",
                f"8B kv_heads={kv} gqa_group={grp_sz} ({tag}) [mode={mode}]",
                wl_ov=dict(
                    num_kv_heads=kv,
                    gqa_group=grp_sz,
                    mode=mode,
                    gen_steps=32 if mode == "decode" else wl8b.get("gen_steps", 1)),
                model="8b")

    # ── 4b. GQA sweep 70B ────────────────────────────────────────────────────
    nq70 = wl70b["num_q_heads"]
    for kv, grp_sz, tag in [
        (nq70,      1,  "MHA"),
        (nq70//8,   8,  "GQA-8 — default Llama-3-70B"),
        (nq70//16, 16,  "GQA-16"),
        (1,      nq70,  "MQA"),
    ]:
        if kv < 1: continue
        for mode in ("prefill_decode", "decode"):
            mode_sfx = "_pd" if mode == "prefill_decode" else "_dec"
            add("4b_gqa_70b", f"kv{kv}_{tag.split()[0]}{mode_sfx}",
                f"70B kv_heads={kv} gqa_group={grp_sz} ({tag}) [mode={mode}]",
                wl_ov=dict(num_kv_heads=kv, gqa_group=grp_sz, mode=mode,
                           gen_steps=32 if mode == "decode" else wl70b.get("gen_steps", 1)),
                model="70b")

    # ── 5. Pareto: array size × HBM bandwidth ─────────────────────────────────
    # 3×3 grid. tile=array so tiler is a passthrough in all 9 cells.
    # Reveals where compute vs memory bound transitions occur.
    for sz in [128, 256, 512]:
        for bw in [1.0, 2.0, 3.35]:
            addm("5_pareto", f"arr{sz}_bw{bw}",
                 f"array={sz}x{sz} hbm_bw={bw}TB/s",
                 arch_ov=dict(array_rows=sz, array_cols=sz, hbm_bw_tb_s=bw),
                 wl_ov  =dict(tile_rows=sz,  tile_cols=sz))

    # ── 3e. Decode throughput vs context length ──────────────────────────────
    # Decode-only group: varies context length so the KV cache footprint
    # changes. All groups 1-5 already run both modes, so hw sensitivity to
    # decode is captured there. This group uniquely answers: how does pure
    # decode TPS degrade as context grows?
    #
    # Note: 2a (HBM bw sweep) already runs in decode mode too, so the
    # "decode bw wall" is captured there without needing a separate 3e_bw group.
    for ctx in [512, 1024, 2048, 4096, 8192]:
        kv_gb = 2 * 32 * 8 * 128 * ctx * 2 / 1e9
        addm("3e_decode_ctx", f"ctx{ctx}",
             f"mode=decode  ctx={ctx}  kv={kv_gb:.2f}GB  true_decode_tps",
             wl_ov=dict(mode="decode", prompt_len=ctx, gen_steps=32,
                        max_seq_len=max(ctx + 32, 8192)),
             modes=("decode",))

    # ── 6. Calibration ───────────────────────────────────────────────────────
    # Runs in decode mode (gen_steps=32) because published bs=1 TPS numbers
    # are decode-only — prefill_decode would mix in prefill HBM traffic and
    # inflate hbm_util_pct relative to the reference figures.
    # hbm_util_pct (bandwidth fraction) is what we compare — not absolute TPS.
    # SDB=OFF shows baseline gap; SDB=ON is the ±20% target.
    for chip, p in CALIB.items():
        for sdb in [False, True]:
            add("6_calibration",
                f"{chip}_{'sdb' if sdb else 'base'}",
                f"{chip} hbm={p['hbm_bw_tb_s']}TB/s sdb={sdb}"
                f"  pub_tps={p['pub_tps_bs1']}"
                f"  pub_hbm_util={p['pub_hbm_util_pct']}%",
                arch_ov={k: v for k, v in p.items()
                         if k not in ("pub_tps_bs1", "pub_hbm_util_pct")}
                        | {"stage_double_buffer": sdb},
                wl_ov=dict(mode="decode", gen_steps=32))

    return runs


# ── Runner ────────────────────────────────────────────────────────────────────
def run_sim(binary, ae, we):
    at = wt = False
    ap = wp = None
    try:
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(arch_yaml(**ae)); f.close(); ap = f.name; at = True

        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(workload_yaml(**we)); f.close(); wp = f.name; wt = True

        t0  = time.time()
        tb  = "/usr/bin/time"
        cmd = ([tb, "-v"] if os.path.isfile(tb) and os.access(tb, os.X_OK) else []) + \
              [binary, "--config", ap, "--llama-workload", wp, "--no-trace"]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=SIM_TIMEOUT_S)
        r = parse_out(res.stdout + res.stderr)
        r["wall_s"] = round(time.time() - t0, 1)
        r["ok"]     = True
        return r
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        for flag, path in [(at, ap), (wt, wp)]:
            if flag and path:
                try: os.unlink(path)
                except: pass


# ── Output parser ─────────────────────────────────────────────────────────────
def parse_out(out):
    r = {}
    def g(pat, cast=float):
        m = re.search(pat, out)
        return cast(m.group(1)) if m else None

    # Core sim counters
    r["cycles"]    = g(r"cycle=(\d+)", int)
    r["MACs"]      = g(r"MACs=(\d+)", int)
    r["hbm_bytes"] = g(r"HBM_bytes=(\d+)", int)

    # Unit utilisation — pool-specific patterns prevent cross-unit captures.
    # std::map iterates alphabetically: access_core → dma → systolic →
    # vector_core. Without specific prefixes, avg_util always picks up
    # systolic (first multi-unit pool printed).
    r["systolic_util"]   = (g(r"systolic\s+busy=\d+\s+util=([\d.]+)") or
                             g(r"systolic x\d+\s+avg_util=([\d.]+)"))
    r["systolic_0_util"] = g(r"systolic_0\s+busy=\d+\s+util=([\d.]+)")
    r["systolic_1_util"] = g(r"systolic_1\s+busy=\d+\s+util=([\d.]+)")

    r["dma_util"]    = (g(r"\bdma\s+busy=\d+\s+util=([\d.]+)") or
                        g(r"\bdma x\d+\s+avg_util=([\d.]+)"))
    r["vec_util"]    = (g(r"vector_core\s+busy=\d+\s+util=([\d.]+)") or
                        g(r"vector_core x\d+\s+avg_util=([\d.]+)"))
    r["access_util"] = (g(r"access_core\s+busy=\d+\s+util=([\d.]+)") or
                        g(r"access_core x\d+\s+avg_util=([\d.]+)"))

    # Roofline — parse compute and memory ceilings explicitly so
    # mem_compute_ratio can be computed independently in derived().
    m = re.search(
        r"roofline: compute=([\d.e+\-]+)cyc\s+memory=([\d.e+\-]+)cyc", out)
    if m:
        r["compute_bound_cyc"] = float(m.group(1))
        r["memory_bound_cyc"]  = float(m.group(2))
    else:
        r["compute_bound_cyc"] = None
        r["memory_bound_cyc"]  = None

    r["roofline_eff"]   = g(r"roofline_efficiency=([\d.]+)")
    m2 = re.search(r"bound=[\d.e+\-]+ \((\w+-bound)\)", out)
    r["roofline_bound"] = m2.group(1) if m2 else None

    # LLaMA KPIs
    r["ttft_ns"]          = g(r"TTFT=([\d.e+\-]+)")
    r["time_per_token_ns"]= g(r"time_per_token=([\d.e+\-]+)")  # decode only
    r["decode_tps"]       = g(r"throughput=([\d.]+)")

    # Resource usage (from /usr/bin/time -v)
    r["rss_kb"] = g(r"Maximum resident set size .kbytes.: (\d+)", int)

    return r


# ── Derived / KPI metrics ─────────────────────────────────────────────────────
def derived(r, ae, we):
    d = {}
    try:
        clk  = ae.get("clock_ghz", 1.0)
        cyc  = r.get("cycles") or 1
        ws   = cyc / (clk * 1e9)           # wall seconds at sim clock
        macs = r.get("MACs")      or 0
        hbm  = r.get("hbm_bytes") or 0

        # Compute throughput
        d["tflops_achieved"]     = round(macs * 2 / ws / 1e12, 4)
        d["hbm_bw_achieved"]     = round(hbm / ws / 1e12, 4)

        # Peak compute — bidirectional doubles MACs/cell/cycle.
        bidir_factor             = 2 if ae.get("bidirectional", False) else 1
        d["tflops_peak"]         = round(
            ae["array_rows"] * ae["array_cols"] * ae["systolic_units"]
            * bidir_factor * 2 * clk * 1e9 / 1e12, 4)

        # HBM utilisation (calibration target)
        d["hbm_util_pct"]        = round(
            d["hbm_bw_achieved"] / ae["hbm_bw_tb_s"] * 100, 2) \
            if ae.get("hbm_bw_tb_s") else None

        # Arithmetic intensity (roofline x-axis)
        d["arith_intensity"]     = round(macs / hbm, 2) if hbm else None

        mode = we.get("mode", "prefill_decode")

        # ttft_per_token_ns: meaningful for prefill only.
        # For decode, TTFT = one decode step time — dividing by prompt_len
        # (context length) gives an uninterpretable ratio. Set to None.
        if mode != "decode":
            ttft = r.get("ttft_ns") or 0
            plen = we.get("prompt_len") or 1
            d["ttft_per_token_ns"] = round(ttft / plen, 2)
        else:
            d["ttft_per_token_ns"] = None

        # bytes_per_token: cost per token differs by mode.
        # prefill: HBM bytes / (prompt_len + gen_steps) — cost per input token
        # decode:  HBM bytes / gen_steps — cost per OUTPUT token generated
        #          (this is the key decode metric: how many GB of weights +
        #           KV cache must be read from HBM to produce one new token)
        if mode == "decode":
            gen = we.get("gen_steps") or 1
            d["bytes_per_token"] = round(hbm / gen) if gen > 0 else None
        else:
            total_tokens = (we.get("prompt_len") or 0) + (we.get("gen_steps") or 0)
            d["bytes_per_token"] = round(hbm / total_tokens) if total_tokens > 0 else None

        # Work distribution across systolic units
        s0 = r.get("systolic_0_util")
        s1 = r.get("systolic_1_util")
        d["systolic_imbalance"]  = round(abs(s0 - s1), 2) \
                                   if (s0 is not None and s1 is not None) else None

        # Bound strength: >>1 = deeply memory-bound, ~1 = ridge, <1 = compute-bound
        comp = r.get("compute_bound_cyc") or 0
        mem  = r.get("memory_bound_cyc")  or 0
        d["mem_compute_ratio"]   = round(mem / comp, 3) if comp > 0 else None

        # FFN vs attention regime split
        d.update(regime(we))

    except Exception:
        pass
    return d


# ── Build one CSV row ─────────────────────────────────────────────────────────
def build_row(grp, name, desc, ae, we, mdl, r, dv,
              pub_hbm_util=None, pub_tps=None, calib_err=None):
    return {
        # identification
        "group": grp, "name": name, "model": mdl, "description": desc,
        # arch
        "clock_ghz":           ae.get("clock_ghz"),
        "array_rows":          ae.get("array_rows"),
        "array_cols":          ae.get("array_cols"),
        "bidirectional":       ae.get("bidirectional"),
        "systolic_units":      ae.get("systolic_units"),
        "vector_cores":        ae.get("vector_cores"),
        "access_cores":        ae.get("access_cores"),
        "vec_simd":            ae.get("vec_simd"),
        "exp_lat":             ae.get("exp_lat"),
        "access_bw":           ae.get("access_bw"),
        "dma_channels":        ae.get("dma_channels", 1),
        "hbm_bw_tb_s":         ae.get("hbm_bw_tb_s"),
        "hbm_lat_cycles":      ae.get("hbm_lat_cycles"),
        "ibuf_kb":             ae.get("ibuf_kb"),
        "obuf_kb":             ae.get("obuf_kb"),
        "stage_double_buffer": ae.get("stage_double_buffer"),
        "model_sram":          ae.get("model_sram"),
        # workload
        "tile_rows":           we.get("tile_rows"),
        "tile_cols":           we.get("tile_cols"),
        "mode":                we.get("mode", "prefill_decode"),
        "num_layers":          we.get("num_layers"),
        "hidden_dim":          we.get("hidden_dim"),
        "head_dim":            we.get("head_dim"),
        "intermediate_dim":    we.get("intermediate_dim"),
        "num_q_heads":         we.get("num_q_heads"),
        "num_kv_heads":        we.get("num_kv_heads"),
        "gqa_group":           we.get("gqa_group"),
        "prompt_len":          we.get("prompt_len"),
        "gen_steps":           we.get("gen_steps"),
        "max_seq_len":         we.get("max_seq_len"),
        "kv_cache":            we.get("kv_cache"),
        # raw sim
        "cycles":              r.get("cycles"),
        "MACs":                r.get("MACs"),
        "hbm_bytes":           r.get("hbm_bytes"),
        "systolic_util_pct":   r.get("systolic_util"),
        "systolic_0_util_pct": r.get("systolic_0_util"),
        "systolic_1_util_pct": r.get("systolic_1_util"),
        "dma_util_pct":        r.get("dma_util"),
        "vec_util_pct":        r.get("vec_util"),
        "access_util_pct":     r.get("access_util"),
        "roofline_eff_pct":    r.get("roofline_eff"),
        "roofline_bound":      r.get("roofline_bound"),
        "compute_bound_cyc":   r.get("compute_bound_cyc"),
        "memory_bound_cyc":    r.get("memory_bound_cyc"),
        "ttft_ns":             r.get("ttft_ns"),
        "decode_tps":          r.get("decode_tps"),
        # derived KPIs
        "tflops_achieved":        dv.get("tflops_achieved"),
        "hbm_bw_achieved_tb_s":   dv.get("hbm_bw_achieved"),
        "tflops_peak":            dv.get("tflops_peak"),
        "hbm_util_pct":           dv.get("hbm_util_pct"),
        "arith_intensity":        dv.get("arith_intensity"),
        "ttft_per_token_ns":      dv.get("ttft_per_token_ns"),
        "bytes_per_token":        dv.get("bytes_per_token"),
        "systolic_imbalance":     dv.get("systolic_imbalance"),
        "mem_compute_ratio":      dv.get("mem_compute_ratio"),
        "ffn_mac_pct":            dv.get("ffn_pct"),
        "attn_mac_pct":           dv.get("attn_pct"),
        "ffn_attn_crossover_tok": dv.get("crossover"),
        # calibration
        "pub_hbm_util_pct":  pub_hbm_util,
        "pub_tps_bs1":        pub_tps,
        "calib_hbm_err_pct":  calib_err,
        # metadata
        "wall_s":      r.get("wall_s"),
        "peak_rss_mb": ((r.get("rss_kb") or 0) // 1024) or None,
        "status":      "OK" if r.get("ok") else r.get("error", "ERR"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--binary",      default=DEFAULT_BINARY)
    p.add_argument("--hw",          default=DEFAULT_HW)
    p.add_argument("--workload1b",  default=DEFAULT_WL1B)
    p.add_argument("--workload8b",  default=DEFAULT_WL8B)
    p.add_argument("--workload70b", default=DEFAULT_WL70B)
    p.add_argument("--model",       default="8b",
                   choices=["1b", "8b", "70b", "both"])
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--group",       default=None,
                   help="Run only groups whose name starts with PREFIX")
    p.add_argument("--out",         default=DEFAULT_OUTFILE)
    args = p.parse_args()

    models = {"8b", "70b"} if args.model == "both" else {args.model}

    # ── Load configs ──────────────────────────────────────────────────────
    for lbl, path in [("--hw", args.hw),
                      ("--workload1b", args.workload1b),
                      ("--workload8b", args.workload8b),
                      ("--workload70b", args.workload70b)]:
        if not Path(path).exists():
            print(f"WARNING: {lbl} not found: {path}", file=sys.stderr)

    hw    = read_arch(args.hw)             if Path(args.hw).exists()            else None
    wl1b  = read_workload(args.workload1b) if Path(args.workload1b).exists()    else None
    wl8b  = read_workload(args.workload8b) if Path(args.workload8b).exists()    else None
    wl70b = read_workload(args.workload70b) if Path(args.workload70b).exists()  else None

    if not hw:              sys.exit(f"Cannot read --hw: {args.hw}")
    if "1b"  in models and not wl1b:
        sys.exit(f"Cannot read --workload1b: {args.workload1b}")
    if "8b"  in models and not wl8b:
        sys.exit(f"Cannot read --workload8b: {args.workload8b}")
    if "70b" in models and not wl70b:
        sys.exit(f"Cannot read --workload70b: {args.workload70b}")
    if not wl8b:  wl8b  = {}
    if not wl70b: wl70b = {}

    sweep = make_sweep(hw, wl1b, wl8b, wl70b, models)
    if args.group:
        sweep = [e for e in sweep if e[0].startswith(args.group)]
        if not sweep:
            sys.exit(f"No configs match --group '{args.group}'")

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        hdr = (f"{'#':>3}  {'GROUP':<24} {'NAME':<30} {'MDL':<4}"
               f" {'ARR':>8} {'SU':>3} {'BD':>2} {'VC':>2}"
               f" {'SIMD':>4} {'EXP':>3} {'ABW':>4}"
               f" {'BW':>5} {'LAT':>4} {'DMA':>3}"
               f" {'SDB':>3} {'MS':>2}"
               f" {'TILE':>8} {'PLEN':>5} {'GSTEP':>5} {'MSEQ':>5}")
        sep = "-" * len(hdr)
        print(f"--hw={args.hw}  --model={args.model}")
        print(); print(hdr); print(sep)
        for i, (grp, name, _, ae, we, mdl) in enumerate(sweep):
            def b(v): return "Y" if v else "N"
            print(
                f"{i+1:>3}  {grp:<24} {name:<30} {mdl:<4}"
                f" {ae['array_rows']:>3}x{ae['array_cols']:<3}"
                f" {ae['systolic_units']:>3} {b(ae['bidirectional']):>2}"
                f" {ae['vector_cores']:>2}"
                f" {ae['vec_simd']:>4} {ae['exp_lat']:>3} {ae['access_bw']:>4}"
                f" {ae['hbm_bw_tb_s']:>5.2f} {ae['hbm_lat_cycles']:>4}"
                f" {ae.get('dma_channels',1):>3}"
                f" {b(ae['stage_double_buffer']):>3} {b(ae['model_sram']):>2}"
                f" {we['tile_rows']:>3}x{we['tile_cols']:<3}"
                f" {we['prompt_len']:>5} {we.get('gen_steps',1):>5}"
                f" {we.get('max_seq_len',8192):>5}")
        print(sep)
        gc = {}
        for g, *_ in sweep: gc[g] = gc.get(g, 0) + 1
        print(f"\nTotal: {len(sweep)} configs across {len(gc)} groups:")
        for g, c in sorted(gc.items()): print(f"  {g:<32} {c:>3}")
        print(f"\n--model={args.model}   BD=bidir  SDB=stage_double_buffer"
              f"  MS=model_sram  ABW=access_bw  EXP=exp_latency")
        print("Remove --dry-run to execute.")
        return

    if not Path(args.binary).exists():
        sys.exit(f"Binary not found: {args.binary}")

    # Runtime estimate: prefill_decode ~35s for 8B, ~86s for 70B (80 layers).
    # Decode mode cost depends on gen_steps: 1 step ~2s/4s, 32 steps ~64s/128s.
    pd_8b   = sum(1 for e in sweep if e[4].get("mode") == "prefill_decode" and e[5] == "8b")
    pd_70b  = sum(1 for e in sweep if e[4].get("mode") == "prefill_decode" and e[5] == "70b")
    dec_8b_s  = sum(1 for e in sweep if e[4].get("mode") == "decode" and e[5] == "8b"
                    and e[4].get("gen_steps", 1) <= 1)
    dec_8b_l  = sum(1 for e in sweep if e[4].get("mode") == "decode" and e[5] == "8b"
                    and e[4].get("gen_steps", 1) > 1)
    dec_70b_s = sum(1 for e in sweep if e[4].get("mode") == "decode" and e[5] == "70b"
                    and e[4].get("gen_steps", 1) <= 1)
    dec_70b_l = sum(1 for e in sweep if e[4].get("mode") == "decode" and e[5] == "70b"
                    and e[4].get("gen_steps", 1) > 1)
    dec_8b  = dec_8b_s  + dec_8b_l
    dec_70b = dec_70b_s + dec_70b_l
    total_est = (pd_8b*35 + pd_70b*86
                 + dec_8b_s*2   + dec_8b_l*64
                 + dec_70b_s*4  + dec_70b_l*128) // 60
    print(f"Sweep: {len(sweep)} configs (~{total_est} min est.)  "
          f"--model={args.model}  "
          f"[pd_8b={pd_8b} pd_70b={pd_70b} dec_8b={dec_8b} dec_70b={dec_70b}]")
    print(f"  hw={args.hw}  out={args.out}\n")

    # ── Run — write CSV rows incrementally so partial results survive ─────
    rows = []
    t0   = time.time()

    csv_file = open(args.out, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS,
                               extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()

    try:
        for i, (grp, name, desc, ae, we, mdl) in enumerate(sweep):
            ela = int(time.time() - t0)
            eta = int((len(sweep) - i) * ela / max(i, 1)) if i else "?"
            print(f"[{i+1:3}/{len(sweep)}] {grp}/{name} [{mdl}]"
                  f"  t={ela}s eta~{eta}s ... ", end="", flush=True)

            r  = run_sim(args.binary, ae, we)
            dv = derived(r, ae, we) if r.get("ok") else {}

            # Calibration — compare bytes_per_token (sim) vs BYTES_PER_TOKEN
            # (reference). hbm_util_pct is kept as reference info but is NOT
            # the pass/fail metric: the sim's GEMV overhead means hbm_util will
            # always read ~6% for decode on a systolic array at bs=1. What we
            # CAN validate is that the sim reads the correct weight volume per
            # token (~16.27 GB), which bytes_per_token captures correctly.
            pub_hbm_util = pub_tps = calib_err = None
            if grp == "6_calibration":
                chip = name.rsplit("_sdb", 1)[0].rsplit("_base", 1)[0]
                if chip in CALIB:
                    pub_hbm_util = CALIB[chip]["pub_hbm_util_pct"]
                    pub_tps      = CALIB[chip]["pub_tps_bs1"]
                    sim_bpt      = dv.get("bytes_per_token")
                    if sim_bpt is not None:
                        calib_err = round(
                            (sim_bpt - BYTES_PER_TOKEN) / BYTES_PER_TOKEN * 100, 1)

            row = build_row(grp, name, desc, ae, we, mdl, r, dv,
                            pub_hbm_util, pub_tps, calib_err)

            # Write immediately — survives a kill mid-sweep
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

            # Live progress line
            if r.get("ok"):
                cal_str = ""
                if pub_hbm_util is not None:
                    sim_bpt = dv.get("bytes_per_token") or 0
                    ok_str  = "✓" if abs(calib_err or 0) <= 20 else "✗"
                    cal_str = (f" [B/tok={sim_bpt/1e9:.2f}GB"
                               f" ref={BYTES_PER_TOKEN/1e9:.2f}GB"
                               f" err={calib_err:+.1f}% {ok_str}]")
                print(f"sys={r.get('systolic_util') or 0:4.1f}%"
                      f" dma={r.get('dma_util') or 0:4.1f}%"
                      f" rl={r.get('roofline_eff') or 0:5.2f}%"
                      f" AI={dv.get('arith_intensity') or 0:5.1f}"
                      f" mcr={dv.get('mem_compute_ratio') or 0:5.2f}"
                      f" tps={r.get('decode_tps') or 0:6.1f}"
                      f" ttft/tok={dv.get('ttft_per_token_ns') or 0:8.0f}ns"
                      f" {(r.get('roofline_bound') or ''):>14}"
                      f"  {r.get('wall_s',0):.0f}s{cal_str}")
            else:
                print(f"FAILED: {r.get('error', '?')}")

    finally:
        # Append calibration reference block regardless of how we exit
        csv_file.write("\n\n")
        csv_file.write("# ===== CALIBRATION REFERENCE =====\n")
        csv_file.write("# Metric : hbm_util_pct = hbm_bw_achieved / hbm_bw_peak\n")
        csv_file.write("# Target : ±20% of pub_hbm_util_pct  with  stage_double_buffer=ON\n")
        csv_file.write("# NOT absolute TPS (single array ≠ full GPU)\n")
        csv_file.write("# chip,hbm_bw_tb_s,pub_bs1_tps,pub_hbm_util_pct\n")
        for chip, cp in CALIB.items():
            csv_file.write(f"# {chip},{cp['hbm_bw_tb_s']},"
                           f"{cp['pub_tps_bs1']},{cp['pub_hbm_util_pct']}\n")
        csv_file.close()

    # ── Final summaries ───────────────────────────────────────────────────
    total = int(time.time() - t0)
    ok    = sum(1 for r in rows if r["status"] == "OK")
    print(f"\nDone {total//60}m{total%60}s  {ok}/{len(rows)} OK  →  {args.out}")

    # Calibration table
    cal = [r for r in rows if r.get("pub_hbm_util_pct") and r["status"] == "OK"]
    if cal:
        print("\n── Calibration (bytes_per_token vs reference ~16.27 GB) ────────────────")
        print(f"  {'Config':<34} {'sim_GB/tok':>10} {'ref_GB/tok':>10} {'err':>7}  pass?")
        print("  " + "-" * 67)
        for r in cal:
            err    = r.get("calib_hbm_err_pct") or 0
            passed = "✓" if abs(err) <= 20 else "✗  >±20%"
            sim_bpt = (r.get("bytes_per_token") or 0) / 1e9
            print(f"  {r['name']:<34}"
                  f" {sim_bpt:>10.2f}"
                  f" {BYTES_PER_TOKEN/1e9:>10.2f}"
                  f" {err:>+6.1f}%  {passed}")

    # Full summary table
    print("\n── Summary ──────────────────────────────────────────────────────────────────────────")
    print(f"{'GROUP':<24} {'NAME':<28} {'M':<4}"
          f" {'SYS%':>5} {'DMA%':>5} {'RL%':>6}"
          f" {'AI':>5} {'MCR':>5} {'FFN%':>5}"
          f" {'TPS':>7} {'TTFT/tok':>10} {'BOUND':<14}")
    print("-" * 120)
    for r in rows:
        if r["status"] == "OK":
            print(
                f"{r['group']:<24} {r['name']:<28} {r['model']:<4}"
                f" {r['systolic_util_pct']    or 0:5.1f}"
                f" {r['dma_util_pct']         or 0:5.1f}"
                f" {r['roofline_eff_pct']     or 0:6.2f}"
                f" {r['arith_intensity']      or 0:5.1f}"
                f" {r['mem_compute_ratio']    or 0:5.2f}"
                f" {r['ffn_mac_pct']          or 0:5.1f}"
                f" {r['decode_tps']           or 0:7.1f}"
                f" {r['ttft_per_token_ns']    or 0:10.0f}"
                f" {r['roofline_bound']       or '':14}")


if __name__ == "__main__":
    main()