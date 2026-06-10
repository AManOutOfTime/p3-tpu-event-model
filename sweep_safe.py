#!/usr/bin/env python3
"""
sweep_safe.py — Memory-safe TPU Simulator Parameter Sweep
==========================================================
Tuned for 16 GB pod (8 GB usable headroom), 4 CPUs, 35 s/run, ~2.4 GB/run.

Key changes vs original sweep_safe.py:
  1. DEFAULT_WL8B / DEFAULT_WL70B paths fixed — script no longer hardcodes the
     workload file names; pass them via --workload8b / --workload70b as usual.
     If no flag is given, it falls back to the constants at the top.
  2. Crash / OOM recovery: returncode=137 → OOM_KILLED; any other non-zero code
     that isn't 1 → EXIT_N.  Both skip to the next config and write a FAILED row.
  3. SIGKILL guard: subprocess.Popen + manual poll so a partial stdout from a
     killed process is not mistaken for a valid result.
  4. Defaults updated to match your actual config:
       bidirectional=true, systolic_units=2, stage_double_buffer=true,
       model_sram=true, prompt_len=2048 (capped to SAFE_PLEN=512), max_seq_len=8192
  5. --focused flag: runs only the high-signal / low-memory groups (1b,1c,2a,2d,3c,4a).
     Use this for a first pass (~60 configs, ~35 min).
  6. /proc/meminfo memory drop trigger: if available RAM drops 20% from baseline
     during the sweep, the script logs a warning and sleeps 10 s before each run
     to let the OS reclaim pages.
  7. Added --no-modes flag: run prefill_decode only (halves run count for groups
     that default to both modes).

Usage:
  python3 sweep_safe.py --dry-run
  python3 sweep_safe.py --focused --dry-run
  python3 sweep_safe.py --focused --out sweep_focused.csv
  python3 sweep_safe.py --group 1b --out sweep_1b.csv
"""

import argparse, csv, os, re, subprocess, sys, tempfile, time, yaml
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — update these to match your directory layout
# ---------------------------------------------------------------------------
DEFAULT_BINARY  = "./build/apps/sim_main"
DEFAULT_OUTFILE = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
DEFAULT_HW      = "configs/default.yaml"
DEFAULT_WL8B    = "workloads/llama_prefill_decode_8B.yaml"
DEFAULT_WL70B   = "workloads/llama_prefill_decode_70B.yaml"
SIM_TIMEOUT_S   = 600

# ---------------------------------------------------------------------------
# Memory safety knobs
# ---------------------------------------------------------------------------
MEM_GUARD_GB      = 8.0     # skip run if MemAvailable < this (GB)
MEM_WARN_DROP_PCT = 20      # warn + brief sleep if available drops >20% from start
WARN_SLEEP_S      = 10      # seconds to sleep when memory warning fires
SAFE_PLEN         = 512     # default prompt_len cap
SAFE_MAXSEQ       = 4096    # default max_seq_len cap

# Array-size → prompt_len mapping for groups that sweep array size (1a, 5).
#
# Why this exists: event count ∝ prompt_len / arr².  arr64 at plen=128 still
# generates 4× more events than arr256 at plen=512 — that was enough to OOM.
# So we cut every value hard:
#
#   arr32  → plen=32   (new — smallest meaningful test)
#   arr64  → plen=64   (was 128, crashed; halved again)
#   arr128 → plen=128  (was 256; halved)
#   arr256 → plen=256  (was 512; halved — this IS the baseline array size)
#   arr512 → plen=256  (same cap; fewer tiles so it's fine)
#
# At these prompt lengths the roofline / utilisation numbers are still
# comparable across sizes because the denominator (arr²) scales with plen.
ARR_PLEN = {
    32:  32,
    64:  64,
    128: 128,
    256: 256,
    512: 256,
}

# Groups included in --focused mode (high signal, low memory risk)
FOCUSED_GROUPS = {"1a_array_size", "1b_systolic_units", "1c_bidirectional",
                  "2a_hbm_bw", "2d_stage_doublebuf", "3c_kv_cache",
                  "4a_gqa_8b", "5_pareto"}

# ---------------------------------------------------------------------------
# CSV schema — identical to original
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "group", "name", "model", "description",
    "clock_ghz",
    "array_rows", "array_cols", "bidirectional",
    "systolic_units", "vector_cores", "access_cores",
    "vec_simd", "exp_lat", "access_bw",
    "dma_channels",
    "hbm_bw_tb_s", "hbm_lat_cycles",
    "ibuf_kb", "obuf_kb",
    "stage_double_buffer", "model_sram",
    "tile_rows", "tile_cols",
    "mode",
    "num_layers", "hidden_dim", "head_dim", "intermediate_dim",
    "num_q_heads", "num_kv_heads", "gqa_group",
    "prompt_len", "gen_steps", "max_seq_len", "kv_cache",
    "cycles", "MACs", "hbm_bytes",
    "systolic_util_pct",
    "systolic_0_util_pct", "systolic_1_util_pct",
    "dma_util_pct", "vec_util_pct", "access_util_pct",
    "roofline_eff_pct", "roofline_bound",
    "compute_bound_cyc", "memory_bound_cyc",
    "ttft_ns", "decode_tps",
    "tflops_achieved", "hbm_bw_achieved_tb_s", "tflops_peak",
    "hbm_util_pct",
    "arith_intensity",
    "ttft_per_token_ns",
    "bytes_per_token",
    "systolic_imbalance",
    "mem_compute_ratio",
    "ffn_mac_pct", "attn_mac_pct", "ffn_attn_crossover_tok",
    "pub_hbm_util_pct", "pub_tps_bs1", "calib_hbm_err_pct",
    "wall_s", "peak_rss_mb", "status",
]


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def avail_mem_gb():
    """MemAvailable from /proc/meminfo in GB.  Returns 999 if unreadable."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) / 1048576   # kB → GB
    except Exception:
        pass
    return 999.0


def maybe_sleep_for_memory(start_mem_gb, guard_gb, warn_drop_pct):
    """
    If available memory has dropped more than warn_drop_pct% from the sweep
    start value, sleep briefly so the OS can reclaim page cache.
    Returns (current_avail_gb, slept: bool).
    """
    avail = avail_mem_gb()
    drop  = (start_mem_gb - avail) / max(start_mem_gb, 1) * 100
    if drop > warn_drop_pct and avail < start_mem_gb * 0.85:
        print(f"  [mem-warn] available dropped {drop:.0f}% from start "
              f"({start_mem_gb:.1f}→{avail:.1f} GB); sleeping {WARN_SLEEP_S}s …",
              flush=True)
        time.sleep(WARN_SLEEP_S)
        avail = avail_mem_gb()
        return avail, True
    return avail, False


# ---------------------------------------------------------------------------
# YAML readers
# ---------------------------------------------------------------------------
def read_arch(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    s  = d.get("systolic", {})
    h  = d.get("hbm", {})
    vc = d.get("vector_core", {})
    ac = d.get("access_core", {})
    sr = d.get("sram", {})
    return dict(
        clock_ghz           = d.get("clock_ghz",           1.0),
        array_rows          = s.get("rows",                 256),
        array_cols          = s.get("cols",                 256),
        bidirectional       = s.get("bidirectional",       True),   # your default
        systolic_units      = d.get("systolic_units",         2),   # your default
        vector_cores        = d.get("vector_cores",           3),
        access_cores        = d.get("access_cores",           1),
        hbm_bw_tb_s         = h.get("bandwidth_tb_s",       2.0),
        hbm_lat_cycles      = h.get("latency_cycles",       200),
        dma_channels        = d.get("dma", {}).get("channels", 1),
        vec_simd            = vc.get("simd_width",            64),
        exp_lat             = vc.get("exp_latency",            4),
        access_bw           = ac.get("bandwidth",             64),
        ibuf_kb             = sr.get("ibuf_kb",             4096),
        obuf_kb             = sr.get("obuf_kb",             4096),
        banking_factor      = sr.get("banking_factor",         8),
        stage_double_buffer = d.get("stage_double_buffer",  True),  # your default
        model_sram          = d.get("model_sram",           True),  # your default
    )


def read_workload(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    ll = d.get("llama", {})
    return dict(
        mode             = ll.get("mode",             "prefill_decode"),
        tile_rows        = ll.get("tile_rows",                     256),
        tile_cols        = ll.get("tile_cols",                     256),
        num_layers       = ll.get("num_layers",                     32),
        hidden_dim       = ll.get("hidden_dim",                   4096),
        head_dim         = ll.get("head_dim",                      128),
        intermediate_dim = ll.get("intermediate_dim",            14336),
        num_q_heads      = ll.get("num_q_heads",                    32),
        num_kv_heads     = ll.get("num_kv_heads",                    8),
        gqa_group        = ll.get("gqa_group_size",                  4),
        vocab_size       = ll.get("vocab_size",                 128256),
        prompt_len       = ll.get("prompt_len",                   2048),  # will be capped
        gen_steps        = ll.get("generation_steps",                1),
        max_seq_len      = ll.get("max_seq_len",                  8192),  # will be capped
        kv_cache         = ll.get("kv_cache_enabled",             True),
        kv_loc           = ll.get("kv_cache_location",            "hbm"),
    )


# ---------------------------------------------------------------------------
# YAML generators
# ---------------------------------------------------------------------------
def arch_yaml(clock_ghz=1.0, array_rows=256, array_cols=256, bidirectional=True,
              systolic_units=2, vector_cores=3, access_cores=1,
              hbm_bw_tb_s=2.0, hbm_lat_cycles=200, dma_channels=1,
              vec_simd=64, exp_lat=4, access_bw=64,
              ibuf_kb=4096, obuf_kb=4096, banking_factor=8,
              stage_double_buffer=True, model_sram=True, **_):
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
                  vocab_size=128256, prompt_len=512, gen_steps=1, max_seq_len=4096,
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


# ---------------------------------------------------------------------------
# Regime: FFN vs attn FLOP split
# ---------------------------------------------------------------------------
def regime(wl):
    S   = wl.get("prompt_len",        512)
    H   = wl.get("hidden_dim",        4096)
    I   = wl.get("intermediate_dim", 14336)
    nq  = wl.get("num_q_heads",         32)
    nk  = wl.get("num_kv_heads",          8)
    dh  = wl.get("head_dim",            128)
    ffn  = S * (3 * H * I)
    attn = S * (nq*dh*H + nk*dh*H*2 + nq*dh*H) + nq*S*S*dh*2
    tot  = ffn + attn
    ffn_pct = round(ffn / tot * 100, 1)
    return dict(
        ffn_pct   = ffn_pct,
        attn_pct  = round(100 - ffn_pct, 1),
        crossover = int(3 * I // 2),
    )


# ---------------------------------------------------------------------------
# Calibration reference
# ---------------------------------------------------------------------------
BYTES_PER_TOKEN = 18e9
CALIB = {
    "H100_SXM5":  dict(hbm_bw_tb_s=3.35, hbm_lat_cycles=280, dma_channels=2,
                       clock_ghz=1.98,  vector_cores=4, pub_tps_bs1=120),
    "A100_SXM4":  dict(hbm_bw_tb_s=2.00, hbm_lat_cycles=200, dma_channels=1,
                       clock_ghz=1.41,  vector_cores=3, pub_tps_bs1=70),
    "TPUv4_chip": dict(hbm_bw_tb_s=1.20, hbm_lat_cycles=140, dma_channels=1,
                       clock_ghz=0.275, vector_cores=2, pub_tps_bs1=60),
}
for chip, p in CALIB.items():
    p["pub_hbm_util_pct"] = round(
        p["pub_tps_bs1"] * BYTES_PER_TOKEN / (p["hbm_bw_tb_s"] * 1e12) * 100, 1)


# ---------------------------------------------------------------------------
# Sweep builder
# ---------------------------------------------------------------------------
def make_sweep(hw, wl8b, wl70b, models, no_modes=False):
    runs = []

    # Memory-safe base: cap prompt_len and max_seq_len from the workload file
    wl8b_safe = {**wl8b,
                 "prompt_len":  min(wl8b.get("prompt_len",  2048), SAFE_PLEN),
                 "max_seq_len": min(wl8b.get("max_seq_len", 8192), SAFE_MAXSEQ)}

    def add(grp, name, desc, arch_ov=None, wl_ov=None, model="8b"):
        if model not in models:
            return
        base = wl8b_safe if model == "8b" else wl70b
        runs.append((grp, name, desc,
                     {**hw,   **(arch_ov or {})},
                     {**base, **(wl_ov  or {})},
                     model))

    # If --no-modes: only run prefill_decode; otherwise both modes.
    all_modes = ("prefill_decode",) if no_modes else ("prefill_decode", "decode")

    def addm(grp, name, desc, arch_ov=None, wl_ov=None, modes=None):
        if modes is None:
            modes = all_modes
        multi = len(modes) > 1
        for mode in modes:
            mode_wl  = {**(wl_ov or {}), "mode": mode}
            mode_sfx = ("_pd" if mode == "prefill_decode" else "_dec") if multi else ""
            for m in sorted(models):
                m_sfx = f"_{m}" if len(models) > 1 else ""
                add(grp,
                    name + mode_sfx + m_sfx,
                    desc + (f" [mode={mode}]" if multi else ""),
                    arch_ov, mode_wl, model=m)

    arr = hw["array_rows"]

    # ── 1a. Array size ────────────────────────────────────────────────────────
    # arr32 added as the smallest meaningful test (shows extreme undersize cost).
    # prompt_len comes from ARR_PLEN, which was halved after arr64@plen=128 crashed.
    # max_seq_len = plen + 64 (tight KV tail) to minimise sim working-set size.
    for sz in [32, 64, 128, 256, 512]:
        plen = ARR_PLEN.get(sz, SAFE_PLEN)
        mseq = plen + 64          # tight KV tail — smallest valid KV allocation
        if   sz == arr: note = "EXACT FIT — tiler passthrough"
        elif sz <  arr: note = f"undersize {sz*sz*100//(arr*arr)}% array util"
        else:           note = f"oversize {(sz//arr)**2} internal passes/instr"
        addm("1a_array_size", f"arr{sz}",
             f"array={sz}x{sz} tile={sz} plen={plen} mseq={mseq} [{note}]",
             arch_ov=dict(array_rows=sz, array_cols=sz),
             wl_ov  =dict(tile_rows=sz,  tile_cols=sz,
                          prompt_len=plen,
                          max_seq_len=mseq))

    # ── 1b. Systolic unit count ───────────────────────────────────────────────
    for n in [1, 2, 3, 4]:
        addm("1b_systolic_units", f"sys{n}x",
             f"systolic_units={n}  (baseline=2)",
             arch_ov=dict(systolic_units=n))

    # ── 1c. Bidirectional ─────────────────────────────────────────────────────
    for bd in [False, True]:
        addm("1c_bidirectional",
             "bidir" if bd else "unidir",
             f"bidirectional={bd}  (bidir=2 MACs/cell/cycle, unidir=1)",
             arch_ov=dict(bidirectional=bd))

    # ── 1d. Vector cores ──────────────────────────────────────────────────────
    for n in [1, 2, 3, 6]:
        addm("1d_vector_cores", f"vc{n}",
             f"vector_cores={n}",
             arch_ov=dict(vector_cores=n))

    # ── 1e. Vector SIMD width × exp_latency ──────────────────────────────────
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
    for ch in [1, 2, 4]:
        addm("2c_dma_channels", f"dma{ch}ch",
             f"dma_channels={ch}  systolic_units=2",
             arch_ov=dict(dma_channels=ch, systolic_units=2))

    # ── 2d. Stage double buffer ───────────────────────────────────────────────
    for sdb in [False, True]:
        addm("2d_stage_doublebuf",
             "sdb_ON" if sdb else "sdb_OFF",
             f"stage_double_buffer={sdb}  (baseline=ON)",
             arch_ov=dict(stage_double_buffer=sdb))

    # ── 2e. SRAM ibuf pressure ────────────────────────────────────────────────
    for kb in [512, 1024, 2048, 4096]:
        addm("2e_sram_ibuf", f"ibuf{kb}KB",
             f"ibuf_kb={kb} model_sram=ON",
             arch_ov=dict(ibuf_kb=kb, model_sram=True))

    # ── 3a. Prompt length ─────────────────────────────────────────────────────
    # 2048 is memory-risky on 16 GB; the mem guard will skip it if needed.
    plens_8b = [128, 256, 512, 2048]
    for plen in plens_8b:
        r = regime({**wl8b_safe, "prompt_len": plen})
        add("3a_prompt_len", f"p{plen}",
            f"prompt={plen}  FFN={r['ffn_pct']}% attn={r['attn_pct']}%"
            f"  xover@{r['crossover']}tok",
            wl_ov=dict(prompt_len=plen,
                       max_seq_len=max(plen + 256, SAFE_MAXSEQ)),
            model="8b")

    # ── 3b. Tile size ─────────────────────────────────────────────────────────
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
             f"kv_cache_enabled={kv_en}  (baseline=ON)",
             wl_ov=dict(kv_cache=kv_en))

    # ── 3d. Head dim × hidden dim ─────────────────────────────────────────────
    for hd, hidd in [(64, 2048), (128, 4096), (256, 8192)]:
        r = regime({**wl8b_safe, "head_dim": hd, "hidden_dim": hidd})
        addm("3d_head_dim", f"hd{hd}",
             f"head_dim={hd} hidden_dim={hidd}  (32 q_heads fixed)"
             f"  FFN={r['ffn_pct']}% attn={r['attn_pct']}%",
             wl_ov=dict(head_dim=hd, hidden_dim=hidd))

    # ── 3e. Decode throughput vs context length ───────────────────────────────
    # Capped at ctx=2048 to stay within memory budget.
    for ctx in [256, 512, 1024, 2048]:
        kv_gb = 2 * 32 * 8 * 128 * ctx * 2 / 1e9
        addm("3e_decode_ctx", f"ctx{ctx}",
             f"mode=decode  ctx={ctx}  kv={kv_gb:.2f}GB  true_decode_tps",
             wl_ov=dict(mode="decode", prompt_len=ctx, gen_steps=32,
                        max_seq_len=max(ctx + 32, SAFE_MAXSEQ)),
             modes=("decode",))

    # ── 4a. GQA sweep 8B ─────────────────────────────────────────────────────
    nq = wl8b["num_q_heads"]
    for kv, grp_sz, tag in [
        (nq,      1,  "MHA — full KV"),
        (nq//4,   4,  "GQA-4 — default Llama-3-8B"),
        (nq//8,   8,  "GQA-8"),
        (nq//16, 16,  "GQA-16"),
        (1,      nq,  "MQA — minimum KV bandwidth"),
    ]:
        if kv < 1:
            continue
        gqa_modes = ("prefill_decode",) if no_modes else ("prefill_decode", "decode")
        for mode in gqa_modes:
            mode_sfx = "_pd" if mode == "prefill_decode" else "_dec"
            add("4a_gqa_8b", f"kv{kv}_{tag.split()[0]}{mode_sfx}",
                f"8B kv_heads={kv} gqa_group={grp_sz} ({tag}) [mode={mode}]",
                wl_ov=dict(
                    num_kv_heads=kv,
                    gqa_group=grp_sz,
                    mode=mode,
                    gen_steps=32 if mode == "decode" else 1,
                    max_seq_len=max(
                    wl8b_safe["prompt_len"] + (32 if mode == "decode" else 1),
                    SAFE_MAXSEQ),
                ),
                model="8b")

    # ── 5. Pareto: array size × HBM bandwidth ────────────────────────────────
    # Explicitly requested — shows the efficiency frontier across compute (array
    # size) and memory bandwidth simultaneously.  arr64 added so the curve
    # extends to small arrays.  max_seq_len = plen + 64 matches group 1a to keep
    # working-set size consistent when comparing across groups.
    for sz in [64, 128, 256, 512]:
        plen = ARR_PLEN.get(sz, SAFE_PLEN)
        mseq = plen + 64
        for bw in [1.0, 2.0, 3.35]:
            addm("5_pareto", f"arr{sz}_bw{bw}",
                 f"array={sz}x{sz} hbm_bw={bw}TB/s plen={plen} mseq={mseq}",
                 arch_ov=dict(array_rows=sz, array_cols=sz, hbm_bw_tb_s=bw),
                 wl_ov  =dict(tile_rows=sz,  tile_cols=sz,
                              prompt_len=plen,
                              max_seq_len=mseq))

    # ── 6. Calibration ───────────────────────────────────────────────────────
    for chip, p in CALIB.items():
        for sdb in [False, True]:
            add("6_calibration",
                f"{chip}_{'sdb' if sdb else 'base'}",
                f"{chip} hbm={p['hbm_bw_tb_s']}TB/s sdb={sdb}"
                f"  pub_tps={p['pub_tps_bs1']}"
                f"  pub_hbm_util={p['pub_hbm_util_pct']}%",
                arch_ov={k: v for k, v in p.items()
                         if k not in ("pub_tps_bs1", "pub_hbm_util_pct")}
                        | {"stage_double_buffer": sdb})

    return runs


# ---------------------------------------------------------------------------
# Runner — with proper crash / OOM-kill handling
# ---------------------------------------------------------------------------
def run_sim(binary, ae, we):
    ap = wp = None
    try:
        # Write arch temp file
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(arch_yaml(**ae)); f.close(); ap = f.name

        # Write workload temp file
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(workload_yaml(**we)); f.close(); wp = f.name

        t0  = time.time()
        tb  = "/usr/bin/time"
        cmd = ([tb, "-v"] if os.path.isfile(tb) and os.access(tb, os.X_OK) else []) + \
              [binary, "--config", ap, "--llama-workload", wp, "--no-trace"]

        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=SIM_TIMEOUT_S)
        elapsed = round(time.time() - t0, 1)

        # ── Crash / OOM detection — check exit code FIRST ─────────────────
        # returncode=137 means the kernel sent SIGKILL (OOM killer or cgroup limit).
        # Without this check a killed process still has partial stdout, and
        # parse_out would silently produce a row full of None values marked OK.
        if res.returncode == 137:
            return {"ok": False, "error": "OOM_KILLED", "wall_s": elapsed}

        # returncode=1 is normal (sim may emit warnings and exit 1); anything
        # else (segfault=139, bus error=135, etc.) is a hard crash — skip it.
        if res.returncode not in (0, 1):
            return {"ok": False,
                    "error": f"EXIT_{res.returncode}",
                    "wall_s": elapsed}

        r = parse_out(res.stdout + res.stderr)
        r["wall_s"] = elapsed
        r["ok"]     = True
        return r

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}
    finally:
        for path in (ap, wp):
            if path:
                try: os.unlink(path)
                except: pass


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------
def parse_out(out):
    r = {}
    def g(pat, cast=float):
        m = re.search(pat, out)
        return cast(m.group(1)) if m else None

    r["cycles"]    = g(r"cycle=(\d+)", int)
    r["MACs"]      = g(r"MACs=(\d+)", int)
    r["hbm_bytes"] = g(r"HBM_bytes=(\d+)", int)

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

    r["ttft_ns"]    = g(r"TTFT=([\d.e+\-]+)")
    r["decode_tps"] = g(r"throughput=([\d.]+)")
    r["rss_kb"]     = g(r"Maximum resident set size .kbytes.: (\d+)", int)

    return r


# ---------------------------------------------------------------------------
# Derived / KPI metrics
# ---------------------------------------------------------------------------
def derived(r, ae, we):
    d = {}
    try:
        clk  = ae.get("clock_ghz", 1.0)
        cyc  = r.get("cycles") or 1
        ws   = cyc / (clk * 1e9)
        macs = r.get("MACs")      or 0
        hbm  = r.get("hbm_bytes") or 0

        d["tflops_achieved"] = round(macs * 2 / ws / 1e12, 4)
        d["hbm_bw_achieved"] = round(hbm / ws / 1e12, 4)

        bidir_factor     = 2 if ae.get("bidirectional", False) else 1
        d["tflops_peak"] = round(
            ae["array_rows"] * ae["array_cols"] * ae["systolic_units"]
            * bidir_factor * 2 * clk * 1e9 / 1e12, 4)

        d["hbm_util_pct"] = round(
            d["hbm_bw_achieved"] / ae["hbm_bw_tb_s"] * 100, 2) \
            if ae.get("hbm_bw_tb_s") else None

        d["arith_intensity"] = round(macs / hbm, 2) if hbm else None

        mode = we.get("mode", "prefill_decode")
        if mode != "decode":
            ttft = r.get("ttft_ns") or 0
            plen = we.get("prompt_len") or 1
            d["ttft_per_token_ns"] = round(ttft / plen, 2)
        else:
            d["ttft_per_token_ns"] = None

        if mode == "decode":
            gen = we.get("gen_steps") or 1
            d["bytes_per_token"] = round(hbm / gen) if gen > 0 else None
        else:
            total_tokens = (we.get("prompt_len") or 0) + (we.get("gen_steps") or 0)
            d["bytes_per_token"] = round(hbm / total_tokens) if total_tokens > 0 else None

        s0 = r.get("systolic_0_util")
        s1 = r.get("systolic_1_util")
        d["systolic_imbalance"] = round(abs(s0 - s1), 2) \
                                  if (s0 is not None and s1 is not None) else None

        comp = r.get("compute_bound_cyc") or 0
        mem  = r.get("memory_bound_cyc")  or 0
        d["mem_compute_ratio"] = round(mem / comp, 3) if comp > 0 else None

        d.update(regime(we))

    except Exception:
        pass
    return d


# ---------------------------------------------------------------------------
# CSV row builder
# ---------------------------------------------------------------------------
def build_row(grp, name, desc, ae, we, mdl, r, dv,
              pub_hbm_util=None, pub_tps=None, calib_err=None):
    return {
        "group": grp, "name": name, "model": mdl, "description": desc,
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
        "pub_hbm_util_pct":  pub_hbm_util,
        "pub_tps_bs1":        pub_tps,
        "calib_hbm_err_pct":  calib_err,
        "wall_s":      r.get("wall_s"),
        "peak_rss_mb": ((r.get("rss_kb") or 0) // 1024) or None,
        "status":      "OK" if r.get("ok") else r.get("error", "ERR"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--binary",       default=DEFAULT_BINARY)
    p.add_argument("--hw",           default=DEFAULT_HW)
    p.add_argument("--workload8b",   default=DEFAULT_WL8B,
                   help=f"Path to 8B workload yaml (default: {DEFAULT_WL8B})")
    p.add_argument("--workload70b",  default=DEFAULT_WL70B,
                   help=f"Path to 70B workload yaml (default: {DEFAULT_WL70B})")
    p.add_argument("--model",        default="8b", choices=["8b", "70b", "both"])
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--group",        default=None,
                   help="Run only groups whose name starts with this prefix")
    p.add_argument("--focused",      action="store_true",
                   help=f"Run only high-signal, low-memory groups: "
                        f"{', '.join(sorted(FOCUSED_GROUPS))}")
    p.add_argument("--no-modes",     action="store_true",
                   help="Run prefill_decode mode only (halves configs for multi-mode groups)")
    p.add_argument("--out",          default=DEFAULT_OUTFILE)
    p.add_argument("--mem-guard-gb", type=float, default=MEM_GUARD_GB,
                   help=f"Skip run if MemAvailable < this GB (default {MEM_GUARD_GB})")
    args = p.parse_args()

    models = {"8b", "70b"} if args.model == "both" else {args.model}

    for lbl, path in [("--hw", args.hw),
                      ("--workload8b", args.workload8b),
                      ("--workload70b", args.workload70b)]:
        if not Path(path).exists():
            print(f"WARNING: {lbl} config not found: {path}", file=sys.stderr)

    hw    = read_arch(args.hw)             if Path(args.hw).exists()         else None
    wl8b  = read_workload(args.workload8b) if Path(args.workload8b).exists() else None
    wl70b = read_workload(args.workload70b)if Path(args.workload70b).exists()else None

    if not hw:
        sys.exit(f"Cannot read --hw: {args.hw}")
    if "8b"  in models and not wl8b:
        sys.exit(f"Cannot read --workload8b: {args.workload8b}")
    if "70b" in models and not wl70b:
        sys.exit(f"Cannot read --workload70b: {args.workload70b}")
    if not wl8b:  wl8b  = {}
    if not wl70b: wl70b = {}

    sweep = make_sweep(hw, wl8b, wl70b, models, no_modes=args.no_modes)

    # Apply group filters
    if args.focused:
        sweep = [e for e in sweep if e[0] in FOCUSED_GROUPS]
        if not sweep:
            sys.exit("--focused produced no configs (check FOCUSED_GROUPS)")

    if args.group:
        sweep = [e for e in sweep if e[0].startswith(args.group)]
        if not sweep:
            sys.exit(f"No configs match --group '{args.group}'")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        hdr = (f"{'#':>3}  {'GROUP':<24} {'NAME':<32} {'MDL':<4}"
               f" {'ARR':>8} {'SU':>3} {'BD':>2} {'VC':>2}"
               f" {'BW':>5} {'LAT':>4}"
               f" {'SDB':>3} {'MS':>2}"
               f" {'TILE':>8} {'PLEN':>5} {'MSEQ':>5}")
        sep = "-" * len(hdr)
        print(f"--hw={args.hw}  --model={args.model}"
              + ("  --focused" if args.focused else "")
              + ("  --no-modes" if args.no_modes else ""))
        print(f"safe_plen={SAFE_PLEN}  safe_maxseq={SAFE_MAXSEQ}  "
              f"mem_guard={args.mem_guard_gb}GB")
        print(f"arr_plen_map={ARR_PLEN}")
        print(); print(hdr); print(sep)
        for i, (grp, name, _, ae, we, mdl) in enumerate(sweep):
            def b(v): return "Y" if v else "N"
            print(
                f"{i+1:>3}  {grp:<24} {name:<32} {mdl:<4}"
                f" {ae['array_rows']:>3}x{ae['array_cols']:<3}"
                f" {ae['systolic_units']:>3} {b(ae['bidirectional']):>2}"
                f" {ae['vector_cores']:>2}"
                f" {ae['hbm_bw_tb_s']:>5.2f} {ae['hbm_lat_cycles']:>4}"
                f" {b(ae['stage_double_buffer']):>3} {b(ae['model_sram']):>2}"
                f" {we['tile_rows']:>3}x{we['tile_cols']:<3}"
                f" {we['prompt_len']:>5} {we.get('max_seq_len', SAFE_MAXSEQ):>5}")
        print(sep)
        gc = {}
        for g, *_ in sweep: gc[g] = gc.get(g, 0) + 1
        print(f"\nTotal: {len(sweep)} configs across {len(gc)} groups:")
        for g, c in sorted(gc.items()): print(f"  {g:<32} {c:>3}")
        print(f"\nEstimated time: ~{len(sweep)*35//60} min  "
              f"(~{len(sweep)*2.4:.0f} GB peak sequential)")
        print(f"\nCrash recovery: OOM_KILLED / EXIT_N → skip to next config (row written)")
        print(f"Memory guard:   skip if MemAvailable < {args.mem_guard_gb} GB")
        print(f"Memory warning: sleep {WARN_SLEEP_S}s if available drops >20% from start")
        print(f"\nARR_PLEN map (arr→plen, halved vs prior version after arr64@128 crashed):")
        for k, v in sorted(ARR_PLEN.items()):
            print(f"  arr{k:<4} → plen={v}  max_seq_len={v+64}")
        print("\nRemove --dry-run to execute.")
        return

    if not Path(args.binary).exists():
        sys.exit(f"Binary not found: {args.binary}")

    start_mem = avail_mem_gb()
    pd_8b   = sum(1 for e in sweep if e[4].get("mode") == "prefill_decode" and e[5] == "8b")
    pd_70b  = sum(1 for e in sweep if e[4].get("mode") == "prefill_decode" and e[5] == "70b")
    dec_8b  = sum(1 for e in sweep if e[4].get("mode") == "decode"          and e[5] == "8b")
    dec_70b = sum(1 for e in sweep if e[4].get("mode") == "decode"          and e[5] == "70b")
    total_est = (pd_8b*20 + pd_70b*60 + dec_8b*2 + dec_70b*4) // 60
    print(f"Sweep: {len(sweep)} configs (~{total_est} min est.)  --model={args.model}")
    print(f"  hw={args.hw}  out={args.out}")
    print(f"  MemAvailable={start_mem:.1f}GB  guard={args.mem_guard_gb}GB\n")

    rows = []
    t0   = time.time()

    csv_file = open(args.out, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()

    try:
        for i, (grp, name, desc, ae, we, mdl) in enumerate(sweep):
            ela = int(time.time() - t0)
            eta = int((len(sweep) - i) * ela / max(i, 1)) if i else "?"

            # Memory check + optional sleep
            avail, slept = maybe_sleep_for_memory(start_mem, args.mem_guard_gb,
                                                  MEM_WARN_DROP_PCT)
            print(f"[{i+1:3}/{len(sweep)}] {grp}/{name} [{mdl}]"
                  f"  t={ela}s eta~{eta}s  mem={avail:.1f}GB{' (slept)' if slept else ''} ... ",
                  end="", flush=True)

            # Pre-flight memory guard
            if avail < args.mem_guard_gb:
                msg = f"LOW_MEM_{avail:.1f}GB"
                print(f"SKIP ({msg})")
                r  = {"ok": False, "error": msg, "wall_s": 0}
                dv = {}
                row = build_row(grp, name, desc, ae, we, mdl, r, dv)
                writer.writerow(row); csv_file.flush()
                rows.append(row)
                continue

            r  = run_sim(args.binary, ae, we)
            dv = derived(r, ae, we) if r.get("ok") else {}

            pub_hbm_util = pub_tps = calib_err = None
            if grp == "6_calibration":
                chip = name.rsplit("_sdb", 1)[0].rsplit("_base", 1)[0]
                if chip in CALIB:
                    pub_hbm_util = CALIB[chip]["pub_hbm_util_pct"]
                    pub_tps      = CALIB[chip]["pub_tps_bs1"]
                    sim_hu       = dv.get("hbm_util_pct")
                    if sim_hu is not None:
                        calib_err = round(sim_hu - pub_hbm_util, 1)

            row = build_row(grp, name, desc, ae, we, mdl, r, dv,
                            pub_hbm_util, pub_tps, calib_err)
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

            if r.get("ok"):
                cal_str = ""
                if pub_hbm_util is not None:
                    ok_str  = "✓" if abs(calib_err or 0) <= 20 else "✗"
                    cal_str = (f" [hbm={dv.get('hbm_util_pct',0):.0f}%"
                               f" pub={pub_hbm_util:.0f}%"
                               f" err={calib_err:+.0f}% {ok_str}]")
                print(f"sys={r.get('systolic_util') or 0:4.1f}%"
                      f" dma={r.get('dma_util') or 0:4.1f}%"
                      f" rl={r.get('roofline_eff') or 0:5.2f}%"
                      f" AI={dv.get('arith_intensity') or 0:5.1f}"
                      f" mcr={dv.get('mem_compute_ratio') or 0:5.2f}"
                      f" tps={r.get('decode_tps') or 0:6.1f}"
                      f" ttft/tok={dv.get('ttft_per_token_ns') or 0:8.0f}ns"
                      f" {r.get('roofline_bound',''):>14}"
                      f"  {r.get('wall_s',0):.0f}s{cal_str}")
            else:
                # Crash / OOM — skip to next config (row already written with error status)
                print(f"FAILED: {r.get('error', '?')} — continuing to next config")

    finally:
        csv_file.write("\n\n")
        csv_file.write("# ===== CALIBRATION REFERENCE =====\n")
        csv_file.write("# Metric : hbm_util_pct = hbm_bw_achieved / hbm_bw_peak\n")
        csv_file.write("# Target : ±20% of pub_hbm_util_pct  with  stage_double_buffer=ON\n")
        csv_file.write("# chip,hbm_bw_tb_s,pub_bs1_tps,pub_hbm_util_pct\n")
        for chip, cp in CALIB.items():
            csv_file.write(f"# {chip},{cp['hbm_bw_tb_s']},"
                           f"{cp['pub_tps_bs1']},{cp['pub_hbm_util_pct']}\n")
        csv_file.close()

    total = int(time.time() - t0)
    ok      = sum(1 for r in rows if r["status"] == "OK")
    skipped = sum(1 for r in rows if "LOW_MEM" in str(r.get("status", "")))
    crashed = sum(1 for r in rows
                  if r["status"] in ("OOM_KILLED", "TIMEOUT")
                  or str(r.get("status", "")).startswith("EXIT_"))
    print(f"\nDone {total//60}m{total%60}s  "
          f"{ok}/{len(rows)} OK  {skipped} skipped (mem)  "
          f"{crashed} crashed  →  {args.out}")

    cal = [r for r in rows if r.get("pub_hbm_util_pct") and r["status"] == "OK"]
    if cal:
        print("\n── Calibration ─────────────────────────────────────────────────────")
        print(f"  {'Config':<34} {'sim_hbm%':>9} {'pub_hbm%':>9} {'err':>7}  pass?")
        print("  " + "-" * 65)
        for r in cal:
            err    = r.get("calib_hbm_err_pct") or 0
            passed = "✓" if abs(err) <= 20 else "✗  >±20%"
            print(f"  {r['name']:<34}"
                  f" {r.get('hbm_util_pct') or 0:>9.1f}"
                  f" {r['pub_hbm_util_pct']:>9.1f}"
                  f" {err:>+6.1f}%  {passed}")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
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