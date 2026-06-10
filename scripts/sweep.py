#!/usr/bin/env python3
"""
sweep.py — Comprehensive TPU Simulator Parameter Sweep
=======================================================
Usage:
  python3 scripts/sweep.py --dry-run [--group PREFIX]
  python3 scripts/sweep.py --hw configs/default.yaml \\
                           --workload8b workloads/llama_8b.yaml \\
                           --workload70b workloads/llama_70b.yaml

Arguments:
  --hw FILE         Base HW config. Compute/memory sweeps vary ONE param around
                    this file's values. Default: configs/default.yaml
  --workload8b FILE 8B LLM workload (default: workloads/llama_8b.yaml)
  --workload70b FILE 70B LLM workload (default: workloads/llama_70b.yaml)
  --dry-run         Print config table, exit.
  --group PREFIX    Only run groups matching prefix.
  --out FILE        Output CSV path.

Sweep groups
------------
  0_platform        FILE configs: datacenter.yaml + edge_device.yaml × {8B, 70B}
  1_compute         Vary clock / array-size / #systolic-units around --hw
  2_memory          Vary HBM-bw / HBM-latency / DMA-channels around --hw
  3_sw              Vary prompt-len / tile-size / KV-cache / stage-double-buffer
  4_gqa             Vary GQA group (MHA → MQA) on 8B and 70B
  5_pareto          2D grid: array-size × HBM-bw (Pareto frontier data)
  6_calibration     Approximate real HW (H100/TPUv4/A100/M3); compare to published

Calibration target (project requirement): within ±20% of published roofline
efficiency for compute-bound prefill and memory-bound decode.
"""

import argparse, csv, os, re, subprocess, sys, tempfile, time, yaml
from datetime import datetime
from pathlib import Path

DEFAULT_BINARY    = "./build/apps/sim_main"
DEFAULT_OUTFILE   = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
DEFAULT_HW        = "configs/default.yaml"
DEFAULT_WL8B      = "workloads/llama_8b.yaml"
DEFAULT_WL70B     = "workloads/llama_70b.yaml"
SIM_TIMEOUT_S     = 300


# ── YAML readers ─────────────────────────────────────────────────────────────
def read_arch(path: str) -> dict:
    """Read arch YAML → flat param dict suitable for arch_yaml(**d)."""
    with open(path) as f: d = yaml.safe_load(f)
    s = d.get("systolic", {}); h = d.get("hbm", {})
    vc = d.get("vector_core", {}); ac = d.get("access_core", {})
    sram = d.get("sram", {})
    return dict(
        clock_ghz      = d.get("clock_ghz",      1.0),
        array_rows     = s.get("rows",            256),
        array_cols     = s.get("cols",            256),
        bidirectional  = s.get("bidirectional",  False),
        systolic_units = d.get("systolic_units",   1),
        vector_cores   = d.get("vector_cores",     3),
        access_cores   = d.get("access_cores",     1),
        hbm_bw_tb_s    = h.get("bandwidth_tb_s", 2.0),
        hbm_lat_cycles = h.get("latency_cycles", 200),
        dma_channels   = d.get("dma",{}).get("channels", 1),
        vec_simd       = vc.get("simd_width",     64),
        exp_lat        = vc.get("exp_latency",     4),
        access_bw      = ac.get("bandwidth",      64),
        ibuf_kb        = sram.get("ibuf_kb",    4096),
        obuf_kb        = sram.get("obuf_kb",    4096),
        stage_double_buffer = d.get("stage_double_buffer", False),
    )

def read_workload(path: str) -> dict:
    with open(path) as f: d = yaml.safe_load(f)
    ll = d.get("llama", {})
    return dict(
        tile_rows  = ll.get("tile_rows",   256),
        tile_cols  = ll.get("tile_cols",   256),
        num_layers = ll.get("num_layers",   32),
        hidden_dim = ll.get("hidden_dim", 4096),
        num_q_heads= ll.get("num_q_heads",  32),
        num_kv_heads=ll.get("num_kv_heads",  8),
        gqa_group  = ll.get("gqa_group_size",4),
        prompt_len = ll.get("prompt_len", 2048),
        kv_cache   = ll.get("kv_cache_enabled", True),
        kv_loc     = ll.get("kv_cache_location","hbm"),
        intermediate_dim = ll.get("intermediate_dim", 14336),
        head_dim   = ll.get("head_dim", 128),
        vocab_size = ll.get("vocab_size", 128256),
        gen_steps  = ll.get("generation_steps", 1),
    )


# ── YAML generators ───────────────────────────────────────────────────────────
def arch_yaml(clock_ghz=1.0, array_rows=256, array_cols=256, bidirectional=False,
              systolic_units=1, vector_cores=3, access_cores=1,
              hbm_bw_tb_s=2.0, hbm_lat_cycles=200, dma_channels=1,
              vec_simd=64, exp_lat=4, access_bw=64, ibuf_kb=4096, obuf_kb=4096,
              stage_double_buffer=False, **_):
    db = "true"; sb = "true" if stage_double_buffer else "false"
    bd = "true" if bidirectional else "false"
    return (
        f"clock_ghz: {clock_ghz}\n"
        f"systolic:\n  rows: {array_rows}\n  cols: {array_cols}\n"
        f"  precision: BF16\n  bidirectional: {bd}\n  d_head: 128\n"
        f"  dataflow: weight_stationary\n  weight_load_cycles: 0\n"
        f"  weight_double_buffer: {db}\n"
        f"structural_k_tiling: false\nmodel_sram: false\n"
        f"stage_double_buffer: {sb}\n"
        f"systolic_units: {systolic_units}\nvector_cores: {vector_cores}\n"
        f"access_cores: {access_cores}\n"
        f"sram:\n  ibuf_kb: {ibuf_kb}\n  obuf_kb: {obuf_kb}\n"
        f"  banking_factor: 8\n  private_vector_kb: 512\n"
        f"hbm:\n  bandwidth_tb_s: {hbm_bw_tb_s}\n"
        f"  latency_cycles: {hbm_lat_cycles}\n  pipelined: true\n"
        f"dma:\n  channels: {dma_channels}\n"
        f"vector_core:\n  simd_width: {vec_simd}\n  exp_latency: {exp_lat}\n"
        f"access_core:\n  bandwidth: {access_bw}\n")

def workload_yaml(num_layers=32, num_q_heads=32, num_kv_heads=8, gqa_group=4,
                  head_dim=128, hidden_dim=4096, intermediate_dim=14336,
                  vocab_size=128256, prompt_len=2048, gen_steps=1,
                  tile_rows=256, tile_cols=256, kv_cache=True, kv_loc="hbm", **_):
    return (
        f"llama:\n  mode: prefill_decode\n  prompt_len: {prompt_len}\n"
        f"  generation_steps: {gen_steps}\n  num_layers: {num_layers}\n"
        f"  num_q_heads: {num_q_heads}\n  num_kv_heads: {num_kv_heads}\n"
        f"  gqa_group_size: {gqa_group}\n  head_dim: {head_dim}\n"
        f"  hidden_dim: {hidden_dim}\n  intermediate_dim: {intermediate_dim}\n"
        f"  vocab_size: {vocab_size}\n  max_seq_len: 8192\n  dtype_bytes: 2\n"
        f"  tile_rows: {tile_rows}\n  tile_cols: {tile_cols}\n"
        f"  kv_cache_enabled: {'true' if kv_cache else 'false'}\n"
        f"  kv_cache_location: {kv_loc}\n")


# ── Calibration hardware approximations ──────────────────────────────────────
# Model each chip as a single 256×256 array unit with scaled clock/bandwidth.
# Comparison metric: roofline_efficiency and TPS for batch=1 decode.
# Published bs=1 decode TPS from llama.cpp / vLLM single-stream benchmarks.
CALIB_HW = {
    "H100_SXM5":  dict(clock_ghz=1.98, array_rows=256, array_cols=256,
                       systolic_units=1,  hbm_bw_tb_s=3.35, hbm_lat_cycles=280,
                       vector_cores=4, dma_channels=2),
    "A100_SXM4":  dict(clock_ghz=1.41, array_rows=256, array_cols=256,
                       systolic_units=1,  hbm_bw_tb_s=2.00, hbm_lat_cycles=200,
                       vector_cores=3, dma_channels=1),
    "TPUv4_chip": dict(clock_ghz=0.275, array_rows=256, array_cols=256,
                       systolic_units=1,  hbm_bw_tb_s=1.20, hbm_lat_cycles=140,
                       vector_cores=2, dma_channels=1),
    "AppleM3Max": dict(clock_ghz=4.05, array_rows=32,  array_cols=32,
                       systolic_units=1,  hbm_bw_tb_s=0.30, hbm_lat_cycles=70,
                       vector_cores=2, dma_channels=1),
    "RTX4090":    dict(clock_ghz=2.52, array_rows=128, array_cols=128,
                       systolic_units=1,  hbm_bw_tb_s=1.00, hbm_lat_cycles=200,
                       vector_cores=4, dma_channels=1),
}
# Published bs=1 decode tok/s for LLaMA-3-8B  (sources: llama.cpp, vLLM, mlc-llm)
PUBLISHED_TPS = {
    "H100_SXM5":  120,   # vLLM single-stream; H100 80GB SXM5
    "A100_SXM4":  80,    # llama.cpp / vLLM single-stream
    "TPUv4_chip": 70,    # Google Cloud TPU v4; per-chip estimate
    "AppleM3Max": 55,    # llama.cpp Metal, 18.4 GB/s effective per token
    "RTX4090":    70,    # llama.cpp CUDA, Q8_0
}
# Expected roofline efficiency for bs=1 decode: 3-8% (memory-bound)
# Expected roofline efficiency for large-batch prefill: 40-55% (compute-bound)


# ── Sweep builder ─────────────────────────────────────────────────────────────
def make_sweep(hw: dict, wl8b: dict, wl70b: dict,
               hw_path: str, wl8b_path: str, wl70b_path: str):
    """
    hw    : flat arch params read from --hw file (used as base for gen sweeps)
    wl8b  : flat workload params from --workload8b
    wl70b : flat workload params from --workload70b
    *_path: original file paths (used for FILE entries)
    """
    runs = []

    def file(grp, name, desc, cfg_path, wl_path):
        runs.append((grp, name, desc, {"_file": cfg_path}, {"_file": wl_path}))

    def gen(grp, name, desc, arch_ov=None, wl_ov=None, wl_base=None):
        """Vary one axis; everything else from user's --hw / --workload8b."""
        base_wl = wl_base if wl_base is not None else wl8b
        a = {**hw,      **(arch_ov or {})}
        w = {**base_wl, **(wl_ov  or {})}
        runs.append((grp, name, desc, a, w))

    # ── 0. Platform comparison (FILE configs) ─────────────────────────────────
    # Always parse datacenter.yaml and edge_device.yaml from disk; user can
    # edit those files freely and re-run.
    for cfg_tag, cfg_file in [("datacenter", "configs/datacenter.yaml"),
                               ("edge",       "configs/edge_device.yaml"),
                               ("user_hw",    hw_path)]:
        if not Path(cfg_file).exists(): continue
        for wl_tag, wl_file in [("8B", wl8b_path), ("70B", wl70b_path)]:
            if not Path(wl_file).exists(): continue
            file("0_platform", f"{cfg_tag}_{wl_tag}",
                 f"{cfg_file} + {wl_file}", cfg_file, wl_file)

    # ── 1. Compute sweep (around user's --hw config) ──────────────────────────
    for clk in [0.5, 1.0, 2.0, 3.0]:
        gen("1a_clock", f"clk_{clk}GHz",
            f"--hw + clock_ghz={clk}",
            arch_ov=dict(clock_ghz=clk))

    # Array size: when tile == array, tiler is a passthrough (perfect fit).
    for sz in [64, 128, 256, 512]:
        note = "PERFECT FIT tiler passthrough" if sz == hw["array_rows"] else \
               ("array<tile systolic sub-tiles" if sz < hw["array_rows"] else
                "tile>array systolic does >1 pass")
        gen("1b_array_size", f"array_{sz}x{sz}",
            f"--hw + array={sz}x{sz}  tiles={sz}  [{note}]",
            arch_ov=dict(array_rows=sz, array_cols=sz),
            wl_ov=dict(tile_rows=sz, tile_cols=sz))

    for n in [1, 2, 3, 4]:
        gen("1c_systolic_units", f"sys_{n}x",
            f"--hw + systolic_units={n}",
            arch_ov=dict(systolic_units=n))

    # Bidirectional doubles throughput at same clock (two MACs per cycle per cell)
    for bd in [False, True]:
        tag = "bidir" if bd else "unidir"
        gen("1d_bidirectional", f"{tag}",
            f"--hw + bidirectional={bd}  (bidir=2 MACs/cell/cyc)",
            arch_ov=dict(bidirectional=bd))

    # ── 2. Memory sweep (around user's --hw config) ───────────────────────────
    for bw, tag in [(0.5,"LPDDR5X"),(1.0,"TPUv4"),(2.0,"A100"),(3.35,"H100")]:
        gen("2a_hbm_bw", f"bw_{bw}TB",
            f"--hw + hbm_bw={bw} TB/s ({tag})",
            arch_ov=dict(hbm_bw_tb_s=bw))

    for lat in [50, 100, 200, 400]:
        gen("2b_hbm_lat", f"lat_{lat}cyc",
            f"--hw + hbm_lat={lat} cycles",
            arch_ov=dict(hbm_lat_cycles=lat))

    # DMA channels: 2 arrays set here to show pipelining effect.
    for ch in [1, 2, 4]:
        gen("2c_dma_channels", f"dma_{ch}ch",
            f"--hw + dma_channels={ch}  (2 systolic units to show overlap)",
            arch_ov=dict(dma_channels=ch, systolic_units=2))

    # Stage double-buffer: overlap DMA prefetch and compute.
    # Off→on should close most of the roofline gap (15x serialisation penalty).
    for sdb in [False, True]:
        tag = "ON" if sdb else "OFF_baseline"
        gen("2d_stage_doublebuf", f"sdb_{tag}",
            f"--hw + stage_double_buffer={sdb}  "
            f"({'overlaps DMA+compute → roofline gap shrinks' if sdb else 'serialized baseline'})",
            arch_ov=dict(stage_double_buffer=sdb))

    # ── 3. SW sweep ───────────────────────────────────────────────────────────
    # Prompt length: short=decode-dominated (memory), long=attention O(n²) grows.
    for plen in [128, 256, 512, 1024, 2048, 4096]:
        regime = "decode-dominated (memory-bound)" if plen <= 256 else \
                 "balanced" if plen <= 1024 else "prefill-dominated / attention O(n²)"
        gen("3a_prompt_len", f"p{plen}",
            f"--hw + prompt_len={plen}  [{regime}]",
            wl_ov=dict(prompt_len=plen))

    # Tile size relative to array size (hw["array_rows"]):
    # tile == array  → tiler is passthrough, 1 array pass per instruction (best)
    # tile <  array  → underutilizes array rows/cols
    # tile >  array  → systolic unit does ceil(tile/array)² internal passes
    arr = hw["array_rows"]
    for ts in [64, 128, 256, 512]:
        if ts == arr:
            note = f"EXACT FIT ({ts}=={arr}) tiler passthrough, max efficiency"
        elif ts < arr:
            eff = round((ts*ts)/(arr*arr)*100,1)
            note = f"undersize ({ts}<{arr}): {eff}% array utilisation, {(arr//ts)**2}× more instr"
        else:
            passes = (ts//arr)**2
            note = f"oversize ({ts}>{arr}): systolic does {passes} internal passes/instr"
        gen("3b_tile_size", f"tile_{ts}x{ts}",
            f"--hw + tile={ts}x{ts}  [{note}]",
            wl_ov=dict(tile_rows=ts, tile_cols=ts))

    # KV cache location: HBM (default) vs disabled (pure prefill throughput).
    for kv_en, kv_loc in [(True,"hbm"),(False,"hbm")]:
        tag = "on_hbm" if kv_en else "off"
        gen("3c_kv_cache", f"kv_{tag}",
            f"--hw + kv_cache={'on (HBM decode step modeled)' if kv_en else 'off (prefill only)'}",
            wl_ov=dict(kv_cache=kv_en))

    # ── 4. GQA sweep ─────────────────────────────────────────────────────────
    # 8B: 32 Q-heads. Vary num_kv_heads: more KV heads = more HBM bytes per token.
    # MHA (kv=32): 4× more KV bandwidth vs default GQA-4.
    # MQA (kv=1): 1/8 the KV bandwidth → faster decode, worse quality.
    wl8b_base_q = wl8b["num_q_heads"]
    for kv, tag in [(wl8b_base_q, "MHA"), (wl8b_base_q//4, "GQA-4_default"),
                    (wl8b_base_q//8, "GQA-8"), (max(1, wl8b_base_q//16), "GQA-16"),
                    (1, "MQA")]:
        if kv <= 0 or kv > wl8b_base_q: continue
        grp = wl8b_base_q // kv
        gen("4a_gqa_8b", f"8b_kv{kv}_{tag}",
            f"8B  num_kv_heads={kv}  gqa_group={grp}  "
            f"KV-HBM={'max' if kv==wl8b_base_q else str(round(kv/wl8b_base_q*100))+'%_of_MHA'}",
            wl_ov=dict(num_kv_heads=kv,
                       gqa_group=max(1, wl8b_base_q//kv) if kv > 0 else 1))

    # 70B: 64 Q-heads, default kv=8.
    wl70b_base_q = wl70b["num_q_heads"]
    for kv, tag in [(wl70b_base_q,"MHA"),(wl70b_base_q//8,"GQA-8_default"),
                    (wl70b_base_q//16,"GQA-16"),(1,"MQA")]:
        if kv <= 0: continue
        grp = max(1, wl70b_base_q//kv)
        gen("4b_gqa_70b", f"70b_kv{kv}_{tag}",
            f"70B  num_kv_heads={kv}  gqa_group={grp}",
            wl_base=wl70b,
            wl_ov=dict(num_kv_heads=kv, gqa_group=grp))

    # ── 5. Pareto frontier: array-size × HBM-bandwidth ───────────────────────
    # 2D grid sweep. Each point records (tflops_peak, hbm_bw, tps, ttft).
    # Pareto frontier = configs not dominated on both axes simultaneously.
    # Use 8B, prompt=2048 for all cells so results are comparable.
    for sz in [128, 256, 512]:
        for bw in [1.0, 2.0, 3.35]:
            gen("5_pareto_array_x_bw", f"arr{sz}_bw{bw}",
                f"Pareto: array={sz}x{sz}  hbm_bw={bw}TB/s",
                arch_ov=dict(array_rows=sz, array_cols=sz, hbm_bw_tb_s=bw),
                wl_ov=dict(tile_rows=sz, tile_cols=sz))

    # ── 6. Calibration against published results ──────────────────────────────
    # Approximate real chips with single-array configs.
    # Compare simulated TPS to published bs=1 decode TPS.
    # Target: roofline efficiency within ±20% of real hardware.
    # NOTE: our simulator models batch=1 (single stream).
    for chip, params in CALIB_HW.items():
        pub_tps = PUBLISHED_TPS.get(chip, None)
        gen("6_calibration", f"calib_{chip}",
            f"Approx {chip}: {params['hbm_bw_tb_s']}TB/s  "
            f"{params['clock_ghz']}GHz  "
            f"pub_bs1_tps={pub_tps}",
            arch_ov=params)

    return runs


# ── Runner + parser ───────────────────────────────────────────────────────────
def run_sim(binary, ae, we):
    ap = wp = None; at = wt = False
    try:
        if "_file" in ae:
            ap = ae["_file"]
        else:
            f  = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
            f.write(arch_yaml(**ae)); f.close(); ap = f.name; at = True
        if "_file" in we:
            wp = we["_file"]
        else:
            f  = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
            f.write(workload_yaml(**we)); f.close(); wp = f.name; wt = True
        t0  = time.time()
        tb  = "/usr/bin/time"
        cmd = ([tb,"-v"] if os.path.isfile(tb) and os.access(tb,os.X_OK) else []) + \
              [binary, "--config", ap, "--llama-workload", wp, "--no-trace"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT_S)
        r   = parse_out(res.stdout + res.stderr)
        r["wall_s"] = round(time.time()-t0, 1); r["ok"] = True
        return r
    except subprocess.TimeoutExpired: return {"ok":False,"error":"TIMEOUT"}
    except Exception as e:            return {"ok":False,"error":str(e)}
    finally:
        for tmp,path in [(at,ap),(wt,wp)]:
            if tmp and path:
                try: os.unlink(path)
                except: pass

def parse_out(out):
    r = {}
    def g(pat, cast=float):
        m = re.search(pat,out); return cast(m.group(1)) if m else None
    r["cycles"]        = g(r"cycle=(\d+)", int)
    r["MACs"]          = g(r"MACs=(\d+)", int)
    r["hbm_bytes"]     = g(r"HBM_bytes=(\d+)", int)
    r["systolic_util"] = (g(r"systolic\s+busy=\d+\s+util=([\d.]+)") or
                          g(r"systolic.*?avg_util=([\d.]+)"))
    r["dma_util"]      = g(r"\bdma\s+busy=\d+\s+util=([\d.]+)")
    r["vec_avg_util"]  = g(r"avg_util=([\d.]+)")
    r["roofline_eff"]  = g(r"roofline_efficiency=([\d.]+)")
    r["roofline_bound"]= (lambda m: m.group(1) if m else None)(
                          re.search(r"bound=[\d.e+]+ \((\w+-bound)\)",out))
    r["ttft_ns"]       = g(r"TTFT=([\d.e+]+)")
    r["decode_tps"]    = g(r"throughput=([\d.]+)")
    r["peak_rss_kb"]   = g(r"Maximum resident set size .kbytes.: (\d+)", int)
    return r

def get_dp(ae, we):
    ap = read_arch(ae["_file"]) if "_file" in ae else ae
    wp = read_workload(we["_file"]) if "_file" in we else we
    return {**ap, **wp}

def derived(r, ae, we):
    d = {}
    try:
        ap  = read_arch(ae["_file"]) if "_file" in ae else ae
        wp  = read_workload(we["_file"]) if "_file" in we else we
        clk = ap["clock_ghz"]; cyc = r.get("cycles") or 1
        ws  = cyc/(clk*1e9)
        macs= r.get("MACs") or 0; hbm = r.get("hbm_bytes") or 0
        d["tflops_achieved"]      = round(macs*2/ws/1e12, 4)
        d["hbm_bw_achieved_tb_s"] = round(hbm/ws/1e12, 4)
        d["tflops_peak"]          = round(ap["array_rows"]*ap["array_cols"]*
                                          ap["systolic_units"]*2*clk*1e9/1e12, 4)
        # Arithmetic intensity = MACs / HBM_bytes (higher = more compute per byte)
        d["arith_intensity"]      = round(macs/hbm, 2) if hbm else None
        # Calibration: compare to published TPS
        chip = next((c for c in CALIB_HW if ae.get("_file","").find(c)>=0 or
                     (not "_file" in ae and ae.get("hbm_bw_tb_s")==CALIB_HW.get(c,{}).get("hbm_bw_tb_s"))),
                    None)
        if chip and chip in PUBLISHED_TPS and r.get("decode_tps"):
            pub = PUBLISHED_TPS[chip]
            sim = r["decode_tps"]
            err = (sim - pub) / pub * 100
            d["calib_published_tps"] = pub
            d["calib_error_pct"]     = round(err, 1)
    except Exception: pass
    return d


# ── Dry-run table ─────────────────────────────────────────────────────────────
def print_dry_run(sweep):
    hdr = (f"{'#':>3}  {'GROUP':<24} {'NAME':<28} {'SRC':<5}"
           f" {'CLK':>4} {'ARR':>8} {'SU':>3} {'VC':>3}"
           f" {'DMA':>4} {'BW':>6} {'LAT':>5}"
           f" {'TILE':>8} {'kv':>3} {'GQA':>3} {'PLEN':>6}")
    sep = "-"*len(hdr)
    print(hdr); print(sep)
    for i,(grp,name,_,ae,we) in enumerate(sweep):
        p   = get_dp(ae,we)
        src = "file" if "_file" in ae else "gen"
        arr = f"{p['array_rows']}x{p['array_cols']}"
        til = f"{p['tile_rows']}x{p['tile_cols']}"
        print(
            f"{i+1:>3}  {grp:<24} {name:<28} {src:<5}"
            f" {p['clock_ghz']:>4.1f} {arr:>8} {p['systolic_units']:>3} {p['vector_cores']:>3}"
            f" {p.get('dma_channels',1):>4} {p['hbm_bw_tb_s']:>6.2f} {p['hbm_lat_cycles']:>5}"
            f" {til:>8} {p['num_kv_heads']:>3} {p.get('gqa_group',4):>3} {p['prompt_len']:>6}")
    print(sep)
    grp_counts = {}
    for g,*_ in sweep: grp_counts[g] = grp_counts.get(g,0)+1
    print(f"Total: {len(sweep)} configs across {len(grp_counts)} groups:")
    for g,c in sorted(grp_counts.items()): print(f"  {g:<30} {c} configs")
    print()
    print("FILE = use YAML as-is  |  gen = vary one param around --hw config")
    print("\nRemove --dry-run to execute.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--binary",      default=DEFAULT_BINARY)
    ap.add_argument("--hw",          default=DEFAULT_HW,
                    help="Base HW config file. Compute/memory sweeps vary "
                         "one param around this.")
    ap.add_argument("--workload8b",  default=DEFAULT_WL8B,
                    help="8B model workload YAML")
    ap.add_argument("--workload70b", default=DEFAULT_WL70B,
                    help="70B model workload YAML (40/80 layers; scale ×2 for full)")
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--group",       default=None,
                    help="Only run groups starting with this prefix")
    ap.add_argument("--out",         default=DEFAULT_OUTFILE)
    args = ap.parse_args()

    # Validate inputs
    for label, path in [("--hw", args.hw),
                        ("--workload8b", args.workload8b),
                        ("--workload70b", args.workload70b)]:
        if not Path(path).exists():
            print(f"WARNING: {label} file not found: {path}", file=sys.stderr)

    # Read base configs
    hw_params  = read_arch(args.hw)     if Path(args.hw).exists()          else {}
    wl8b_params= read_workload(args.workload8b)  if Path(args.workload8b).exists()  else {}
    wl70b_params=read_workload(args.workload70b) if Path(args.workload70b).exists() else {}

    if not hw_params:
        sys.exit(f"Cannot read --hw file: {args.hw}")
    if not wl8b_params:
        sys.exit(f"Cannot read --workload8b file: {args.workload8b}")
    if not wl70b_params:
        sys.exit(f"Cannot read --workload70b file: {args.workload70b}")

    sweep = make_sweep(hw_params, wl8b_params, wl70b_params,
                       args.hw, args.workload8b, args.workload70b)
    if args.group:
        sweep = [(g,n,d,a,w) for g,n,d,a,w in sweep if g.startswith(args.group)]
        if not sweep: sys.exit(f"No configs match --group '{args.group}'")

    if args.dry_run:
        print(f"--hw        : {args.hw}")
        print(f"--workload8b: {args.workload8b}")
        print(f"--workload70b:{args.workload70b}")
        print()
        print_dry_run(sweep)
        return

    if not Path(args.binary).exists():
        sys.exit(f"Binary not found: {args.binary}")

    print(f"Sweep: {len(sweep)} configs")
    print(f"HW base: {args.hw}   8B: {args.workload8b}   70B: {args.workload70b}")
    print(f"Output:  {args.out}\n")

    rows = []; t0 = time.time()
    for i,(grp,name,desc,ae,we) in enumerate(sweep):
        ela = int(time.time()-t0)
        eta = int((len(sweep)-i)*ela/max(i,1)) if i else "?"
        src = "file" if "_file" in ae else "gen"
        print(f"[{i+1:2}/{len(sweep)}] {grp}/{name} ({src})"
              f"  t={ela}s eta~{eta}s ... ", end="", flush=True)

        r  = run_sim(args.binary, ae, we)
        dv = derived(r, ae, we) if r.get("ok") else {}
        p  = get_dp(ae, we)

        # Calibration lookup
        pub_tps = None
        for chip, params in CALIB_HW.items():
            if name == f"calib_{chip}":
                pub_tps = PUBLISHED_TPS.get(chip)

        calib_err = None
        if pub_tps and r.get("ok") and r.get("decode_tps"):
            calib_err = round((r["decode_tps"] - pub_tps)/pub_tps*100, 1)

        if r.get("ok"):
            cal_str = f"  [{calib_err:+.0f}% vs pub {pub_tps}]" if calib_err is not None else ""
            print(f"sys={r.get('systolic_util') or 0:.1f}%"
                  f" dma={r.get('dma_util') or 0:.1f}%"
                  f" rl={r.get('roofline_eff') or 0:.2f}%"
                  f" tps={r.get('decode_tps') or 0:.1f}"
                  f" {r.get('roofline_bound',''):>14}"
                  f"  {r.get('wall_s',0):.0f}s{cal_str}")
        else:
            print(f"FAILED: {r.get('error','?')}")

        rows.append({
            "group":grp, "name":name, "description":desc,
            "config_source":ae.get("_file","generated"),
            # arch
            "clock_ghz":p.get("clock_ghz"), "array_rows":p.get("array_rows"),
            "array_cols":p.get("array_cols"), "bidirectional":p.get("bidirectional"),
            "systolic_units":p.get("systolic_units"), "vector_cores":p.get("vector_cores"),
            "dma_channels":p.get("dma_channels",1),
            "hbm_bw_tb_s":p.get("hbm_bw_tb_s"), "hbm_lat_cycles":p.get("hbm_lat_cycles"),
            "stage_double_buffer":p.get("stage_double_buffer",False),
            # workload
            "tile_rows":p.get("tile_rows"), "tile_cols":p.get("tile_cols"),
            "num_layers":p.get("num_layers"), "hidden_dim":p.get("hidden_dim"),
            "num_q_heads":p.get("num_q_heads"), "num_kv_heads":p.get("num_kv_heads"),
            "gqa_group":p.get("gqa_group"), "prompt_len":p.get("prompt_len"),
            "kv_cache":p.get("kv_cache"),
            # sim results
            "cycles":r.get("cycles"), "MACs":r.get("MACs"),
            "hbm_bytes":r.get("hbm_bytes"),
            "systolic_util_pct":r.get("systolic_util"),
            "dma_util_pct":r.get("dma_util"),
            "vec_avg_util_pct":r.get("vec_avg_util"),
            "roofline_eff_pct":r.get("roofline_eff"),
            "roofline_bound":r.get("roofline_bound"),
            "ttft_ns":r.get("ttft_ns"), "decode_tps":r.get("decode_tps"),
            "peak_rss_mb":((r.get("peak_rss_kb") or 0)//1024) or None,
            "wall_s":r.get("wall_s"),
            # derived
            "tflops_achieved":      dv.get("tflops_achieved"),
            "hbm_bw_achieved_tb_s": dv.get("hbm_bw_achieved_tb_s"),
            "tflops_peak":          dv.get("tflops_peak"),
            "arith_intensity":      dv.get("arith_intensity"),
            # calibration
            "pub_tps":    pub_tps,
            "calib_err_pct": calib_err,
            "status": "OK" if r.get("ok") else r.get("error","ERR"),
        })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(args.out,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
        # Published benchmarks section
        f.write("\n\n# ===== PUBLISHED BENCHMARKS (batch=1 decode, LLaMA-3-8B) =====\n")
        f.write("# chip,pub_bs1_tps_tok_s,notes\n")
        refs = [
            ("H100_SXM5", 120,  "vLLM single-stream BF16; 3.35TB/s HBM3"),
            ("A100_SXM4",  80,  "vLLM single-stream BF16; 2.0TB/s HBM2e"),
            ("TPUv4",      70,  "Google Cloud TPU v4; per-chip estimate BF16"),
            ("AppleM3Max", 55,  "llama.cpp Metal; 300GB/s unified memory"),
            ("RTX4090",    70,  "llama.cpp CUDA Q8_0; 1TB/s GDDR6X"),
        ]
        for chip,tps,note in refs:
            f.write(f"{chip},{tps},{note}\n")
        f.write("\n# Calibration notes\n")
        f.write("# - Roofline efficiency bs=1 decode: 3-8% (memory-bound) matches real HW\n")
        f.write("# - Roofline efficiency large-batch prefill: 40-55% (compute-bound)\n")
        f.write("# - Our batch=1 TPS << published batch=64 TPS (linear scaling with batch)\n")
        f.write("# - Project target: roofline_eff within ±20% of published values above\n")

    total = int(time.time()-t0)
    ok    = sum(1 for r in rows if r["status"]=="OK")
    cal   = [(r["name"],r["calib_err_pct"]) for r in rows
             if r.get("calib_err_pct") is not None]

    print(f"\nDone {total//60}m{total%60}s  —  {ok}/{len(rows)} OK  →  {args.out}")

    if cal:
        print("\n── Calibration summary ──────────────────────────────────────────")
        print(f"  {'Config':<30} {'sim_tps':>8} {'pub_tps':>8} {'error':>8} {'pass?':>6}")
        print("  " + "-"*60)
        for r in rows:
            if r.get("calib_err_pct") is not None:
                ok_str = "✓" if abs(r["calib_err_pct"]) <= 20 else "✗ >±20%"
                print(f"  {r['name']:<30} {r.get('decode_tps') or 0:>8.1f}"
                      f" {r['pub_tps']:>8} {r['calib_err_pct']:>+7.1f}%  {ok_str}")

    print("\n── Summary table ────────────────────────────────────────────────────")
    print(f"{'GROUP':<24} {'NAME':<26} {'SYS%':>5} {'DMA%':>5}"
          f" {'RL%':>6} {'AI':>6} {'TPS':>8} {'BOUND':<14}")
    print("-"*100)
    for r in rows:
        if r["status"]=="OK":
            print(
                f"{r['group']:<24} {r['name']:<26}"
                f" {r['systolic_util_pct'] or 0:5.1f}"
                f" {r['dma_util_pct']      or 0:5.1f}"
                f" {r['roofline_eff_pct']  or 0:6.2f}"
                f" {r['arith_intensity']   or 0:6.1f}"
                f" {r['decode_tps']        or 0:8.1f}"
                f" {r['roofline_bound']    or '':14}")

if __name__ == "__main__":
    main()