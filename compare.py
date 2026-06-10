#!/usr/bin/env python3
"""
compare.py — HW × Workload Cross-Product KPI Comparison
=========================================================
Runs every combination of HW configs × workload configs and emits a
structured CSV of KPIs, plus a console summary table.

Hardcoded matrix:
  HW        : datacenter.yaml, edge_dev.yaml
  Workloads : workloads/llama_prefill_decode_1B.yaml
              workloads/llama_prefill_decode_8B.yaml
              workloads/llama_prefill_decode_70B.yaml
  Modes     : prefill_decode  (TTFT primary KPI)
              decode           (decode_tps primary KPI)

  → 2 HW × 3 workloads × 2 modes = 12 runs total.

Memory safety
-------------
  If the simulator exits non-zero AND its output contains any of the
  OOM_SIGNALS strings, the run is marked MEM_ERR and skipped rather than
  treated as a generic failure. The CSV row is written with status=MEM_ERR
  and all metric columns left empty so downstream tooling can filter it out.

Usage:
  python3 scripts/compare.py [--binary PATH] [--dry-run] [--out FILE]
                             [--modes prefill_decode,decode]
                             [--hw datacenter.yaml,edge_dev.yaml]
                             [--workloads wl1.yaml,wl2.yaml,wl3.yaml]

Output columns mirror sweep.py CSV_FIELDS so the two CSVs can be
concatenated and analysed together.
"""

import argparse, csv, os, re, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BINARY  = "./build/apps/sim_main"
DEFAULT_OUTFILE = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
SIM_TIMEOUT_S   = 600   # 10 min hard cap per run

DEFAULT_HW_CONFIGS = [
    "datacenter.yaml",
    "edge_dev.yaml",
]

DEFAULT_WORKLOADS = [
    ("1b", "workloads/llama_prefill_decode_1B.yaml"),
    ("8b", "workloads/llama_prefill_decode_8B.yaml"),
    ("70b","workloads/llama_prefill_decode_70B.yaml"),
]

DEFAULT_MODES = ["prefill_decode", "decode"]

# Strings in stdout/stderr that indicate an OOM / memory allocation failure.
# Extend this list if your simulator uses different messages.
OOM_SIGNALS = [
    "out of memory",
    "bad_alloc",
    "std::bad_alloc",
    "oom",
    "cannot allocate",
    "memory allocation failed",
    "sram overflow",
    "ibuf overflow",
    "obuf overflow",
    "buffer overflow",
    "insufficient memory",
    "kv cache too large",
    "kv_cache too large",
]

# ── CSV schema (mirrors sweep.py CSV_FIELDS) ──────────────────────────────────
CSV_FIELDS = [
    # identification
    "hw_config", "model", "mode", "description",
    # arch params
    "clock_ghz",
    "array_rows", "array_cols", "bidirectional",
    "systolic_units", "vector_cores", "access_cores",
    "vec_simd", "exp_lat", "access_bw",
    "dma_channels",
    "hbm_bw_tb_s", "hbm_lat_cycles",
    "ibuf_kb", "obuf_kb",
    "stage_double_buffer", "model_sram",
    # workload params
    "tile_rows", "tile_cols",
    "num_layers", "hidden_dim", "head_dim", "intermediate_dim",
    "num_q_heads", "num_kv_heads", "gqa_group",
    "prompt_len", "gen_steps", "max_seq_len", "kv_cache",
    # raw sim outputs
    "cycles", "MACs", "hbm_bytes",
    "systolic_util_pct",
    "systolic_0_util_pct", "systolic_1_util_pct",
    "dma_util_pct", "vec_util_pct", "access_util_pct",
    "roofline_eff_pct", "roofline_bound",
    "compute_bound_cyc", "memory_bound_cyc",
    "ttft_ns", "decode_tps",
    # derived / KPI metrics
    "tflops_achieved", "hbm_bw_achieved_tb_s", "tflops_peak",
    "hbm_util_pct",
    "arith_intensity",
    "ttft_per_token_ns",
    "bytes_per_token",
    "systolic_imbalance",
    "mem_compute_ratio",
    "ffn_mac_pct", "attn_mac_pct", "ffn_attn_crossover_tok",
    # run metadata
    "wall_s", "peak_rss_mb", "status",
]


# ── YAML readers (identical to sweep.py) ─────────────────────────────────────
def parse_yaml_value(value):
    v = value.strip().strip('"').strip("'")
    if v.lower() == "true":  return True
    if v.lower() == "false": return False
    try:    return int(v)
    except ValueError: pass
    try:    return float(v)
    except ValueError: pass
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
    d  = read_simple_yaml(path)
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
    d  = read_simple_yaml(path)
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


# ── YAML generators (identical to sweep.py) ──────────────────────────────────
def arch_yaml(clock_ghz=1.0, array_rows=256, array_cols=256, bidirectional=False,
              systolic_units=1, vector_cores=3, access_cores=1,
              hbm_bw_tb_s=2.0, hbm_lat_cycles=200, dma_channels=1,
              vec_simd=64, exp_lat=4, access_bw=64,
              ibuf_kb=4096, obuf_kb=4096, banking_factor=8,
              stage_double_buffer=False, model_sram=False, **_):
    b = lambda v: "true" if v else "false"
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


# ── Regime helper ─────────────────────────────────────────────────────────────
def regime(wl):
    S  = wl.get("prompt_len",        2048)
    H  = wl.get("hidden_dim",        4096)
    I  = wl.get("intermediate_dim", 14336)
    nq = wl.get("num_q_heads",         32)
    nk = wl.get("num_kv_heads",         8)
    dh = wl.get("head_dim",           128)
    ffn  = S * (3 * H * I)
    attn = S * (nq*dh*H + nk*dh*H*2 + nq*dh*H) + nq*S*S*dh*2
    tot  = ffn + attn
    ffn_pct = round(ffn / tot * 100, 1)
    return dict(
        ffn_pct   = ffn_pct,
        attn_pct  = round(100 - ffn_pct, 1),
        crossover = int(3 * I // 2),
    )


# ── Memory error detection ────────────────────────────────────────────────────
def is_oom(text: str) -> bool:
    """Return True if the simulator output signals an OOM / memory error."""
    lower = text.lower()
    return any(sig in lower for sig in OOM_SIGNALS)


# ── Simulator runner ──────────────────────────────────────────────────────────
def run_sim(binary, ae, we):
    """
    Runs the simulator. Returns dict with:
      ok=True   → normal completion, metrics populated
      ok=False  → failure; 'error' key has reason string
      oom=True  → memory allocation/OOM failure detected
    """
    ap = wp = None
    try:
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(arch_yaml(**ae)); f.close(); ap = f.name

        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(workload_yaml(**we)); f.close(); wp = f.name

        t0  = time.time()
        tb  = "/usr/bin/time"
        cmd = ([tb, "-v"] if os.path.isfile(tb) and os.access(tb, os.X_OK) else []) + \
              [binary, "--config", ap, "--llama-workload", wp, "--no-trace"]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=SIM_TIMEOUT_S)
        combined = res.stdout + res.stderr

        # Check for OOM before anything else
        if is_oom(combined) or (res.returncode != 0 and is_oom(combined)):
            return {"ok": False, "oom": True,
                    "error": "MEM_ERR",
                    "raw_output": combined[:800]}

        r = parse_out(combined)
        r["wall_s"] = round(time.time() - t0, 1)
        r["ok"]     = True
        r["oom"]    = False

        # Treat non-zero exit as failure even if no OOM signal
        if res.returncode != 0:
            return {"ok": False, "oom": False,
                    "error": f"EXIT_{res.returncode}",
                    "raw_output": combined[:400]}
        return r

    except subprocess.TimeoutExpired:
        return {"ok": False, "oom": False, "error": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "oom": False, "error": str(e)}
    finally:
        for path in [ap, wp]:
            if path:
                try: os.unlink(path)
                except: pass


# ── Output parser (identical to sweep.py) ────────────────────────────────────
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


# ── Derived / KPI metrics (identical to sweep.py) ────────────────────────────
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

        bidir_factor = 2 if ae.get("bidirectional", False) else 1
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


# ── Build CSV row ─────────────────────────────────────────────────────────────
def build_row(hw_label, model_tag, mode, ae, we, r, dv):
    status = "OK"
    if not r.get("ok"):
        status = "MEM_ERR" if r.get("oom") else r.get("error", "ERR")

    row = {
        # identification
        "hw_config":   hw_label,
        "model":       model_tag,
        "mode":        mode,
        "description": f"hw={hw_label} model={model_tag} mode={mode}",
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
        # raw sim (all None if failed)
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
        # metadata
        "wall_s":      r.get("wall_s"),
        "peak_rss_mb": ((r.get("rss_kb") or 0) // 1024) or None,
        "status":      status,
    }
    return row


# ── Console summary ───────────────────────────────────────────────────────────
def print_summary(rows):
    """Print a compact KPI table grouped by hw_config."""
    print("\n" + "═" * 130)
    print("SUMMARY  (MEM_ERR rows skipped below)")
    print("═" * 130)
    hdr = (f"{'HW':<18} {'MODEL':<5} {'MODE':<16}"
           f" {'SYS%':>5} {'DMA%':>5} {'RL%':>6}"
           f" {'AI':>5} {'MCR':>5} {'HBM_util%':>10}"
           f" {'TPS':>7} {'TTFT/tok_ns':>12} {'BOUND':<14} STATUS")
    print(hdr)
    print("-" * 130)

    for hw_lbl in dict.fromkeys(r["hw_config"] for r in rows):
        first = True
        for r in rows:
            if r["hw_config"] != hw_lbl: continue
            if r["status"] == "MEM_ERR":
                print(f"  {'':18} {r['model']:<5} {r['mode']:<16}"
                      f"  *** SKIPPED — MEM_ERR ***")
                continue
            if r["status"] != "OK":
                print(f"  {'':18} {r['model']:<5} {r['mode']:<16}"
                      f"  *** FAILED: {r['status']} ***")
                continue
            hw_col = hw_lbl if first else ""
            first = False
            print(
                f"  {hw_col:<18} {r['model']:<5} {r['mode']:<16}"
                f" {r['systolic_util_pct']    or 0:5.1f}"
                f" {r['dma_util_pct']         or 0:5.1f}"
                f" {r['roofline_eff_pct']     or 0:6.2f}"
                f" {r['arith_intensity']      or 0:5.1f}"
                f" {r['mem_compute_ratio']    or 0:5.2f}"
                f" {r['hbm_util_pct']         or 0:10.1f}"
                f" {r['decode_tps']           or 0:7.1f}"
                f" {r['ttft_per_token_ns']    or 0:12.0f}"
                f" {r['roofline_bound']       or '':14}"
                f" {r['status']}")
        print()

    ok       = sum(1 for r in rows if r["status"] == "OK")
    mem_err  = sum(1 for r in rows if r["status"] == "MEM_ERR")
    other    = len(rows) - ok - mem_err
    print(f"Totals: {ok} OK  |  {mem_err} MEM_ERR (skipped)  |  {other} other failures")
    print("═" * 130)


# ── Dry-run printer ───────────────────────────────────────────────────────────
def print_dry_run(matrix, hw_paths, wl_pairs, modes):
    print(f"DRY RUN — {len(matrix)} configs would execute")
    print(f"  HW      : {hw_paths}")
    print(f"  Workloads: {[tag for tag, _ in wl_pairs]}")
    print(f"  Modes   : {modes}\n")
    hdr = (f"{'#':>3}  {'HW':<20} {'MODEL':<5} {'MODE':<16}"
           f" {'ARR':>8} {'SU':>3} {'BW':>5} {'LAT':>4} {'SDB':>3}"
           f" {'PLEN':>6} {'GSTEP':>5} {'MSEQ':>5}")
    print(hdr)
    print("-" * len(hdr))
    b = lambda v: "Y" if v else "N"
    for i, (hw_lbl, mdl, mode, ae, we) in enumerate(matrix):
        print(
            f"{i+1:>3}  {hw_lbl:<20} {mdl:<5} {mode:<16}"
            f" {ae['array_rows']:>3}x{ae['array_cols']:<3}"
            f" {ae['systolic_units']:>3}"
            f" {ae['hbm_bw_tb_s']:>5.2f} {ae['hbm_lat_cycles']:>4}"
            f" {b(ae['stage_double_buffer']):>3}"
            f" {we['prompt_len']:>6} {we.get('gen_steps',1):>5}"
            f" {we.get('max_seq_len',8192):>5}")
    print()
    print("Remove --dry-run to execute.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--binary",    default=DEFAULT_BINARY,
                    help="Path to simulator binary")
    ap.add_argument("--hw",        default=None,
                    help="Comma-separated HW yaml paths "
                         f"(default: {','.join(DEFAULT_HW_CONFIGS)})")
    ap.add_argument("--workloads", default=None,
                    help="Comma-separated tag:path pairs, e.g. "
                         "1b:wl1b.yaml,8b:wl8b.yaml,70b:wl70b.yaml")
    ap.add_argument("--modes",     default=None,
                    help="Comma-separated modes to run "
                         f"(default: {','.join(DEFAULT_MODES)})")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--out",       default=DEFAULT_OUTFILE)
    args = ap.parse_args()

    # ── Parse HW list ─────────────────────────────────────────────────────
    hw_paths = [p.strip() for p in args.hw.split(",")] \
               if args.hw else DEFAULT_HW_CONFIGS

    # ── Parse workload list ───────────────────────────────────────────────
    if args.workloads:
        wl_pairs = []
        for item in args.workloads.split(","):
            item = item.strip()
            if ":" in item:
                tag, path = item.split(":", 1)
            else:
                tag  = Path(item).stem
                path = item
            wl_pairs.append((tag.strip(), path.strip()))
    else:
        wl_pairs = DEFAULT_WORKLOADS

    # ── Parse modes ───────────────────────────────────────────────────────
    modes = [m.strip() for m in args.modes.split(",")] \
            if args.modes else DEFAULT_MODES

    # ── Validate paths ────────────────────────────────────────────────────
    missing = []
    for p in hw_paths:
        if not Path(p).exists(): missing.append(f"HW: {p}")
    for tag, p in wl_pairs:
        if not Path(p).exists(): missing.append(f"workload({tag}): {p}")
    if missing:
        for m in missing:
            print(f"ERROR: file not found — {m}", file=sys.stderr)
        sys.exit(1)

    # ── Load configs ──────────────────────────────────────────────────────
    hw_configs = {Path(p).stem: (p, read_arch(p)) for p in hw_paths}
    wl_configs = {tag: (p, read_workload(p)) for tag, p in wl_pairs}

# ── Build cross-product run matrix ────────────────────────────────────
    # Order: hw × model × mode  (group by hw for readable output)
    matrix = []
    for hw_lbl, (_, ae) in hw_configs.items():
        for mdl, (_, we_base) in wl_configs.items():
            for mode in modes:
                we = {**we_base, "mode": mode}

                # ── Memory Safety Caps for Edge Device ───────────────────
                if "edge_dev" in hw_lbl.lower():
                    if "8b" in mdl.lower():
                        we["prompt_len"] = 512
                    elif "70b" in mdl.lower():
                        we["prompt_len"] = 32  # Safe down-scaling to prevent Pod exit code 137

                if mode == "decode":
                    we["gen_steps"] = 32
                    we["max_seq_len"] = max(
                        we["prompt_len"] + 32,
                        we["max_seq_len"],
                    )

                matrix.append((hw_lbl, mdl, mode, ae, we))
                
    if args.dry_run:
        print_dry_run(matrix, hw_paths, wl_pairs, modes)
        return

    if not Path(args.binary).exists():
        sys.exit(f"Binary not found: {args.binary}")

    # Runtime estimate (same constants as sweep.py)
    pd_8b   = sum(1 for _, m, mo, *_ in matrix if m=="8b"  and mo=="prefill_decode")
    pd_70b  = sum(1 for _, m, mo, *_ in matrix if m=="70b" and mo=="prefill_decode")
    dec_8b  = sum(1 for _, m, mo, *_ in matrix if m=="8b"  and mo=="decode")
    dec_70b = sum(1 for _, m, mo, *_ in matrix if m=="70b" and mo=="decode")
    est_min = (pd_8b*35 + pd_70b*86 + dec_8b*2 + dec_70b*4) // 60
    print(f"compare.py — {len(matrix)} runs (~{est_min} min est.)")
    print(f"  HW      : {list(hw_configs.keys())}")
    print(f"  Models  : {list(wl_configs.keys())}")
    print(f"  Modes   : {modes}")
    print(f"  Output  : {args.out}\n")

    # ── Execute runs ──────────────────────────────────────────────────────
    rows = []
    t0   = time.time()

    csv_fh = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    csv_fh.flush()

    try:
        for i, (hw_lbl, mdl, mode, ae, we) in enumerate(matrix):
            ela = int(time.time() - t0)
            eta = int((len(matrix) - i) * ela / max(i, 1)) if i else "?"
            tag = f"{hw_lbl}/{mdl}/{mode}"
            print(f"[{i+1:2}/{len(matrix)}] {tag:<44}"
                  f" t={ela}s eta~{eta}s ... ", end="", flush=True)

            r  = run_sim(args.binary, ae, we)
            dv = derived(r, ae, we) if r.get("ok") else {}
            row = build_row(hw_lbl, mdl, mode, ae, we, r, dv)

            writer.writerow(row)
            csv_fh.flush()
            rows.append(row)

            if r.get("oom"):
                print(f"⚠  MEM_ERR — skipping (OOM detected)")
            elif not r.get("ok"):
                print(f"✗  FAILED: {r.get('error', '?')}")
            else:
                bound = r.get("roofline_bound") or ""
                mcr   = dv.get("mem_compute_ratio") or 0
                ai    = dv.get("arith_intensity")   or 0
                hu    = dv.get("hbm_util_pct")      or 0
                tps   = r.get("decode_tps")         or 0
                tpt   = dv.get("ttft_per_token_ns") or 0
                ws    = r.get("wall_s", 0)
                print(f"✓  sys={r.get('systolic_util') or 0:4.1f}%"
                      f" dma={r.get('dma_util') or 0:4.1f}%"
                      f" rl={r.get('roofline_eff') or 0:5.2f}%"
                      f" AI={ai:5.1f} mcr={mcr:5.2f}"
                      f" hbm_util={hu:5.1f}%"
                      f" tps={tps:6.1f}"
                      f" ttft/tok={tpt:8.0f}ns"
                      f" {bound:<14} {ws:.0f}s")

    finally:
        csv_fh.close()

    total = int(time.time() - t0)
    ok_n  = sum(1 for r in rows if r["status"] == "OK")
    mem_n = sum(1 for r in rows if r["status"] == "MEM_ERR")
    print(f"\nFinished in {total//60}m{total%60}s  "
          f"{ok_n} OK  |  {mem_n} MEM_ERR  |  "
          f"{len(rows)-ok_n-mem_n} other failures  →  {args.out}")

    print_summary(rows)


if __name__ == "__main__":
    main()