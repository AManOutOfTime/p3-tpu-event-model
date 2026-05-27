#!/usr/bin/env python3
"""
Generate a pipelined multi-tile FA2 schedule for a single attention head.

Models the overlapping execution shown in Norrie et al. 2021 (TPU paper):
  - DMA prefetches K[j+1] while systolic runs GEMM S[j]
  - Access core transposes K[j+1] while DMA loads V[j+1]
  - 3 vector cores overlap softmax stats with systolic compute
  - 2 access cores overlap transpose with rowmax/rowsum

Hardware units assumed:
  - 1 DMA (serializes via reserve_unit_pool available_at)
  - 1 Systolic array (gemm/weight_load: dep-encoded serialization)
  - 3 Vector cores (pool — any op issues to best available)
  - 2 Access cores (pool — transpose, rowmax, init_fill)

LLaMA-3-8B single head default parameters:
  seq_len = 4096, d_head = 128, Br = Bc = 128 (= systolic rows/cols)
  N_q = seq_len / Br = 32 Q tiles
  N_kv = seq_len / Bc = 32 KV tiles per Q tile

Usage:
  python scripts/gen_fa2_full.py                      # full 32x32
  python scripts/gen_fa2_full.py --n-q 2 --n-kv 4    # quick test
  python scripts/gen_fa2_full.py --out schedules/my.yaml
"""

import argparse
import yaml
from pathlib import Path


def gen_schedule(N_q: int, N_kv: int) -> list:
    """
    Build the flat instruction list for N_q outer Q tiles x N_kv inner KV tiles.

    Pipelining structure per Q tile i:
      PRE: load Q[i], init O_acc/m/l, stage Q[i] -> Q_operand, load K[i,0]
      INNER j=0..N_kv-1:
        load V[j]           (DMA, after K[j])
        transpose K[j]      (access_core, after K[j]) -- overlaps with V load
        weight_load K_T[j]  (systolic, after transpose + GEMM_Temp[j-1])
        GEMM S[j]           (systolic, after weight_load K_T + Q_staged)
        scale S[j]          (vector_core, after GEMM S)
        rowmax S[j]         (vector_core, after scale S)
        update_rowmax m[i]  (vector_core, after rowmax + prev rowmax update)
        exp_shift P[j]      (vector_core, after update_rowmax)
        update_rowsum l[i]  (vector_core, after exp_shift + prev rowsum)
        scale O_acc[i]      (vector_core, after update_rowmax + prev accumulate)
        dma_stage P[j]      (DMA, after exp_shift)
        weight_load V[j]    (systolic, after GEMM S + V loaded)
        GEMM Temp[j]        (systolic, after stage P + weight_load V)
        accumulate O_acc    (vector_core, after scale O_acc + GEMM Temp)
        [PREFETCH] load K[j+1] (DMA, after V[j] loaded -- overlaps GEMM S[j])
      POST: normalize, logsumexp, store O, store L
    """
    instrs = []
    uid = [0]

    def new_id():
        v = uid[0]
        uid[0] += 1
        return v

    def add(op, unit, deps, label, params):
        """Append one instruction and return its id."""
        id_ = new_id()
        d = {"id": id_, "op": op, "unit": unit, "label": label, "params": params}
        if deps:
            d["depends_on"] = deps
        instrs.append(d)
        return id_

    # ── Symbolic dimensions (resolved by sim_main at runtime) ──────────────
    Br = "Br"       # systolic rows (128)
    Bc = "Bc"       # systolic cols (128)
    d_k = "d_k"     # d_head (128)
    d_head = "d_head"

    prev_q_final_dma = None   # last DMA op of previous Q tile (for DMA serial)

    for i in range(N_q):
        qi = f"i{i}"            # Q-tile suffix

        # ── PRE-PHASE ───────────────────────────────────────────────────────
        q_load_deps = [prev_q_final_dma] if prev_q_final_dma is not None else []

        id_Q_load = add("dma_load", "dma", q_load_deps,
            f"[Q{i}][PRE] Load Q_{qi} from HBM",
            {"source": "HBM.Q_full",
             "destination": f"shared_ibuf.Q_{qi}",
             "rows": Br, "cols": d_k})

        # init fills: access_core pool handles 3 fills with 2 units
        id_init_O = add("init_fill", "access_core", [],
            f"[Q{i}][PRE] Init O_acc_{qi} = 0",
            {"destination": f"shared_obuf.O_acc_{qi}",
             "rows": Br, "cols": d_head, "init_value": 0})

        id_init_m = add("init_fill", "access_core", [],
            f"[Q{i}][PRE] Init m_{qi} = -inf",
            {"destination": f"shared_obuf.m_{qi}",
             "length": Br, "init_value": "-inf"})

        id_init_l = add("init_fill", "access_core", [],
            f"[Q{i}][PRE] Init l_{qi} = 0",
            {"destination": f"shared_obuf.l_{qi}",
             "length": Br, "init_value": 0})

        # Stage Q[i] -> Q_operand: on-chip DMA; must wait for Q loaded
        id_Q_stage = add("dma_stage", "dma", [id_Q_load],
            f"[Q{i}][PRE] Stage Q_{qi} -> systolic_array.Q_operand",
            {"source": f"shared_ibuf.Q_{qi}",
             "destination": "systolic_array.Q_operand",
             "rows": Br, "cols": d_k})

        # Prefetch K[0]: DMA serial after Q stage
        id_K_prev = add("dma_load", "dma", [id_Q_stage],
            f"[Q{i}][j=0][PRE] Load K_{qi}_j0 from HBM",
            {"source": "HBM.K_full",
             "destination": f"shared_ibuf.K_{qi}_j0",
             "rows": Bc, "cols": d_k})

        # Loop-carried state
        prev_upd_rowmax = id_init_m
        prev_upd_rowsum = id_init_l
        prev_accumulate = id_init_O
        prev_gemm_temp  = None     # systolic structural hazard tracker

        # ── INNER LOOP ──────────────────────────────────────────────────────
        for j in range(N_kv):
            jid = f"i{i}_j{j}"
            id_K = id_K_prev    # K[j] already scheduled

            # Load V[j]: DMA serial after K[j]
            id_V = add("dma_load", "dma", [id_K],
                f"[Q{i}][j={j}] Load V_{jid} from HBM",
                {"source": "HBM.V_full",
                 "destination": f"shared_ibuf.V_{jid}",
                 "rows": Bc, "cols": d_head})

            # Transpose K[j]: access_core, after K loaded (overlaps V load)
            id_tr = add("transpose", "access_core", [id_K],
                f"[Q{i}][j={j}] Transpose K_{jid} -> KT_{jid}",
                {"source": f"shared_ibuf.K_{qi}_j{j}",
                 "destination": f"shared_ibuf.KT_{jid}",
                 "input_rows": Bc, "input_cols": d_k,
                 "output_rows": d_k, "output_cols": Bc})

            # Weight load K_T[j]: systolic, after transpose.
            # Structural hazard: also waits for GEMM_Temp[j-1] to free PE weight regs.
            wl_K_deps = [id_tr]
            if prev_gemm_temp is not None:
                wl_K_deps.append(prev_gemm_temp)
            id_wl_K = add("weight_load", "systolic", wl_K_deps,
                f"[Q{i}][j={j}] weight_load KT_{jid} -> PE regs",
                {"source": f"shared_ibuf.KT_{jid}",
                 "destination": "systolic_array.weight_reg"})

            # GEMM S[j] = Q_operand x K_T[j]
            id_gemm_S = add("gemm", "systolic", [id_Q_stage, id_wl_K],
                f"[Q{i}][j={j}] GEMM S_{jid} = Q_operand x KT_{jid}  [{Br}x{d_k}x{Bc}]",
                {"source_a": "systolic_array.Q_operand",
                 "source_b": f"shared_ibuf.KT_{jid}",
                 "destination": f"shared_obuf.S_{jid}",
                 "M": Br, "K": d_k, "N": Bc})

            # Scale S[j] /= sqrt(d_k) — in-place
            id_scS = add("scale", "vector_core", [id_gemm_S],
                f"[Q{i}][j={j}] Scale S_{jid} /= sqrt(d_k)",
                {"source": f"shared_obuf.S_{jid}",
                 "destination": f"shared_obuf.S_{jid}",
                 "rows": Br, "cols": Bc,
                 "scalar": "1/sqrt(d_k)"})

            # rowmax S[j] -> rowmax_tmp
            id_rowmax = add("rowmax", "vector_core", [id_scS],
                f"[Q{i}][j={j}] rowmax(S_{jid}) -> rowmax_{jid}",
                {"source": f"shared_obuf.S_{jid}",
                 "destination": f"vector_scratch.rowmax_{jid}",
                 "rows": Br, "cols": Bc})

            # update_rowmax: m[i]=max(m,r), correction=exp(m_old-m_new)
            # Serial: waits for previous m update (prev_upd_rowmax)
            id_upd_rm = add("update_rowmax", "vector_core",
                [id_rowmax, prev_upd_rowmax],
                f"[Q{i}][j={j}] update_rowmax m_{qi}; -> corr_{jid}",
                {"source_m_old": f"shared_obuf.m_{qi}",
                 "source_rowmax": f"vector_scratch.rowmax_{jid}",
                 "destination_m": f"shared_obuf.m_{qi}",
                 "destination_correction": f"shared_obuf.corr_{jid}",
                 "length": Br})

            # exp_shift: P[j] = exp(S[j] - m[i])
            id_exp = add("exp_shift", "vector_core", [id_upd_rm],
                f"[Q{i}][j={j}] exp_shift P_{jid} = exp(S_{jid} - m_{qi})",
                {"source_matrix": f"shared_obuf.S_{jid}",
                 "source_shift": f"shared_obuf.m_{qi}",
                 "destination": f"shared_ibuf.P_{jid}",
                 "rows": Br, "cols": Bc})

            # update_rowsum: l[i] = correction * l_old + rowsum(P[j])
            # Serial: waits for previous l update (prev_upd_rowsum)
            id_upd_rs = add("update_rowsum", "vector_core",
                [id_exp, id_upd_rm, prev_upd_rowsum],
                f"[Q{i}][j={j}] update_rowsum l_{qi}",
                {"source_p": f"shared_ibuf.P_{jid}",
                 "source_correction": f"shared_obuf.corr_{jid}",
                 "source_l_old": f"shared_obuf.l_{qi}",
                 "destination": f"shared_obuf.l_{qi}",
                 "rows": Br, "cols": Bc})

            # scale O_acc *= correction — in-place row-broadcast
            # Waits for correction (upd_rm) and previous O_acc modification
            id_scO = add("scale", "vector_core",
                [id_upd_rm, prev_accumulate],
                f"[Q{i}][j={j}] Scale O_acc_{qi} *= corr_{jid}",
                {"source": f"shared_obuf.O_acc_{qi}",
                 "source_scale": f"shared_obuf.corr_{jid}",
                 "destination": f"shared_obuf.O_acc_{qi}",
                 "rows": Br, "cols": d_head})

            # Stage P[j] -> P_operand (DMA serial; auto-queued after prefetch loads)
            id_stage_P = add("dma_stage", "dma", [id_exp],
                f"[Q{i}][j={j}] Stage P_{jid} -> systolic_array.P_operand",
                {"source": f"shared_ibuf.P_{jid}",
                 "destination": "systolic_array.P_operand",
                 "rows": Br, "cols": Bc})

            # weight_load V[j]: structural dep on GEMM S[j] freeing weight regs
            id_wl_V = add("weight_load", "systolic", [id_gemm_S, id_V],
                f"[Q{i}][j={j}] weight_load V_{jid} -> PE regs",
                {"source": f"shared_ibuf.V_{jid}",
                 "destination": "systolic_array.weight_reg"})

            # GEMM Temp[j] = P_operand x V[j]
            id_gemm_T = add("gemm", "systolic", [id_stage_P, id_wl_V],
                f"[Q{i}][j={j}] GEMM Temp_{jid} = P_operand x V_{jid}  [{Br}x{Bc}x{d_head}]",
                {"source_a": "systolic_array.P_operand",
                 "source_b": f"shared_ibuf.V_{jid}",
                 "destination": f"shared_obuf.Temp_{jid}",
                 "M": Br, "K": Bc, "N": d_head})

            # Accumulate: O_acc[i] += Temp[j]
            id_acc = add("accumulate", "vector_core", [id_scO, id_gemm_T],
                f"[Q{i}][j={j}] O_acc_{qi} += Temp_{jid}",
                {"source_a": f"shared_obuf.O_acc_{qi}",
                 "source_b": f"shared_obuf.Temp_{jid}",
                 "destination": f"shared_obuf.O_acc_{qi}",
                 "rows": Br, "cols": d_head})

            # PREFETCH K[j+1]: DMA serial after V[j] load.
            # Starts while GEMM S[j] (and later GEMM Temp[j]) are running.
            if j + 1 < N_kv:
                id_K_next = add("dma_load", "dma", [id_V],
                    f"[Q{i}][j={j}][PREFETCH] Load K_{qi}_j{j+1} from HBM",
                    {"source": "HBM.K_full",
                     "destination": f"shared_ibuf.K_{qi}_j{j+1}",
                     "rows": Bc, "cols": d_k})
                id_K_prev = id_K_next

            # Update loop-carried state
            prev_upd_rowmax = id_upd_rm
            prev_upd_rowsum = id_upd_rs
            prev_accumulate = id_acc
            prev_gemm_temp  = id_gemm_T

        # ── POST-PHASE ──────────────────────────────────────────────────────
        id_norm = add("normalize", "vector_core",
            [prev_accumulate, prev_upd_rowsum],
            f"[Q{i}][POST] O_tile_{qi} = O_acc_{qi} / l_{qi}",
            {"source_matrix": f"shared_obuf.O_acc_{qi}",
             "source_denom": f"shared_obuf.l_{qi}",
             "destination": f"shared_obuf.O_tile_{qi}",
             "rows": Br, "cols": d_head})

        id_lse = add("logsumexp", "vector_core",
            [prev_upd_rowmax, prev_upd_rowsum, id_norm],
            f"[Q{i}][POST] L_tile_{qi} = m_{qi} + log(l_{qi})",
            {"source_m": f"shared_obuf.m_{qi}",
             "source_l": f"shared_obuf.l_{qi}",
             "destination": f"shared_obuf.L_tile_{qi}",
             "length": Br})

        id_stO = add("dma_store", "dma", [id_norm],
            f"[Q{i}][POST] Store O_tile_{qi} -> HBM",
            {"source": f"shared_obuf.O_tile_{qi}",
             "destination": f"HBM.O_{qi}",
             "rows": Br, "cols": d_head})

        id_stL = add("dma_store", "dma", [id_lse, id_stO],
            f"[Q{i}][POST] Store L_tile_{qi} -> HBM",
            {"source": f"shared_obuf.L_tile_{qi}",
             "destination": f"HBM.L_{qi}",
             "length": Br})

        prev_q_final_dma = id_stL

    return instrs


def count_stats(instrs: list) -> dict:
    from collections import Counter
    c = Counter(i["op"] for i in instrs)
    return dict(c)


def main():
    ap = argparse.ArgumentParser(description="Generate full FA2 multi-tile schedule")
    ap.add_argument("--n-q",  type=int, default=32,
                    help="Number of Q tiles (default 32 for LLaMA-3-8B, seq_len=4096)")
    ap.add_argument("--n-kv", type=int, default=32,
                    help="Number of KV tiles per Q tile (default 32)")
    ap.add_argument("--out",  type=str, default="",
                    help="Output path (default: schedules/fa2_full_{N_q}q_{N_kv}kv.yaml)")
    args = ap.parse_args()

    N_q  = args.n_q
    N_kv = args.n_kv

    print(f"Generating FA2 full schedule: N_q={N_q}, N_kv={N_kv}")
    instrs = gen_schedule(N_q, N_kv)

    stats = count_stats(instrs)
    total = len(instrs)
    print(f"  Total instructions : {total}")
    print(f"  Op breakdown       :")
    for op, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {op:20s}: {cnt}")

    out_path = args.out or f"schedules/fa2_full_{N_q}q_{N_kv}kv.yaml"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    header = f"""\
# FlashAttention-2 full single-head schedule  ({N_q} Q-tiles x {N_kv} KV-tiles)
#
# Generated by scripts/gen_fa2_full.py
#
# LLaMA-3-8B target: seq_len=4096, d_head=128, Br=Bc=128 (systolic array size)
#   N_q  = seq_len/Br = {N_q}  outer Q tile iterations
#   N_kv = seq_len/Bc = {N_kv} inner KV tile iterations per Q tile
#
# Overlapping pipeline (Norrie et al. 2021 Fig.3 inspired):
#   DMA: K[j+1] prefetch starts after V[j] load (overlaps GEMM S[j])
#   Access core: transpose K[j+1] overlaps V[j+1] DMA load
#   3 vector cores: softmax stats (scale/exp/rowsum) overlap next-tile GEMM S
#   Systolic structural hazard: weight_load K_T[j+1] depends on GEMM_Temp[j]
#
# Total instructions: {total}
# Op breakdown: {stats}
#
# Run:
#   .\\build\\apps\\Release\\sim_main.exe --schedule {out_path} --no-trace
#
"""

    doc = {"schedule": instrs}
    yaml_str = yaml.dump(doc, default_flow_style=False, sort_keys=False, width=120)

    with open(out_path, "w") as f:
        f.write(header)
        f.write(yaml_str)

    print(f"  Written to         : {out_path}")
    print(f"\nRun:")
    print(f"  .\\build\\apps\\Release\\sim_main.exe --schedule {out_path} --no-trace")


if __name__ == "__main__":
    main()
