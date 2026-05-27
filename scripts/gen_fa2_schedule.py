#!/usr/bin/env python3
"""
gen_fa2_schedule.py  —  Generate a pipelined FA2 multi-tile schedule YAML.

Produces a schedule for one Q head of Flash Attention 2 (Dao, ICLR 2024).
The outer loop iterates over Nq Q-tiles; for each Q-tile the inner loop
iterates over Nkv KV-tiles in a pipelined fashion:

  - K[j+1] / V[j+1] are DMA-prefetched while tile j is being computed.
  - Three vector cores allow softmax stats (exp_shift, update_rowsum,
    scale_O) from tile j to overlap with tile j+1's scale / rowmax phases.
  - Q tiles run serially (no inter-Q-tile overlap needed for correctness).

LLaMA-3-8B parameters (single GQA head):
  seq_len=4096, d_head=128, Nq=32, Nkv=32, Br=128, Bc=128

Usage:
  python scripts/gen_fa2_schedule.py                 # stdout
  python scripts/gen_fa2_schedule.py --Nq 4 --Nkv 4 # small demo
  python scripts/gen_fa2_schedule.py --output schedules/fa2_full_matrix.yaml
"""

import argparse
import sys
try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Custom YAML dumper: keep lists inline (flow style) for depends_on.
# ---------------------------------------------------------------------------
class _InlineDeps(yaml.Dumper):
    pass

def _represent_int_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data,
                                     flow_style=True)

_InlineDeps.add_representer(list, _represent_int_list)

def _dump(doc):
    return yaml.dump(doc, Dumper=_InlineDeps,
                     default_flow_style=False, sort_keys=False,
                     allow_unicode=True)


# ---------------------------------------------------------------------------
# Schedule generator
# ---------------------------------------------------------------------------
def gen_fa2_schedule(Nq: int, Nkv: int,
                     Br: int = 128, Bc: int = 128,
                     d_head: int = 128) -> dict:
    """Return a dict ready for yaml.dump."""

    instrs = []
    nid = [0]

    def add(op, unit, label, params, deps=None):
        i = nid[0]
        d = {'id': i, 'op': op, 'unit': unit, 'label': label, 'params': params}
        if deps:
            d['depends_on'] = sorted(set(deps))
        instrs.append(d)
        nid[0] += 1
        return i

    prev_store_L = None   # last instruction of previous Q-tile (DMA + data)

    for qi in range(Nq):
        q_row0 = qi * Br
        q_dep = [prev_store_L] if prev_store_L is not None else None

        # ── Prologue ─────────────────────────────────────────────────────
        # Init accumulators (access_core: init_fill)
        init_O = add('init_fill', 'access_core',
            f'[Q{qi}] Init O_acc = 0  [{Br}x{d_head}]',
            {'destination': 'shared_obuf.O_acc',
             'rows': Br, 'cols': d_head, 'init_value': 0},
            deps=q_dep)

        init_m = add('init_fill', 'access_core',
            f'[Q{qi}] Init m = -inf  [{Br}]',
            {'destination': 'shared_obuf.m',
             'length': Br, 'init_value': '-inf'},
            deps=q_dep)

        init_l = add('init_fill', 'access_core',
            f'[Q{qi}] Init l = 0  [{Br}]',
            {'destination': 'shared_obuf.l',
             'length': Br, 'init_value': 0},
            deps=q_dep)

        # Load Q tile from HBM
        load_Q = add('dma_load', 'dma',
            f'[Q{qi}] Load Q[{q_row0}:{q_row0+Br},0:{d_head}]',
            {'source': f'HBM.Q[{q_row0}:{q_row0+Br},0:{d_head}]',
             'destination': 'shared_ibuf.Q_tile',
             'rows': Br, 'cols': d_head},
            deps=q_dep)

        # Stage Q into systolic input registers (on-chip, fast)
        stage_Q = add('dma_stage', 'dma',
            f'[Q{qi}] Stage Q_tile → systolic_array.Q_operand',
            {'source': 'shared_ibuf.Q_tile',
             'destination': 'systolic_array.Q_operand',
             'rows': Br, 'cols': d_head},
            deps=[load_Q])

        # Running state (per-KV accumulator pointers)
        prev_update_rowmax = init_m
        prev_update_rowsum = init_l
        prev_accumulate    = init_O
        prev_dma           = stage_Q  # last DMA op (for load-chain ordering)

        # ── KV-tile loop ──────────────────────────────────────────────────
        for kv in range(Nkv):
            buf = kv % 2              # ping-pong buffer index (0 or 1)
            row0 = kv * Bc

            # ─ DMA: Load K[kv] and V[kv] ─────────────────────────────────
            # Chained off prev_dma so loads happen in order on the single
            # DMA channel.  The DMA is free while the systolic runs GEMM,
            # so K[kv+1]/V[kv+1] are naturally prefetched during compute.
            load_K = add('dma_load', 'dma',
                f'[Q{qi}/KV{kv}] Load K[{row0}:{row0+Bc},0:{d_head}] → K_buf{buf}',
                {'source': f'HBM.K[{row0}:{row0+Bc},0:{d_head}]',
                 'destination': f'shared_ibuf.K_buf{buf}',
                 'rows': Bc, 'cols': d_head},
                deps=[prev_dma])

            load_V = add('dma_load', 'dma',
                f'[Q{qi}/KV{kv}] Load V[{row0}:{row0+Bc},0:{d_head}] → V_buf{buf}',
                {'source': f'HBM.V[{row0}:{row0+Bc},0:{d_head}]',
                 'destination': f'shared_ibuf.V_buf{buf}',
                 'rows': Bc, 'cols': d_head},
                deps=[load_K])

            # ─ Access core: Transpose K → K_T ─────────────────────────────
            # Starts when K is in IBUF (load_K done).
            # Access core is independent from DMA, so can overlap with load_V.
            transpose = add('transpose', 'access_core',
                f'[Q{qi}/KV{kv}] Transpose K_buf{buf} → KT_buf{buf}  [{Bc}x{d_head}]',
                {'source': f'shared_ibuf.K_buf{buf}',
                 'destination': f'shared_ibuf.KT_buf{buf}',
                 'input_rows': Bc, 'input_cols': d_head,
                 'output_rows': d_head, 'output_cols': Bc},
                deps=[load_K])

            # ─ Systolic: weight-load K_T, then GEMM S = Q × K_T ──────────
            wl_K = add('weight_load', 'systolic',
                f'[Q{qi}/KV{kv}] Weight-load KT_buf{buf} → PE weight registers',
                {'source': f'shared_ibuf.KT_buf{buf}',
                 'destination': 'systolic_array.weight_reg'},
                deps=[transpose])

            gemm_S = add('gemm', 'systolic',
                f'[Q{qi}/KV{kv}] GEMM S = Q_operand × KT_buf{buf}  [{Br}x{Bc}]',
                {'source_a': 'systolic_array.Q_operand',
                 'source_b': f'shared_ibuf.KT_buf{buf}',
                 'destination': 'shared_obuf.S_tile',
                 'M': Br, 'K': d_head, 'N': Bc},
                deps=[stage_Q, wl_K])

            # ─ Vector: scale → rowmax → update_rowmax → exp_shift ─────────
            # These form a serial chain on shared buffers (S_tile, m).
            # With 3 vector cores, exp_shift and scale_O can run in parallel
            # once update_rowmax is done (they depend on it but use diff bufs).
            scale_S = add('scale', 'vector_core',
                f'[Q{qi}/KV{kv}] Scale S /= sqrt(d_head)  [{Br}x{Bc}]',
                {'source': 'shared_obuf.S_tile',
                 'destination': 'shared_obuf.S_tile',
                 'rows': Br, 'cols': Bc,
                 'scalar': '1/sqrt(d_k)'},
                deps=[gemm_S])

            rowmax = add('rowmax', 'vector_core',
                f'[Q{qi}/KV{kv}] rowmax(S_tile) → rowmax_tmp  [{Br}]',
                {'source': 'shared_obuf.S_tile',
                 'destination': 'vector_scratch.rowmax_tmp',
                 'rows': Br, 'cols': Bc},
                deps=[scale_S])

            # update_rowmax: m = max(m_old, rowmax); correction = exp(m_old-m)
            # Must also wait for the PREVIOUS tile's update_rowmax (same m buf).
            update_rowmax = add('update_rowmax', 'vector_core',
                f'[Q{qi}/KV{kv}] update_rowmax → m, correction  [{Br}]',
                {'source_m_old': 'shared_obuf.m',
                 'source_rowmax': 'vector_scratch.rowmax_tmp',
                 'destination_m': 'shared_obuf.m',
                 'destination_correction': 'shared_obuf.correction',
                 'length': Br},
                deps=[rowmax, prev_update_rowmax])

            # exp_shift: P = exp(S - m_new)
            exp_shift = add('exp_shift', 'vector_core',
                f'[Q{qi}/KV{kv}] exp_shift P = exp(S - m_new)  [{Br}x{Bc}]',
                {'source_matrix': 'shared_obuf.S_tile',
                 'source_shift': 'shared_obuf.m',
                 'destination': 'shared_ibuf.P_tile',
                 'rows': Br, 'cols': Bc},
                deps=[update_rowmax])

            # update_rowsum: l = correction * l_old + rowsum(P)
            # Wait for prev tile's update_rowsum (same l buf).
            update_rowsum = add('update_rowsum', 'vector_core',
                f'[Q{qi}/KV{kv}] update_rowsum → l  [{Br}]',
                {'source_p': 'shared_ibuf.P_tile',
                 'source_correction': 'shared_obuf.correction',
                 'source_l_old': 'shared_obuf.l',
                 'destination': 'shared_obuf.l',
                 'rows': Br, 'cols': Bc},
                deps=[exp_shift, update_rowmax, prev_update_rowsum])

            # scale_O: O_acc *= correction  (row-broadcast)
            # Independent of exp_shift / update_rowsum — can run in parallel
            # on a different vector_core once update_rowmax is done.
            # Also waits for previous accumulate (same O_acc buf).
            scale_O = add('scale', 'vector_core',
                f'[Q{qi}/KV{kv}] Scale O_acc *= correction  [{Br}x{d_head}]',
                {'source': 'shared_obuf.O_acc',
                 'source_scale': 'shared_obuf.correction',
                 'destination': 'shared_obuf.O_acc',
                 'rows': Br, 'cols': d_head},
                deps=[update_rowmax, prev_accumulate])

            # ─ DMA: stage P → systolic input registers ─────────────────────
            # Reads P_tile (written by exp_shift).  load_V ensures V is in
            # IBUF so weight_load_V can start, and ensures DMA ordering.
            stage_P = add('dma_stage', 'dma',
                f'[Q{qi}/KV{kv}] Stage P_tile → systolic_array.P_operand',
                {'source': 'shared_ibuf.P_tile',
                 'destination': 'systolic_array.P_operand',
                 'rows': Br, 'cols': Bc},
                deps=[exp_shift, load_V])

            # ─ Systolic: weight-load V, then GEMM Temp = P × V ───────────
            # weight_load_V starts after GEMM_S frees the systolic AND V is
            # in IBUF (load_V done).
            wl_V = add('weight_load', 'systolic',
                f'[Q{qi}/KV{kv}] Weight-load V_buf{buf} → PE weight registers',
                {'source': f'shared_ibuf.V_buf{buf}',
                 'destination': 'systolic_array.weight_reg'},
                deps=[gemm_S, load_V])

            gemm_T = add('gemm', 'systolic',
                f'[Q{qi}/KV{kv}] GEMM Temp = P_operand × V_buf{buf}  [{Br}x{d_head}]',
                {'source_a': 'systolic_array.P_operand',
                 'source_b': f'shared_ibuf.V_buf{buf}',
                 'destination': 'shared_obuf.Temp',
                 'M': Br, 'K': Bc, 'N': d_head},
                deps=[stage_P, wl_V])

            # ─ Vector: accumulate O_acc += Temp ───────────────────────────
            accumulate = add('accumulate', 'vector_core',
                f'[Q{qi}/KV{kv}] Accumulate O_acc += Temp  [{Br}x{d_head}]',
                {'source_a': 'shared_obuf.O_acc',
                 'source_b': 'shared_obuf.Temp',
                 'destination': 'shared_obuf.O_acc',
                 'rows': Br, 'cols': d_head},
                deps=[scale_O, gemm_T])

            # ─ Advance running-state pointers ──────────────────────────────
            prev_update_rowmax = update_rowmax
            prev_update_rowsum = update_rowsum
            prev_accumulate    = accumulate
            prev_dma           = load_V   # next Load K chains off this

        # ── Epilogue ──────────────────────────────────────────────────────
        normalize = add('normalize', 'vector_core',
            f'[Q{qi}] Normalize O = O_acc / l  [{Br}x{d_head}]',
            {'source_matrix': 'shared_obuf.O_acc',
             'source_denom':  'shared_obuf.l',
             'destination':   'shared_obuf.O_tile',
             'rows': Br, 'cols': d_head},
            deps=[prev_accumulate, prev_update_rowsum])

        logsumexp = add('logsumexp', 'vector_core',
            f'[Q{qi}] Logsumexp L = m + log(l)  [{Br}]',
            {'source_m':    'shared_obuf.m',
             'source_l':    'shared_obuf.l',
             'destination': 'shared_obuf.L_tile',
             'length': Br},
            deps=[prev_update_rowmax, prev_update_rowsum, normalize])

        store_O = add('dma_store', 'dma',
            f'[Q{qi}] Store O[{q_row0}:{q_row0+Br},0:{d_head}] → HBM',
            {'source':      'shared_obuf.O_tile',
             'destination': f'HBM.O[{q_row0}:{q_row0+Br},0:{d_head}]',
             'rows': Br, 'cols': d_head},
            deps=[normalize, prev_dma])   # wait for last load + normalize

        store_L = add('dma_store', 'dma',
            f'[Q{qi}] Store L[{q_row0}:{q_row0+Br}] → HBM',
            {'source':      'shared_obuf.L_tile',
             'destination': f'HBM.L[{q_row0}:{q_row0+Br}]',
             'length': Br},
            deps=[logsumexp, store_O])

        prev_store_L = store_L

    n = len(instrs)
    return {
        'metadata': {
            'type':        'fa2_full_matrix',
            'description': (
                f'FA2 pipelined schedule: Nq={Nq}, Nkv={Nkv}, '
                f'Br={Br}, Bc={Bc}, d_head={d_head}.  '
                f'Run with: sim_main --schedule <this_file> --no-trace'
            ),
            'Nq': Nq, 'Nkv': Nkv, 'Br': Br, 'Bc': Bc, 'd_head': d_head,
            'total_instructions': n,
        },
        'schedule': instrs,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Generate pipelined FA2 schedule YAML',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--Nq',     type=int, default=32,
                   help='Q tiles  (seq_len / Br;  LLaMA-3-8B single head = 32)')
    p.add_argument('--Nkv',    type=int, default=32,
                   help='KV tiles (seq_len / Bc;  LLaMA-3-8B single head = 32)')
    p.add_argument('--Br',     type=int, default=128,
                   help='Q tile rows  (= systolic array rows)')
    p.add_argument('--Bc',     type=int, default=128,
                   help='KV tile cols (= systolic array cols)')
    p.add_argument('--d_head', type=int, default=128,
                   help='Attention head dimension')
    p.add_argument('--output', '-o', default='-',
                   help='Output path (- = stdout)')
    args = p.parse_args()

    doc = gen_fa2_schedule(args.Nq, args.Nkv, args.Br, args.Bc, args.d_head)
    n   = doc['metadata']['total_instructions']

    out = open(args.output, 'w', encoding='utf-8') if args.output != '-' else sys.stdout
    out.write(_dump(doc))
    if args.output != '-':
        out.close()
        print(f'Wrote {n} instructions → {args.output}', file=sys.stderr)
    else:
        print(f'# {n} instructions total', file=sys.stderr)


if __name__ == '__main__':
    main()
