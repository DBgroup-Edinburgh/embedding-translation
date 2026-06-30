"""LA2M integration scan across many (source, target, dataset) cells.

For each cell, runs the same integration protocol as exp_la2m.py:
  - union  (no-alignment baseline)
  - LA2M over a small-n grid (n in {4,8,16}); reports the best-n R1 per cell.
Each cell records union R/R1, LA2M R/R1 per n, the best-n R1, the native emb2
ceiling, and R1 as a fraction of that ceiling.

Target must have query embeddings cached (query with emb2(q)); see the cell list.
mistral is excluded as a target in this run.

Resumable: one JSON keyed by "src->tgt@dataset"; done cells are skipped.

Run (numpy-heavy -> cap BLAS threads):
    VB_DIR=... PYTHONPATH=src:. OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
        MKL_NUM_THREADS=16 <venv>/bin/python reproduce-scripts/la2m/exp_la2m_matrix.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parent),       # sibling modules (exp_la2m)
                 str(_pl.Path(__file__).resolve().parents[1])]   # reproduce-scripts/ (harness)
from harness import load_beir_meta, load_embeddings, recall_at_k
from exp_la2m import split_o1_cap_o2, evaluate

# Diverse cells: varied source/target dims, 4 datasets. Targets restricted to
# models with cached query embeddings on that dataset (SciFact: all; FiQA/
# ArguAna/SciDocs: gte,kalm only -- ArguAna/SciDocs only have gte,kalm corpus).
CELLS = [
    # SciFact -- rich; target openai (1536) and kalm (3840), varied sources
    ("mistral", "openai", "scifact"),     # anchor (paper Table 4 proxy)
    ("kalm",    "openai", "scifact"),
    ("gte",     "openai", "scifact"),
    ("sfr",     "openai", "scifact"),
    ("nemotron","kalm",   "scifact"),
    ("e5",      "kalm",   "scifact"),
    ("qwen",    "kalm",   "scifact"),
    ("openai",  "kalm",   "scifact"),
    # FiQA -- target gte (3584) / kalm (3840)
    ("mistral", "kalm",   "fiqa"),
    ("openai",  "kalm",   "fiqa"),
    ("sfr",     "gte",    "fiqa"),
    ("e5",      "gte",    "fiqa"),
    # ArguAna / SciDocs -- only gte<->kalm available
    ("gte",     "kalm",   "arguana"),
    ("kalm",    "gte",    "arguana"),
    ("gte",     "kalm",   "scidocs"),
    ("kalm",    "gte",    "scidocs"),
]
N_GRID = [4, 8, 16]
SEEDS = 2
REF_RATIO = 0.20


def run_cell(src_m, tgt_m, dataset):
    corpus_ids, query_ids, qrels = load_beir_meta(dataset)
    src = np.nan_to_num(load_embeddings(src_m, dataset, "corpus"))
    tgt = np.nan_to_num(load_embeddings(tgt_m, dataset, "corpus"))
    tgt_q = np.nan_to_num(load_embeddings(tgt_m, dataset, "query"))
    id2row = {cid: i for i, cid in enumerate(corpus_ids)}
    answer_rows = sorted({id2row[d] for q in query_ids
                          for d, r in qrels.get(q, {}).items()
                          if r > 0 and d in id2row})
    ceiling = recall_at_k(tgt, tgt_q, qrels, corpus_ids, query_ids, k=100)

    def avg(method, n):
        Rs, R1s = [], []
        for s in range(SEEDS):
            o1, o_cap, o2 = split_o1_cap_o2(src.shape[0], answer_rows, REF_RATIO, s)
            R, R1, _ = evaluate(method, src, tgt, tgt_q, o1, o_cap, o2, n, s,
                                qrels, corpus_ids, query_ids, center=True)
            Rs.append(R); R1s.append(R1)
        return float(np.mean(Rs)), float(np.mean(R1s))

    union_R, union_R1 = avg("union", 0)
    la2m = {}
    for n in N_GRID:
        la2m[n] = avg("la2m", n)
    best_n = max(N_GRID, key=lambda n: la2m[n][1])
    best_R, best_R1 = la2m[best_n]
    return {
        "src": src_m, "tgt": tgt_m, "dataset": dataset,
        "d_src": int(src.shape[1]), "d_tgt": int(tgt.shape[1]),
        "n_docs": int(src.shape[0]), "ceiling": ceiling,
        "union_R": union_R, "union_R1": union_R1,
        "la2m_by_n": {str(n): {"R": la2m[n][0], "R1": la2m[n][1]} for n in N_GRID},
        "best_n": best_n, "la2m_best_R": best_R, "la2m_best_R1": best_R1,
        "R1_frac_of_ceiling": best_R1 / ceiling if ceiling > 0 else 0.0,
    }


def main() -> None:
    out = Path(os.environ.get("OUT", "output/repro/la2m_matrix.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}
    print(f"LA2M generality matrix: {len(CELLS)} cells, n_grid={N_GRID}, "
          f"seeds={SEEDS}, ref_ratio={REF_RATIO}; {len(done)} already done")
    for src_m, tgt_m, ds in CELLS:
        key = f"{src_m}->{tgt_m}@{ds}"
        if key in done:
            continue
        try:
            rec = run_cell(src_m, tgt_m, ds)
        except Exception as e:
            print(f"  {key:28s} FAILED: {e}")
            continue
        done[key] = rec
        out.write_text(json.dumps(done, indent=2))
        print(f"  {key:28s} ceil={rec['ceiling']*100:5.1f}  union_R1={rec['union_R1']*100:5.1f}  "
              f"LA2M_R1={rec['la2m_best_R1']*100:5.1f}(n={rec['best_n']})  "
              f"R={rec['la2m_best_R']*100:5.1f}  frac_ceil={rec['R1_frac_of_ceiling']*100:5.1f}%")

    vals = list(done.values())
    if vals:
        fr = np.mean([v["R1_frac_of_ceiling"] for v in vals])
        print(f"\n{len(vals)} cells: mean LA2M_best_R1 = {np.mean([v['la2m_best_R1'] for v in vals])*100:.1f}, "
              f"mean R1/ceiling = {fr*100:.1f}%, mean union_R1 = {np.mean([v['union_R1'] for v in vals])*100:.1f}")


if __name__ == "__main__":
    main()
