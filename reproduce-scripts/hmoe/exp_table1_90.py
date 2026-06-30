"""Table 1 -- Cross-data (OOD) pairwise translation, Recall@100 (ICML'26 paper).

Reproduces the paper's Table 1 setting: translators are trained on **Fever** and
evaluated out-of-distribution on multiple target datasets -- SciDocs, ArguAna,
FiQA-2018, NFCorpus, SciFact (paper Sec. 5 / Table 1, "we adopt the setting from
Fever to multiple target datasets"). All 10 embedding models are used as both
source and target -> 10x9 = 90 directed translation pairs; each pair is evaluated
on every OOD target dataset.

For a pair (src, tgt): train f: emb_src -> emb_tgt on the Fever corpus, then for
each OOD test dataset translate emb_src(test corpus) and measure Recall@100 of the
real emb_tgt(test queries) against the translated corpus.

Embeddings are the precomputed VectorBenchmark vectors (HF dataset
DB-Edinburgh/VectorBenchmark); set VB_DIR to where they are stored
(see reproduce-scripts/README.md).

Methods: hmoe (H-MoE, K=8 on Fever, r=8). 

Run (resumable; one method per invocation):
    METHOD=hmoe VB_DIR=... OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
        python reproduce-scripts/hmoe/exp_table1_90.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))   # reproduce-scripts/ (harness)
from harness import (
    hmoe_config, load_beir_meta, load_embeddings, recall_at_k, train_hmoe,
)

# The 10 paper embedding models (Appendix G).
MODELS = ["kalm", "nemotron", "qwen", "gemini", "linq", "e5", "sfr", "gritlm", "openai", "mistral"]
TRAIN_DS = "fever"
# OOD target datasets (paper Sec. 5 / Table 1). Override with TEST_DATASETS="a,b".
TEST_DATASETS = os.environ.get(
    "TEST_DATASETS", "scidocs,arguana,fiqa,nfcorpus,scifact"
).split(",")


def procrustes_fit(src_train, tgt_train):
    sm = src_train.mean(0, keepdims=True)
    tm = tgt_train.mean(0, keepdims=True)
    M = (tgt_train - tm).T @ (src_train - sm)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return sm.astype(np.float32), tm.astype(np.float32), (U @ Vt).astype(np.float32)


def fit_translator(src, tgt, method):
    """Train a (src->tgt) translator on the Fever corpus; return a transform fn."""
    src_tr = load_embeddings(src, TRAIN_DS, "corpus")
    tgt_tr = load_embeddings(tgt, TRAIN_DS, "corpus")
    keep = ~(np.isnan(src_tr).any(1) | np.isnan(tgt_tr).any(1))
    if not keep.all():
        src_tr, tgt_tr = src_tr[keep], tgt_tr[keep]

    if method == "procrustes":
        sm, tm, W = procrustes_fit(src_tr, tgt_tr)
        return lambda x: ((x - sm) @ W.T + tm).astype(np.float32)

    if method == "hmoe":
        # H-MoE: K=8 on Fever (num_levels=4, branch_factor=2), r=8, cos base loss
        # + global-retrieval objective (retr_weight=6); L_dir off (beta=0).
        cfg = hmoe_config(num_levels=4, branch_factor=2, lora_rank=8,
                          base_loss="cos", beta=0.0, beta_base=0.0,
                          retr_weight=6.0, base_epochs=80, lora_epochs=60)
        mapper, _ = train_hmoe(src_tr, tgt_tr, np.arange(src_tr.shape[0]), cfg)
        return mapper.transform

    raise ValueError(method)


def score_on(transform, src, tgt, dataset):
    """Translate src(dataset corpus), eval Recall@100 vs tgt(dataset queries)."""
    corpus_ids, query_ids, qrels = load_beir_meta(dataset)
    src_te = np.nan_to_num(load_embeddings(src, dataset, "corpus"))
    tgt_q = load_embeddings(tgt, dataset, "query")
    translated = transform(src_te)
    return recall_at_k(translated, tgt_q, qrels, corpus_ids, query_ids, k=100)


def main() -> None:
    method = os.environ.get("METHOD", "hmoe")
    out = Path(os.environ.get("OUT", f"output/repro/table1_{method}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}

    pairs = [(s, t) for s in MODELS for t in MODELS if s != t]   # 10x9 = 90 directions
    print(f"METHOD={method}  train={TRAIN_DS}  test={TEST_DATASETS}  "
          f"{len(pairs)} directed pairs  {len(done)} cells already done")

    for i, (src, tgt) in enumerate(pairs):
        pending = [d for d in TEST_DATASETS if f"{src}->{tgt}@{d}" not in done]
        if not pending:
            continue
        try:
            transform = fit_translator(src, tgt, method)        # train once on Fever
        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] {src}->{tgt} TRAIN FAILED: {e}")
            continue
        for dataset in pending:
            key = f"{src}->{tgt}@{dataset}"
            try:
                r = score_on(transform, src, tgt, dataset)
            except Exception as e:
                print(f"    {key} FAILED: {e}")
                continue
            done[key] = r
            out.write_text(json.dumps(done, indent=2))          # checkpoint every cell
            print(f"  [{i+1}/{len(pairs)}] {key}  R@100={r:.4f}")

    if done:
        for d in TEST_DATASETS:
            vals = [v for k, v in done.items() if k.endswith(f"@{d}")]
            if vals:
                print(f"METHOD={method}  {d:9s} mean R@100 over {len(vals)} pairs = {np.mean(vals):.4f}")
        print(f"METHOD={method}  overall mean R@100 over {len(done)} cells = {np.mean(list(done.values())):.4f}")


if __name__ == "__main__":
    main()
