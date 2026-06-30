"""Source-blind translator ENSEMBLING for multi-model mixing.

Averages N independently-trained translators per source and compares the ensemble
(N) against the single translator (N=1) on the same mixing assembly, over several
permutation seeds. Every translator stays source-blind: it sees only its own
(source, target) pair. For each source, N translators are trained (global-retr
retr=6, different torch seeds) and their test-space outputs are averaged.

Run:
    PYTHONPATH=src:.  python scripts/repro/exp_mixing_ensemble.py \
        --ensemble 1,3 --perm-seeds 3 --out output/repro/mixing_ensemble.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.repro.harness import (
    hmoe_config, load_beir_meta, load_embeddings, recall_at_k, train_hmoe,
)

SRC_MODELS = ["kalm", "openai"]
TGT_MODEL = "nemotron"
TRAIN_DS, TEST_DS = "fiqa", "scifact"


def _build_mixed(n, dim, perm, native, translated):
    n_parts = len(translated) + 1
    part = n // n_parts
    out = np.empty((n, dim), dtype=np.float32)
    out[perm[:part]] = native[perm[:part]]
    for i, (_src, tr) in enumerate(translated.items(), 1):
        start, end = i * part, (n if i == n_parts - 1 else (i + 1) * part)
        out[perm[start:end]] = tr[perm[start:end]]
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/repro/mixing_ensemble.json")
    ap.add_argument("--ensemble", default="1,3", help="comma list of ensemble sizes N")
    ap.add_argument("--perm-seeds", type=int, default=3, help="mixing-permutation seeds to average over")
    ap.add_argument("--retr-weight", type=float, default=6.0)
    ap.add_argument("--base-loss", default="cos")
    ap.add_argument("--base-epochs", type=int, default=80)
    ap.add_argument("--lora-epochs", type=int, default=60)
    args = ap.parse_args()
    ens_sizes = [int(e) for e in args.ensemble.split(",")]
    n_members = max(ens_sizes)
    perm_seeds = list(range(args.perm_seeds))

    corpus_ids, query_ids, qrels = load_beir_meta(TEST_DS)
    tgt_corpus = load_embeddings(TGT_MODEL, TEST_DS, "corpus")
    tgt_queries = load_embeddings(TGT_MODEL, TEST_DS, "query")
    n, dim = tgt_corpus.shape
    train_tgt = load_embeddings(TGT_MODEL, TRAIN_DS, "corpus")
    direct_R = recall_at_k(tgt_corpus, tgt_queries, qrels, corpus_ids, query_ids, k=100)

    # Train n_members translators per source once; cache each member's raw test output.
    members = {src: [] for src in SRC_MODELS}   # src -> list of (n,dim) test translations
    for src in SRC_MODELS:
        train_src = load_embeddings(src, TRAIN_DS, "corpus")
        test_src = load_embeddings(src, TEST_DS, "corpus")
        for member in range(n_members):
            torch.manual_seed(member)
            np.random.seed(member)
            cfg = hmoe_config(base_loss=args.base_loss, beta=0.0, beta_base=0.0,
                              retr_weight=args.retr_weight, base_epochs=args.base_epochs,
                              lora_epochs=args.lora_epochs)
            mapper, _ = train_hmoe(train_src, train_tgt, np.arange(train_src.shape[0]), cfg)
            members[src].append(mapper.transform(test_src).astype(np.float32))
            print(f"  trained {src} member {member}")

    rows = []
    for N in ens_sizes:
        # ensemble translation = mean of the first N members (source-blind average)
        translated = {src: np.mean(members[src][:N], axis=0).astype(np.float32)
                      for src in SRC_MODELS}
        per_pair = [recall_at_k(translated[s], tgt_queries, qrels, corpus_ids, query_ids, k=100)
                    for s in translated]
        avg_pairwise = float(np.mean(per_pair + [direct_R]))
        mixed_runs, drop_runs = [], []
        for ps in perm_seeds:
            perm = np.random.default_rng(ps).permutation(n)
            mixed = _build_mixed(n, dim, perm, tgt_corpus, translated)
            r_mixed = recall_at_k(mixed, tgt_queries, qrels, corpus_ids, query_ids, k=100)
            mixed_runs.append(r_mixed)
            drop_runs.append((avg_pairwise - r_mixed) * 100.0)
        mm, ms = float(np.mean(mixed_runs)), float(np.std(mixed_runs))
        dm, ds = float(np.mean(drop_runs)), float(np.std(drop_runs))
        print(f"N={N}  per_pair={[round(p,3) for p in per_pair]}  "
              f"mixed={mm:.4f}+/-{ms:.4f}  drop={dm:.2f}+/-{ds:.2f}%  (perm_n={len(perm_seeds)})")
        rows.append({"ensemble": N, "per_pair": per_pair, "avg_pairwise": avg_pairwise,
                     "mixed_mean": mm, "mixed_std": ms, "drop_mean": dm, "drop_std": ds,
                     "mixed_runs": mixed_runs, "drop_runs": drop_runs})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"direct_R": direct_R, "perm_seeds": len(perm_seeds),
                                          "retr_weight": args.retr_weight, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
