"""Mixing with Procrustes translators (paper setup, source-blind).

Uses mean-centered orthogonal Procrustes as the per-source translator for the
paper mixing setup ({kalm,openai}->nemotron, FiQA->SciFact). Each translator is
source-blind (only its own pair). Procrustes is deterministic, so only the merge
permutation varies across seeds.

Run:
    PYTHONPATH=src:.  python scripts/repro/exp_mixing_procrustes.py \
        --sources kalm,openai --target nemotron --out output/repro/mixing_procrustes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.repro.harness import load_beir_meta, load_embeddings, recall_at_k


def procrustes_translate(src_train, tgt_train, src_test):
    sm = src_train.mean(0, keepdims=True)
    tm = tgt_train.mean(0, keepdims=True)
    M = (tgt_train - tm).T @ (src_train - sm)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    return ((src_test - sm) @ W.T + tm).astype(np.float32)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="kalm,openai")
    ap.add_argument("--target", default="nemotron")
    ap.add_argument("--train-ds", default="fiqa")
    ap.add_argument("--test-ds", default="scifact")
    ap.add_argument("--perm-seeds", type=int, default=5)
    ap.add_argument("--out", default="output/repro/mixing_procrustes.json")
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",")]
    perm_seeds = list(range(args.perm_seeds))
    print(f"PROCRUSTES MIXING: {sources} -> {args.target}  ({args.train_ds}->{args.test_ds})")

    corpus_ids, query_ids, qrels = load_beir_meta(args.test_ds)
    tgt_corpus = load_embeddings(args.target, args.test_ds, "corpus")
    tgt_queries = load_embeddings(args.target, args.test_ds, "query")
    n, dim = tgt_corpus.shape
    tgt_train = load_embeddings(args.target, args.train_ds, "corpus")
    direct_R = recall_at_k(tgt_corpus, tgt_queries, qrels, corpus_ids, query_ids, k=100)

    translated = {}
    for src in sources:
        src_train = load_embeddings(src, args.train_ds, "corpus")
        src_test = load_embeddings(src, args.test_ds, "corpus")
        translated[src] = procrustes_translate(src_train, tgt_train, src_test)
    per_pair = {s: recall_at_k(translated[s], tgt_queries, qrels, corpus_ids, query_ids, k=100)
                for s in sources}
    avg_pairwise = float(np.mean(list(per_pair.values()) + [direct_R]))

    mixed_runs, drop_runs = [], []
    for ps in perm_seeds:
        perm = np.random.default_rng(ps).permutation(n)
        mixed = _build_mixed(n, dim, perm, tgt_corpus, translated)
        r_mixed = recall_at_k(mixed, tgt_queries, qrels, corpus_ids, query_ids, k=100)
        mixed_runs.append(r_mixed)
        drop_runs.append((avg_pairwise - r_mixed) * 100.0)
    mm, ms = float(np.mean(mixed_runs)), float(np.std(mixed_runs))
    dm, ds = float(np.mean(drop_runs)), float(np.std(drop_runs))
    print(f"per_pair={ {s: round(v,3) for s,v in per_pair.items()} }  native={direct_R:.3f}")
    print(f"mixed={mm:.4f}+/-{ms:.4f}  drop={dm:.2f}+/-{ds:.2f}%  (perm_n={len(perm_seeds)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "sources": sources, "target": args.target, "direct_R": direct_R,
        "per_pair": per_pair, "avg_pairwise": avg_pairwise,
        "mixed_mean": mm, "mixed_std": ms, "drop_mean": dm, "drop_std": ds,
    }, indent=2))


if __name__ == "__main__":
    main()
