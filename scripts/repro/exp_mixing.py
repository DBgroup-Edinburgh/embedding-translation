"""Multi-Model Mixing (paper Section 5.4 / Fig 9a, Appendix H).

Sources {kalm, openai} are translated into a shared target space (nemotron);
translators are trained on FiQA and applied OOD on SciFact. The mixed corpus is
assembled 3-way (1/3 native nemotron, 1/3 from f_{kalm->nemotron}, 1/3 from
f_{openai->nemotron}) and searched with native nemotron queries.

drop = avg-pairwise Recall@100 - mixed Recall@100. The H-MoE translators use our
config: cos base loss + global-retrieval objective (retr_weight=6), L_dir off (beta=0).

Run:
    PYTHONPATH=src:.  python scripts/repro/exp_mixing.py --out output/repro/mixing.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.repro.harness import (
    hmoe_config, load_beir_meta, load_embeddings, recall_at_k, train_hmoe,
)

# Mixing setup per paper Appendix H ({kalm,openai}->nemotron, FiQA->SciFact);
# overridable via --sources/--target/--train-ds/--test-ds.
SRC_MODELS = ["kalm", "openai"]
TGT_MODEL = "nemotron"
TRAIN_DS, TEST_DS = "fiqa", "scifact"


def _fit_align(tr_train, tgt_train, mode):
    """Per-source output distribution alignment to the native target moments.

    Stats come from the TRAINING set only (no test labels) — the affine map is a
    fixed property of the deployed translator. Applied identically to per-pair
    and mixed eval so the drop comparison stays honest.

    - "mean":       shift translated cloud to the native target mean.
    - "coral_diag": shift + per-dim std rescale (diagonal CORAL).
    Returns a callable applied to test-time translated outputs.
    """
    mu_s = tr_train.mean(axis=0, keepdims=True)
    mu_t = tgt_train.mean(axis=0, keepdims=True)
    if mode == "mean":
        return lambda x: (x - mu_s + mu_t).astype(np.float32)
    if mode == "coral_diag":
        sd_s = tr_train.std(axis=0, keepdims=True) + 1e-6
        sd_t = tgt_train.std(axis=0, keepdims=True)
        scale = (sd_t / sd_s).astype(np.float32)
        return lambda x: ((x - mu_s) * scale + mu_t).astype(np.float32)
    return lambda x: x


def _build_mixed(n, dim, perm, native, translated):
    """1 native part + one part per translated source, assembled by perm."""
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
    ap.add_argument("--out", default="output/repro/mixing.json")
    ap.add_argument("--base-epochs", type=int, default=80)
    ap.add_argument("--lora-epochs", type=int, default=60)
    ap.add_argument("--betas", default="0", help="comma list of L_dir beta values")
    ap.add_argument("--retr-weights", default="6", help="comma list of global-retr weights")
    ap.add_argument("--retr-pool-sizes", default="2048", help="comma list of global negative-pool sizes")
    ap.add_argument("--retr-taus", default="0.05", help="comma list of global-retr InfoNCE temperatures")
    ap.add_argument("--retr-hard-ks", default="0",
                    help="comma list of hard-negative-mining sizes (0=uniform pool)")
    ap.add_argument("--align", default="none",
                    help="comma list of per-source output alignment modes to native "
                         "target moments: none|mean|coral_diag (evaluated on shared translators)")
    ap.add_argument("--sources", default=None, help="comma list of source models (override)")
    ap.add_argument("--target", default=None, help="shared target model (override)")
    ap.add_argument("--train-ds", default=None, help="train dataset (override)")
    ap.add_argument("--test-ds", default=None, help="test dataset (override)")
    ap.add_argument("--base-loss", default="cos")
    ap.add_argument("--inner-loss", default="mse_cos", help="LoRA-expert L_reg loss kind")
    ap.add_argument("--seeds", type=int, default=3, help="repeats per config for mean+/-std")
    args = ap.parse_args()
    betas = [float(b) for b in args.betas.split(",")]
    retr_weights = [float(w) for w in args.retr_weights.split(",")]
    pool_sizes = [int(p) for p in args.retr_pool_sizes.split(",")]
    taus = [float(t) for t in args.retr_taus.split(",")]
    hard_ks = [int(h) for h in args.retr_hard_ks.split(",")]
    seeds = list(range(args.seeds))

    global SRC_MODELS, TGT_MODEL, TRAIN_DS, TEST_DS
    if args.sources:   SRC_MODELS = [s.strip() for s in args.sources.split(",")]
    if args.target:    TGT_MODEL = args.target
    if args.train_ds:  TRAIN_DS = args.train_ds
    if args.test_ds:   TEST_DS = args.test_ds
    print(f"MIXING setup: sources={SRC_MODELS} -> target={TGT_MODEL}  ({TRAIN_DS}->{TEST_DS})")

    corpus_ids, query_ids, qrels = load_beir_meta(TEST_DS)
    tgt_corpus = load_embeddings(TGT_MODEL, TEST_DS, "corpus")
    tgt_queries = load_embeddings(TGT_MODEL, TEST_DS, "query")
    n, dim = tgt_corpus.shape
    train_tgt = load_embeddings(TGT_MODEL, TRAIN_DS, "corpus")
    direct_R = recall_at_k(tgt_corpus, tgt_queries, qrels, corpus_ids, query_ids, k=100)

    align_modes = [m.strip() for m in args.align.split(",")]

    rows = []
    for beta, retr_w, pool, tau, hard_k in [(b, w, p, t, h) for b in betas for w in retr_weights
                                            for p in pool_sizes for t in taus for h in hard_ks]:
        # per align mode: lists of per-seed (mixed, drop)
        acc = {m: {"mixed": [], "drop": []} for m in align_modes}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            perm = np.random.default_rng(seed).permutation(n)
            # Train each source ONCE; cache its raw test+train translations so every
            # align mode is measured on identical translators (paired comparison).
            raw_test, aligners = {}, {m: {} for m in align_modes}
            for src in SRC_MODELS:
                train_src = load_embeddings(src, TRAIN_DS, "corpus")
                cfg = hmoe_config(base_loss=args.base_loss, inner_loss=args.inner_loss,
                                  beta=beta, beta_base=beta,
                                  retr_weight=retr_w, retr_tau=tau,
                                  retr_pool_size=pool, retr_hard_k=hard_k,
                                  base_epochs=args.base_epochs,
                                  lora_epochs=args.lora_epochs)
                mapper, _ = train_hmoe(train_src, train_tgt, np.arange(train_src.shape[0]), cfg)
                raw_test[src] = mapper.transform(load_embeddings(src, TEST_DS, "corpus"))
                tr_train = mapper.transform(train_src)
                for m in align_modes:
                    aligners[m][src] = _fit_align(tr_train, train_tgt, m)
            for m in align_modes:
                translated = {s: aligners[m][s](raw_test[s]) for s in SRC_MODELS}
                per_pair = [recall_at_k(translated[s], tgt_queries, qrels, corpus_ids, query_ids, k=100)
                            for s in translated]
                avg_pairwise = float(np.mean(per_pair + [direct_R]))
                mixed = _build_mixed(n, dim, perm, tgt_corpus, translated)
                r_mixed = recall_at_k(mixed, tgt_queries, qrels, corpus_ids, query_ids, k=100)
                drop = (avg_pairwise - r_mixed) * 100.0
                acc[m]["mixed"].append(r_mixed)
                acc[m]["drop"].append(drop)
                pp = {s: round(float(v), 3) for s, v in zip(SRC_MODELS, per_pair)}
                print(f"  seed{seed} b={beta} retr={retr_w} pool={pool} tau={tau} hardk={hard_k} align={m} "
                      f"per_pair={pp} native={direct_R:.3f} mixed={r_mixed:.4f} drop={drop:.2f}%")
        for m in align_modes:
            mm, ms = float(np.mean(acc[m]["mixed"])), float(np.std(acc[m]["mixed"]))
            dm, ds = float(np.mean(acc[m]["drop"])), float(np.std(acc[m]["drop"]))
            print(f"b={beta} retr={retr_w} pool={pool} tau={tau} hardk={hard_k} align={m}  mixed={mm:.4f}+/-{ms:.4f}  drop={dm:.2f}+/-{ds:.2f}%  (n={len(seeds)})")
            rows.append({"beta": beta, "retr_weight": retr_w, "retr_pool_size": pool,
                         "retr_tau": tau, "retr_hard_k": hard_k, "align": m, "mixed_mean": mm, "mixed_std": ms,
                         "drop_mean": dm, "drop_std": ds,
                         "mixed_runs": acc[m]["mixed"], "drop_runs": acc[m]["drop"]})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"direct_R": direct_R, "seeds": len(seeds),
                                           "align": args.align, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
