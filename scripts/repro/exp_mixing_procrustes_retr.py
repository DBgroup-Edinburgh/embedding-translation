"""Procrustes translation + global-retr residual calibration (source-blind).

Combines mean-centered orthogonal Procrustes with a small per-source residual MLP:

    out_s(x) = P_s(x) + g_s(P_s(x))

P_s = mean-centered orthogonal Procrustes (closed form). g_s is a small residual
MLP (last layer zero-init, so out == P_s at start). g_s is trained with
cos(out, y) + retr_w * InfoNCE(anchor=y, positive=out, negatives=in-batch +
native pool). Each source's g_s sees only its own pair + the shared native
target — fully source-blind.

Run:
    PYTHONPATH=src:.  python scripts/repro/exp_mixing_procrustes_retr.py \
        --retr-weights 0,3,6 --seeds 3 --out output/repro/mixing_procrustes_retr.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.repro.harness import load_beir_meta, load_embeddings, recall_at_k

SRC_MODELS = ["kalm", "openai"]
TGT_MODEL = "nemotron"
TRAIN_DS, TEST_DS = "fiqa", "scifact"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def procrustes_fit(src_train, tgt_train):
    sm = src_train.mean(0, keepdims=True)
    tm = tgt_train.mean(0, keepdims=True)
    M = (tgt_train - tm).T @ (src_train - sm)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = (U @ Vt).astype(np.float32)                      # (d_tgt, d_src)
    return sm.astype(np.float32), tm.astype(np.float32), W


def procrustes_apply(x, sm, tm, W):
    return (x - sm) @ W.T + tm


class Residual(nn.Module):
    """Residual on top of Procrustes. rank=0 -> SELU MLP (high capacity);
    rank>0 -> low-rank linear residual z + B(A z) (regularized, less OOD overfit)."""
    def __init__(self, d, h=1024, rank=0):
        super().__init__()
        self.rank = rank
        if rank > 0:
            self.A = nn.Linear(d, rank, bias=False)
            self.B = nn.Linear(rank, d, bias=False)
            nn.init.zeros_(self.B.weight)                              # start at identity
        else:
            self.f1 = nn.Linear(d, h)
            self.act = nn.SELU()
            self.f2 = nn.Linear(h, d)
            nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)

    def forward(self, z):
        if self.rank > 0:
            return z + self.B(self.A(z))
        return z + self.f2(self.act(self.f1(z)))


def train_residual(z_train, y_train, retr_w, pool, seed, tau=0.05, epochs=40, bs=1024,
                   rank=0, weight_decay=0.0, cos_weight=1.0):
    torch.manual_seed(seed)
    d = z_train.shape[1]
    g = Residual(d, rank=rank).to(DEVICE)
    opt = torch.optim.Adam(g.parameters(), lr=1e-4, weight_decay=weight_decay)
    z = torch.from_numpy(z_train).float()
    y = torch.from_numpy(y_train).float()
    N = z.shape[0]
    pool_t = nn.functional.normalize(torch.from_numpy(pool).float().to(DEVICE), dim=1)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            zb = z[idx].to(DEVICE); yb = y[idx].to(DEVICE)
            out = g(zb)
            o = nn.functional.normalize(out, dim=1)
            t = nn.functional.normalize(yb, dim=1)
            cos = 1 - (o * t).sum(1).mean()
            loss = cos_weight * cos
            if retr_w > 0 and o.shape[0] >= 8:
                cand = torch.cat([o, pool_t], dim=0)
                logits = (t @ cand.t()) / tau
                lbl = torch.arange(o.shape[0], device=DEVICE)
                with torch.no_grad():
                    dup = (t @ pool_t.t()) > 0.999
                logits[:, o.shape[0]:] = logits[:, o.shape[0]:].masked_fill(dup, float("-inf"))
                loss = cos + retr_w * nn.functional.cross_entropy(logits, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
    g.eval()
    return g


def _build_mixed(n, dim, perm, native, translated):
    n_parts = len(translated) + 1
    part = n // n_parts
    out = np.empty((n, dim), dtype=np.float32)
    out[perm[:part]] = native[perm[:part]]
    for i, (_s, tr) in enumerate(translated.items(), 1):
        start, end = i * part, (n if i == n_parts - 1 else (i + 1) * part)
        out[perm[start:end]] = tr[perm[start:end]]
    return out


@torch.no_grad()
def apply_residual(g, z):
    out = np.empty_like(z)
    zt = torch.from_numpy(z).float()
    for i in range(0, z.shape[0], 4096):
        out[i:i + 4096] = g(zt[i:i + 4096].to(DEVICE)).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retr-weights", default="0,3,6")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--perm-seeds", type=int, default=5)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--residual-rank", type=int, default=0, help="0=MLP residual; >0=low-rank linear")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--out", default="output/repro/mixing_procrustes_retr.json")
    args = ap.parse_args()
    retr_weights = [float(w) for w in args.retr_weights.split(",")]
    seeds = list(range(args.seeds))
    perm_seeds = list(range(args.perm_seeds))

    corpus_ids, query_ids, qrels = load_beir_meta(TEST_DS)
    tgt_corpus = load_embeddings(TGT_MODEL, TEST_DS, "corpus")
    tgt_queries = load_embeddings(TGT_MODEL, TEST_DS, "query")
    n, dim = tgt_corpus.shape
    tgt_train = load_embeddings(TGT_MODEL, TRAIN_DS, "corpus")
    direct_R = recall_at_k(tgt_corpus, tgt_queries, qrels, corpus_ids, query_ids, k=100)

    # Precompute Procrustes train/test outputs per source + native pool.
    pool_idx = np.sort(np.linspace(0, tgt_train.shape[0] - 1, args.pool_size).astype(np.int64))
    pool = np.ascontiguousarray(tgt_train[pool_idx], dtype=np.float32)
    proc = {}
    for src in SRC_MODELS:
        src_train = load_embeddings(src, TRAIN_DS, "corpus")
        src_test = load_embeddings(src, TEST_DS, "corpus")
        sm, tm, W = procrustes_fit(src_train, tgt_train)
        proc[src] = {
            "z_train": procrustes_apply(src_train, sm, tm, W).astype(np.float32),
            "z_test": procrustes_apply(src_test, sm, tm, W).astype(np.float32),
            "y_train": tgt_train,
        }
    print(f"PROCRUSTES+RETR  {SRC_MODELS} -> {TGT_MODEL}")

    rows = []
    for retr_w in retr_weights:
        mixed_runs, drop_runs, pp_runs = [], [], []
        for seed in seeds:
            translated = {}
            for src in SRC_MODELS:
                if retr_w == 0:
                    translated[src] = proc[src]["z_test"]            # Procrustes only
                else:
                    g = train_residual(proc[src]["z_train"], proc[src]["y_train"],
                                       retr_w, pool, seed, rank=args.residual_rank,
                                       weight_decay=args.weight_decay, cos_weight=args.cos_weight)
                    translated[src] = apply_residual(g, proc[src]["z_test"])
            per_pair = {s: recall_at_k(translated[s], tgt_queries, qrels, corpus_ids, query_ids, k=100)
                        for s in SRC_MODELS}
            avg_pairwise = float(np.mean(list(per_pair.values()) + [direct_R]))
            for ps in perm_seeds:
                perm = np.random.default_rng(ps).permutation(n)
                mixed = _build_mixed(n, dim, perm, tgt_corpus, translated)
                r_mixed = recall_at_k(mixed, tgt_queries, qrels, corpus_ids, query_ids, k=100)
                mixed_runs.append(r_mixed)
                drop_runs.append((avg_pairwise - r_mixed) * 100.0)
            pp_runs.append(per_pair)
            print(f"  retr={retr_w} seed{seed} per_pair={ {s: round(v,3) for s,v in per_pair.items()} } "
                  f"avg_pw={avg_pairwise:.3f}")
            if retr_w == 0:
                break  # deterministic, one pass is enough
        mm, ms = float(np.mean(mixed_runs)), float(np.std(mixed_runs))
        dm, ds = float(np.mean(drop_runs)), float(np.std(drop_runs))
        print(f"retr={retr_w}  mixed={mm:.4f}+/-{ms:.4f}  drop={dm:.2f}+/-{ds:.2f}%")
        rows.append({"retr_weight": retr_w, "mixed_mean": mm, "mixed_std": ms,
                     "drop_mean": dm, "drop_std": ds, "per_pair_runs": pp_runs})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"direct_R": direct_R, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
