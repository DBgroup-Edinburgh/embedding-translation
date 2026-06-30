"""LA2M vector-database integration (SIGMOD'26 paper, Section 4.2 + Table 4).

Reproduces the paper's *integration* protocol (NOT the H-MoE pairwise-OOD one):

  1. Partition a benchmark corpus into three disjoint sets O1, O_cap (reference),
     O2. Built-in answers (gold docs) are distributed EVENLY across O1 and O2;
     the reference set O_cap holds no answers (paper Section 7.1 "Evaluation plan").
  2. emb1 encodes O1 u O_cap -> D1; emb2 encodes O2 u O_cap -> D2. The overlap
     O_cap gives reference pairs (X, Y) = (emb1[O_cap], emb2[O_cap]).
  3. An integration method learns T : emb1-space -> emb2-space from (X, Y) only
     (no data objects, no model access). The integrated DB is
        T(D1) u (D2 \ Y)
     i.e. every corpus doc lives once in emb2-space: O1 docs translated by T,
     O_cap + O2 docs native (reference docs map exactly to their known Y).
  4. Query with emb2(q); Recall@100 over the integrated DB.
     - R   : recall@100 over all queries.
     - R1  : recall@100[D1] -- only queries whose gold answer is in O1 (the hard
             half that REQUIRES a good translation; native emb2 cannot answer it).

Methods (paper Section 7.1):
  - union : no alignment -- raw D1 vectors zero-padded to emb2 dim (no-translation baseline).
  - a2m   : single global mean-centred orthogonal isometry (= LA2M with n=1).
  - la2m  : k-means grouping of references into n clusters (paper Fig 5 / Sec 7.3),
            one per-cluster orthogonal isometry, route each O1 vector to its
            nearest cluster centroid and apply that local isometry.

The orthogonal isometry is the cross-dim mean-centred Procrustes solution
(W = U V^T from svd((Y-y_bar)^T (X-x_bar)), shape (d2, d1)) -- distance-preserving
and identical to the closed form used throughout the H-MoE repro.

Run (numpy-heavy -> cap BLAS threads):
    PYTHONPATH=src:. OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
        python reproduce-scripts/la2m/exp_la2m.py --src mistral --tgt openai --dataset scifact \
        --n-clusters 1,20,50,100,200 --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parent),       # sibling modules
                 str(_pl.Path(__file__).resolve().parents[1])]   # reproduce-scripts/ (harness)
from harness import load_beir_meta, load_embeddings, recall_at_k


def procrustes_fit(src, tgt, center=True):
    """Orthogonal isometry src->tgt (cross-dim OK). Returns sm,tm,W.

    center=True (ours): mean-centred Procrustes -- subtract the source/target
    means before the SVD and add the target mean back at apply time, so the map
    captures the cloud translation as well as the rotation.
    center=False: a pure rotation about the origin (W from svd(tgt^T src), no
    mean terms), i.e. the un-centred isometry without mean-offset alignment.
    """
    if center:
        sm = src.mean(0, keepdims=True)
        tm = tgt.mean(0, keepdims=True)
    else:
        sm = np.zeros((1, src.shape[1]), dtype=np.float32)
        tm = np.zeros((1, tgt.shape[1]), dtype=np.float32)
    M = (tgt - tm).T @ (src - sm)                    # (d_tgt, d_src)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = (U @ Vt).astype(np.float32)                  # (d_tgt, d_src), semi-orthogonal
    return sm.astype(np.float32), tm.astype(np.float32), W


def procrustes_apply(x, sm, tm, W):
    return ((x - sm) @ W.T + tm).astype(np.float32)


def split_o1_cap_o2(n_docs, answer_rows, ref_ratio, seed):
    """Partition row indices [0,n_docs) into (O1, O_cap, O2).

    O_cap (reference): ref_ratio of NON-answer docs, no answers.
    Answer docs split evenly O1/O2 so ~half the queries' gold lands in O1.
    Remaining non-answer docs split evenly O1/O2.
    """
    rng = np.random.default_rng(seed)
    all_rows = np.arange(n_docs)
    is_ans = np.zeros(n_docs, dtype=bool)
    is_ans[answer_rows] = True
    non_ans = all_rows[~is_ans]
    ans = all_rows[is_ans]
    rng.shuffle(non_ans)
    rng.shuffle(ans)

    n_ref = int(round(ref_ratio * n_docs))
    n_ref = min(n_ref, len(non_ans))
    o_cap = non_ans[:n_ref]
    rest_non_ans = non_ans[n_ref:]

    # even split of answers and of leftover non-answers
    o1 = np.concatenate([ans[0::2], rest_non_ans[0::2]])
    o2 = np.concatenate([ans[1::2], rest_non_ans[1::2]])
    return np.sort(o1), np.sort(o_cap), np.sort(o2)


def translate_o1(method, src, tgt, o1, o_cap, n_clusters, seed, center=True):
    """Return translated emb2-space vectors for the O1 rows of `src`."""
    Xr = src[o_cap]                                   # reference source (X)
    Yr = tgt[o_cap]                                   # reference target (Y)
    Xo1 = src[o1]
    d_tgt = tgt.shape[1]

    if method == "union":
        out = np.zeros((len(o1), d_tgt), dtype=np.float32)
        c = min(src.shape[1], d_tgt)
        out[:, :c] = Xo1[:, :c]                        # raw, zero-padded -> no alignment
        return out

    if method == "a2m" or n_clusters <= 1:
        sm, tm, W = procrustes_fit(Xr, Yr, center=center)
        return procrustes_apply(Xo1, sm, tm, W)

    # la2m: k-means on references, per-cluster isometry, route O1 by nearest centroid
    km = KMeans(n_clusters=min(n_clusters, len(Xr)), n_init=4, random_state=seed)
    lbl = km.fit_predict(Xr)
    centroids = km.cluster_centers_.astype(np.float32)
    maps = {}
    for c in np.unique(lbl):
        m = lbl == c
        if m.sum() >= 2:
            maps[c] = procrustes_fit(Xr[m], Yr[m], center=center)
        else:                                          # tiny cluster -> global fallback
            maps[c] = None
    g_sm, g_tm, g_W = procrustes_fit(Xr, Yr, center=center)
    # nearest centroid for each O1 vector (k-means predict = argmin distance)
    assign = km.predict(Xo1)
    out = np.empty((len(o1), d_tgt), dtype=np.float32)
    for c in np.unique(assign):
        rows = np.where(assign == c)[0]
        params = maps.get(c) or (g_sm, g_tm, g_W)
        out[rows] = procrustes_apply(Xo1[rows], *params)
    return out


def evaluate(method, src, tgt, tgt_query, o1, o_cap, o2, n_clusters, seed,
             qrels, corpus_ids, query_ids, center=True):
    """Build integrated DB (all corpus rows, emb2-space) and return (R, R1)."""
    n_docs = src.shape[0]
    integrated = tgt.copy()                            # O_cap + O2 native; refs exact
    integrated[o1] = translate_o1(method, src, tgt, o1, o_cap, n_clusters, seed, center=center)

    R = recall_at_k(integrated, tgt_query, qrels, corpus_ids, query_ids, k=100)

    # R1: queries whose gold answer lies in O1 (the half needing translation).
    # recall_at_k pairs query ROWS to query_ids positionally, so the query
    # matrix must be sliced to the same subset (not just the id list).
    o1_set = set(int(i) for i in o1)
    id2row = {cid: i for i, cid in enumerate(corpus_ids)}
    pos_d1 = [i for i, qid in enumerate(query_ids)
              if any(id2row.get(did, -1) in o1_set
                     for did, r in qrels.get(qid, {}).items() if r > 0)]
    if pos_d1:
        sub_q = tgt_query[pos_d1]
        sub_ids = [query_ids[i] for i in pos_d1]
        R1 = recall_at_k(integrated, sub_q, qrels, corpus_ids, sub_ids, k=100)
    else:
        R1 = 0.0
    return R, R1, len(pos_d1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="mistral")
    ap.add_argument("--tgt", default="openai")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--ref-ratio", type=float, default=920 / 5183,  # paper Table 2 SciFact
                    help="fraction of corpus used as reference O_cap")
    ap.add_argument("--n-clusters", default="1,20,50,100,200")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--centers", default="1",
                    help="comma list of mean-centering toggles to sweep: 1=on, 0=off")
    ap.add_argument("--out", default="output/repro/la2m_mistral_openai_scifact.json")
    args = ap.parse_args()
    n_list = [int(x) for x in args.n_clusters.split(",")]
    seeds = list(range(args.seeds))
    center_list = [bool(int(x)) for x in args.centers.split(",")]

    corpus_ids, query_ids, qrels = load_beir_meta(args.dataset)
    src = load_embeddings(args.src, args.dataset, "corpus")
    tgt = load_embeddings(args.tgt, args.dataset, "corpus")
    tgt_query = load_embeddings(args.tgt, args.dataset, "query")
    id2row = {cid: i for i, cid in enumerate(corpus_ids)}
    answer_rows = sorted({id2row[did] for q in query_ids
                          for did, r in qrels.get(q, {}).items()
                          if r > 0 and did in id2row})
    print(f"LA2M  {args.src}({src.shape[1]}d) -> {args.tgt}({tgt.shape[1]}d)  "
          f"{args.dataset}: {src.shape[0]} docs, {len(query_ids)} queries, "
          f"{len(answer_rows)} answer docs, ref_ratio={args.ref_ratio:.3f}")

    # native emb2 ceiling (no integration; only O2/O_cap answers reachable)
    ceil = recall_at_k(tgt, tgt_query, qrels, corpus_ids, query_ids, k=100)
    print(f"native emb2 ceiling (full corpus) R@100 = {ceil:.4f}")

    results = {"src": args.src, "tgt": args.tgt, "dataset": args.dataset,
               "ref_ratio": args.ref_ratio, "ceiling_full": ceil, "rows": []}

    def run(method, n_clusters, center):
        Rs, R1s = [], []
        for seed in seeds:
            o1, o_cap, o2 = split_o1_cap_o2(src.shape[0], answer_rows, args.ref_ratio, seed)
            R, R1, nq1 = evaluate(method, src, tgt, tgt_query, o1, o_cap, o2,
                                  n_clusters, seed, qrels, corpus_ids, query_ids,
                                  center=center)
            Rs.append(R); R1s.append(R1)
        rm, rs = float(np.mean(Rs)), float(np.std(Rs))
        r1m, r1s = float(np.mean(R1s)), float(np.std(R1s))
        base = method if method != "la2m" else f"la2m(n={n_clusters})"
        label = base + ("" if center else " [no-center]")
        print(f"  {label:24s} R={rm*100:5.2f}+/-{rs*100:.2f}  R1={r1m*100:5.2f}+/-{r1s*100:.2f}  "
              f"(|D1 queries|~{nq1})")
        results["rows"].append({"method": label, "n_clusters": n_clusters, "center": center,
                                "R_mean": rm, "R_std": rs, "R1_mean": r1m, "R1_std": r1s})

    run("union", 0, True)                              # union has no isometry; centering n/a
    for center in center_list:
        run("a2m", 1, center)
        for n in n_list:
            if n <= 1:
                continue
            run("la2m", n, center)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
