"""Compare Procrustes and cosine H-MoE translation on three FiQA->SciFact OOD
pairs (kalm->nemotron, openai->nemotron, kalm->sfr). For each pair it reports the
native target self-retrieval R@100 (ceiling), Procrustes R@100, and H-MoE R@100.
"""

from __future__ import annotations

import numpy as np

from scripts.repro.harness import (
    hmoe_config, load_beir_meta, load_embeddings, recall_at_k, train_hmoe,
)


def procrustes_translate(src_train, tgt_train, src_test):
    """Mean-centered orthogonal Procrustes: W = U Vt from svd(tgt_c^T src_c)."""
    sm = src_train.mean(0, keepdims=True)
    tm = tgt_train.mean(0, keepdims=True)
    M = (tgt_train - tm).T @ (src_train - sm)          # (d_tgt, d_src)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt                                          # (d_tgt, d_src), orthogonal-ish
    return (src_test - sm) @ W.T + tm


def main() -> None:
    PAIRS = [("kalm", "nemotron"), ("openai", "nemotron"), ("kalm", "sfr")]
    TRAIN, TEST = "fiqa", "scifact"

    print(f"{'pair':22s} {'ceiling':>8} {'procrustes':>11} {'hmoe(cos)':>10}")
    for src, tgt in PAIRS:
        corpus_ids, query_ids, qrels = load_beir_meta(TEST)
        src_train = load_embeddings(src, TRAIN, "corpus")
        tgt_train = load_embeddings(tgt, TRAIN, "corpus")
        src_test = load_embeddings(src, TEST, "corpus")
        tgt_corpus = load_embeddings(tgt, TEST, "corpus")
        tgt_queries = load_embeddings(tgt, TEST, "query")

        ceiling = recall_at_k(tgt_corpus, tgt_queries, qrels, corpus_ids, query_ids, k=100)

        pro = procrustes_translate(src_train, tgt_train, src_test)
        r_pro = recall_at_k(pro, tgt_queries, qrels, corpus_ids, query_ids, k=100)

        cfg = hmoe_config(base_loss="cos", beta=0.0, beta_base=0.0,
                          retr_weight=0.0, base_epochs=80, lora_epochs=60)
        mapper, _ = train_hmoe(src_train, tgt_train, np.arange(src_train.shape[0]), cfg)
        r_h = recall_at_k(mapper.transform(src_test), tgt_queries, qrels, corpus_ids, query_ids, k=100)

        print(f"{src+'->'+tgt:22s} {ceiling:8.4f} {r_pro:11.4f} {r_h:10.4f}")


if __name__ == "__main__":
    main()
