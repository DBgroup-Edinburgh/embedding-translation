import numpy as np

from scripts.repro.harness import load_embeddings, recall_at_k


def test_recall_at_k_perfect():
    # each query is identical to its single gold doc -> recall@1 == 1.0
    corpus = np.eye(3, dtype=np.float32)
    queries = np.eye(3, dtype=np.float32)
    corpus_ids = ["d0", "d1", "d2"]
    query_ids = ["q0", "q1", "q2"]
    qrels = {"q0": {"d0": 1}, "q1": {"d1": 1}, "q2": {"d2": 1}}
    r = recall_at_k(corpus, queries, qrels, corpus_ids, query_ids, k=1)
    assert r == 1.0


def test_recall_at_k_misses_when_gold_absent_from_topk():
    # gold for q0 is d2 but query q0 points at d0 -> recall@1 == 0 for that query
    corpus = np.eye(3, dtype=np.float32)
    queries = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    qrels = {"q0": {"d2": 1}}
    r = recall_at_k(corpus, queries, qrels, ["d0", "d1", "d2"], ["q0"], k=1)
    assert r == 0.0


def test_load_embeddings_shapes():
    c = load_embeddings("openai", "scifact", "corpus")
    q = load_embeddings("openai", "scifact", "query")
    assert c.ndim == 2 and q.ndim == 2
    assert c.shape[1] == q.shape[1]
