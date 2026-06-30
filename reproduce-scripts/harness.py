"""Shared reproduction harness for ICML 2026 experiments.

Provides:
    - load_embeddings(model, dataset, kind) — read pre-computed vectorbench arrays.
    - load_beir_meta(dataset) — corpus_ids, query_ids, qrels.
    - recall_at_k(...) — BEIR-style cosine retrieval, ignores qrel labels <= 0.
    - train_hmoe(...) / train_linear(...) — canonical training closures.
    - score_pairwise(...) — translate src_test, eval R@100 vs tgt queries.
    - The same scripts can compose these into mixing / chaining experiments.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make the repo's `src/` importable so the H-MoE scripts can find the
# `embedding_translation` package without a manual PYTHONPATH. harness.py lives
# at <repo>/reproduce-scripts/harness.py, so the package is two levels up in src/.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

os.environ.setdefault("LOGURU_LEVEL", "WARNING")

VB_DIR = Path(os.environ.get("VB_DIR", "/root/github/embedding-translation/output/sweep/vb_embeddings"))
BEIR_DIR = Path(os.environ.get("BEIR_DIR", "/root/github/embedding-translation/output/sweep/beir_data"))


# ----------------------------- data ---------------------------------------


def _convert_fp32_in_place(src_path: Path, dst_path: Path, chunk: int = 200_000) -> None:
    """Cast a .npy on disk to float32 by streaming `chunk` rows at a time.

    Avoids pulling the whole source into RAM (openai's fp64 Fever upload is
    ~165 GB), and produces a proper-header .npy that can be memmapped.
    """
    src = np.load(src_path, mmap_mode="r")
    dst = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=np.float32, shape=src.shape
    )
    n = src.shape[0]
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        dst[i:j] = src[i:j].astype(np.float32)
    dst.flush()


def load_embeddings(
    model: str, dataset: str, kind: str = "corpus", *, mmap: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """Read the cached vectorbench .npy for (kind, model, dataset).

    Default mode reads the whole array, casts to float32, and L2-normalizes
    rows to unit vectors (paper Appendix B.1). Pass ``normalize=False`` for the
    raw vectors.

    With ``mmap=True`` the array is opened with ``mmap_mode='r'`` and
    returned without copying. For dtypes other than float32 (only openai —
    uploaded as fp64) a one-time `.fp32.npy` sidecar is written next to
    the source the first time it's loaded, so subsequent memmap reads hit
    the right dtype directly. Use mmap mode for the 5.4M-row Fever
    embeddings (88 GB fp32; openai 165 GB fp64) — otherwise loading two
    sides of a paired training run pulls ~170 GB into RAM unnecessarily.
    """
    p = VB_DIR / f"{kind}_{model}_{dataset}.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing embeddings: {p}")
    if mmap:
        arr = np.load(p, mmap_mode="r")
        if arr.dtype != np.float32:
            fp32_p = p.with_suffix(".fp32.npy")
            if not fp32_p.exists():
                _convert_fp32_in_place(p, fp32_p)
            arr = np.load(fp32_p, mmap_mode="r")
        return arr
    arr = np.load(p)
    # openai is uploaded as float64; cast everything to float32 for torch/faiss.
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if normalize:
        arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        arr = arr.astype(np.float32, copy=False)
    return arr


def nan_mask_chunked(arr: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Return a bool mask `keep` of length len(arr): True iff the row has no NaN.

    Scans `arr` in `chunk`-row blocks instead of materializing the whole
    array — important when `arr` is a memmap of a 5.4M × 4096 corpus
    (88 GB fp32) and an `~np.isnan(arr).any(axis=1)` would force-read the
    full file into RAM.
    """
    n = arr.shape[0]
    mask = np.empty(n, dtype=bool)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        mask[i:j] = ~np.isnan(arr[i:j]).any(axis=1)
    return mask


def load_beir_meta(dataset: str) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    """corpus_ids, query_ids, qrels for a BEIR dataset (test split)."""
    from beir.datasets.data_loader import GenericDataLoader
    data_path = BEIR_DIR / dataset
    if not data_path.exists():
        from beir import util as beir_util
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        BEIR_DIR.mkdir(parents=True, exist_ok=True)
        beir_util.download_and_unzip(url, str(BEIR_DIR))
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")
    return list(corpus.keys()), list(queries.keys()), qrels


# ----------------------------- eval ---------------------------------------


def recall_at_k(
    translated_corpus: np.ndarray,
    queries: np.ndarray,
    qrels: dict[str, dict[str, int]],
    corpus_ids: list[str],
    query_ids: list[str],
    k: int = 100,
) -> float:
    """BEIR-style mean Recall@k via cosine search (FAISS IndexFlatIP on unit vectors).

    For each query: fraction of its gold-truth (qrel score > 0) docs that
    appear in top-k results. Averaged across queries that have any gold.
    """
    import faiss
    c = translated_corpus.astype(np.float32, copy=False)
    q = queries.astype(np.float32, copy=False)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    index = faiss.IndexFlatIP(c.shape[1])
    index.add(c)
    _, I = index.search(q, k)
    id2idx = {cid: i for i, cid in enumerate(corpus_ids)}
    hits, n = 0.0, 0
    for q_pos, qid in enumerate(query_ids):
        gold_ids = [did for did, r in qrels.get(qid, {}).items() if r > 0 and did in id2idx]
        if not gold_ids:
            continue
        gold = {id2idx[did] for did in gold_ids}
        top = set(I[q_pos].tolist())
        hits += len(gold & top) / len(gold)
        n += 1
    return hits / max(n, 1)


def per_query_recall(
    translated_corpus: np.ndarray,
    queries: np.ndarray,
    qrels: dict[str, dict[str, int]],
    corpus_ids: list[str],
    query_ids: list[str],
    k: int = 100,
) -> np.ndarray:
    """Return per-query Recall@k (np.ndarray of length n_eval), in the order
    of `query_ids` (queries without gold are dropped)."""
    import faiss
    c = translated_corpus.astype(np.float32, copy=False)
    q = queries.astype(np.float32, copy=False)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    index = faiss.IndexFlatIP(c.shape[1])
    index.add(c)
    _, I = index.search(q, k)
    id2idx = {cid: i for i, cid in enumerate(corpus_ids)}
    vals: list[float] = []
    for q_pos, qid in enumerate(query_ids):
        gold_ids = [did for did, r in qrels.get(qid, {}).items() if r > 0 and did in id2idx]
        if not gold_ids:
            continue
        gold = {id2idx[did] for did in gold_ids}
        top = set(I[q_pos].tolist())
        vals.append(len(gold & top) / len(gold))
    return np.asarray(vals, dtype=np.float64)


# ----------------------------- training -----------------------------------


def hmoe_config(
    *,
    num_levels: int = 3,
    branch_factor: int = 2,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    alpha: float = 0.5,
    beta: float = 0.0,
    beta_base: float = 0.0,
    tau: float = 0.8,
    local_nn_m: int = 100,
    base_loss: str = "l1",
    dir_norm: str = "fixed",
    train_internal_experts: bool = True,
    local_anchors: int = 0,
    retr_weight: float = 0.0,
    retr_tau: float = 0.05,
    retr_pool_size: int = 2048,
    retr_hard_k: int = 0,
    base_epochs: int = 120,
    lora_epochs: int = 80,
    inner_hidden: int = 2048,
    inner_layers: int = 4,
    inner_epochs: int = 50,
    inner_activation: str = "selu",
    inner_loss: str = "mse_cos",
    inner_normalize_output: bool = False,
    learning_rate: float = 1e-4,
    batch_size: int = 1024,
):
    from embedding_translation.config import (
        GatingMoEConfig,
        MappingConfig,
        SimpleLinearMapperConfig,
    )
    # Bridge the legacy HMOE_LOCAL_ANCHORS env knob (set by the Fever sweep
    # orchestrator for LoRA-stage perf) into the config field. Core src reads
    # only the config; this script-level bridge keeps the Fever infra working.
    if not local_anchors:
        local_anchors = int(os.environ.get("HMOE_LOCAL_ANCHORS", "0"))
    inner = SimpleLinearMapperConfig(
        hidden_dim=inner_hidden,
        layer_num=inner_layers,
        activation=inner_activation,
        num_epochs=inner_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_local_distill=False,
        dropout=0.0,
        loss_kind=inner_loss,
        mse_weight=1.0,
        cosine_weight=0.5,
        normalize_output=inner_normalize_output,
    )
    return MappingConfig(hmoe_config=GatingMoEConfig(
        moe_type="hierarchical_lora",
        num_levels=num_levels, branch_factor=branch_factor,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        base_model_epochs=base_epochs, lora_epochs=lora_epochs,
        alpha=alpha, beta=beta, beta_base=beta_base,
        tau=tau, local_nn_m=local_nn_m,
        base_loss=base_loss,
        dir_norm=dir_norm,
        train_internal_experts=train_internal_experts,
        local_anchors=local_anchors,
        retr_weight=retr_weight, retr_tau=retr_tau, retr_pool_size=retr_pool_size,
        retr_hard_k=retr_hard_k,
        distance_metric="euclidean",
        mapper_config=inner,
    ))


def linear_config(
    *,
    hidden_dim: int = 2048,
    num_epochs: int = 80,
    batch_size: int = 1024,
    learning_rate: float = 5e-4,
    loss_type: str = "cos",
):
    from embedding_translation.config import LinearMapperConfig, MappingConfig
    return MappingConfig(linear_config=LinearMapperConfig(
        hidden_dim=hidden_dim, num_epochs=num_epochs, batch_size=batch_size,
        loss_type=loss_type, learning_rate=learning_rate,
    ))


def train_hmoe(src: np.ndarray, tgt: np.ndarray, ref_idx: np.ndarray, cfg=None):
    from embedding_translation.mapper.hmoe import HMoEMapper
    if cfg is None:
        cfg = hmoe_config()
    m = HMoEMapper(cfg)
    t0 = time.time()
    m.fit(src, tgt, ref_idx)
    return m, time.time() - t0


def train_linear(src: np.ndarray, tgt: np.ndarray, ref_idx: np.ndarray, cfg=None):
    from embedding_translation.mapper import LinearMapper
    if cfg is None:
        cfg = linear_config()
    m = LinearMapper(cfg)
    t0 = time.time()
    m.fit(src, tgt, ref_idx)
    return m, time.time() - t0


# ----------------------------- shared repro helpers -----------------------
# Canonical versions of helpers that were copy-pasted across exp2*/exp3*/exp4*
# scripts. New scripts should import these instead of redefining them.


def l2(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize to unit vectors (paper Appendix B.1)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32, copy=False)


def build_mixed_3way(native: np.ndarray, translated: dict, perm: np.ndarray) -> np.ndarray:
    """Mixing eval corpus: partition into 1 native part + one part per source
    in `translated` (dict src->translated_array), assembled by `perm`."""
    n, d = native.shape
    out = native.copy()
    part = n // (len(translated) + 1)
    for i, (_s, tr) in enumerate(translated.items(), 1):
        start = i * part
        end = n if i == len(translated) else (i + 1) * part
        out[perm[start:end]] = tr[perm[start:end]]
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, finite-safe, no scipy."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on average ranks), no scipy."""
    def _rank(a):
        order = a.argsort(); r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(a)); return r
    m = np.isfinite(x) & np.isfinite(y)
    return pearson(_rank(x[m]), _rank(y[m]))


def pca_dirs(Y: np.ndarray, k: int = 2, max_n: int = 50_000) -> list:
    """Top-k unit PCA directions (sign-fixed) of centered Y; numpy arrays."""
    Yc = Y[:max_n] - Y[:max_n].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    dirs = []
    for i in range(k):
        u = Vt[i].astype(np.float32); u /= (np.linalg.norm(u) + 1e-12)
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        dirs.append(u)
    return dirs


# ----------------------------- experiment scaffolds -----------------------


def score_pairwise(
    src_model: str,
    tgt_model: str,
    train_dataset: str = "fiqa",
    test_dataset: str = "scifact",
    method: str = "hmoe",          # "hmoe" | "linear"
    cfg=None,
    k: int = 100,
) -> dict:
    """Run one pairwise OOD translation experiment end-to-end."""
    src_train = load_embeddings(src_model, train_dataset, "corpus")
    tgt_train = load_embeddings(tgt_model, train_dataset, "corpus")
    src_test = load_embeddings(src_model, test_dataset, "corpus")
    tgt_queries = load_embeddings(tgt_model, test_dataset, "query")

    corpus_ids, query_ids, qrels = load_beir_meta(test_dataset)
    ref_idx = np.arange(src_train.shape[0])

    if method == "hmoe":
        mapper, t_fit = train_hmoe(src_train, tgt_train, ref_idx, cfg)
    elif method == "linear":
        mapper, t_fit = train_linear(src_train, tgt_train, ref_idx, cfg)
    else:
        raise ValueError(method)

    t0 = time.time()
    translated = mapper.transform(src_test)
    t_infer = time.time() - t0
    r = recall_at_k(translated, tgt_queries, qrels, corpus_ids, query_ids, k=k)
    return {
        "src": src_model, "tgt": tgt_model,
        "train": train_dataset, "test": test_dataset,
        "method": method,
        "recall_at_100": r,
        "train_s": t_fit, "infer_s": t_infer,
    }
