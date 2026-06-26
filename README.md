<div align="center">

# 🧭 embedding-translation

### Generalizable & Composable Multi-Model Embedding Translation

*Learn a mapping `f : A → B` between two embedding spaces so that vectors from one
model can be searched, mixed, and chained in another model's space — without
re-embedding the corpus.*

[Quick Start](#-quick-start) ·
[Core Concepts](#-core-concepts) ·
[Reproduction](#-reproducing-the-paper) ·
[Repository Layout](#️-repository-layout) ·
[Citation](#-citation)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Config](https://img.shields.io/badge/config-pydantic%20v2-e92063)
![Venue](https://img.shields.io/badge/ICML-2026-8a2be2)
![Status](https://img.shields.io/badge/status-research-orange)

</div>

---

## 🎉 News

- **2026.04** — H-MoE was selected as an **ICML 2026 Spotlight** 🎉
- **2026.04** — LA2M received the **SIGMOD 2026 Best Paper Honorable Mention** 🎉

---

## 🔥 Highlights

- **One mapper, eight strategies.** Procrustes, CCA, linear / simple-linear MLPs,
  nonlinear, Gromov–Wasserstein, **LA2M**, and the headline **hierarchical
  Mixture-of-Experts (H-MoE)** translator — each in its own folder, selected by config.
- **H-MoE with LoRA experts.** A shared SELU backbone plus per-node low-rank
  adapters over an agglomerative-style tree, with cascade routing. Localized
  capacity at ~2× the parameters of a monolithic MLP.
- **Three deployment scenarios out of the box:** pairwise out-of-distribution
  translation, multi-model **mixing** into one shared search space, and
  multi-model **chaining** through a hub model.
- **Pre-translation confidence (TC).** A per-input reliability signal
  `TC(x) = exp(−δ(x)/σ)` for deciding when a translation is safe to trust.
- **Datasets & embeddings delegated to [VectorBenchmark](https://github.com/DBgroup-Edinburgh/VectorBenchmark)** —
  18 BEIR datasets, 70+ MTEB tasks, 14 embedding models, memmap-backed and cached.
- **Pydantic v2 everywhere.** Every hyperparameter is a validated config field;
  no dataclasses, no YAML round-tripping, no env-driven magic in the core.

---

## Introduction

Different embedding models map the same documents into different, incompatible
vector spaces. Re-embedding a large corpus every time you adopt a new model is
expensive; **embedding translation** instead learns a lightweight map
`f_{A→B} : ℝ^{d_A} → ℝ^{d_B}` from a paired training set so that
`f(emb_A(o)) ≈ emb_B(o)`, then applies it to a held-out corpus.

It is the reference implementation for the ICML 2026 paper
*"Generalizable and Composable Multi-Model Embedding Translation"*
(Beining Yang, Yang Cao).

---

## 🚀 Quick Start

### Install

```bash
git clone <your-fork-url> embedding-translation
cd embedding-translation
pip install -e .          # or: uv pip install -e .
```

> **Note** — `torch` is pinned to the CUDA 11.8 wheels (`torch>=2.4,<2.5`) for
> forward-compatibility with older drivers. Datasets and embeddings come from
> `vectorbench`, declared as a git dependency. Gated embeddings on the Hugging
> Face hub need `HF_TOKEN` in your environment.

### Translate in a few lines

```python
import numpy as np
from embedding_translation.config import GatingMoEConfig, MappingConfig, SimpleLinearMapperConfig
from embedding_translation.embedding import EmbeddingRequest, get_embedding
from embedding_translation.mapper.hmoe import HMoEMapper

# 1. Pre-computed embeddings of the same corpus from two models (via vectorbench)
src = get_embedding(EmbeddingRequest(dataset_name="fiqa", model_name="kalm",     type_="corpus"))
tgt = get_embedding(EmbeddingRequest(dataset_name="fiqa", model_name="nemotron", type_="corpus"))

# 2. Configure an H-MoE translator with the paper-canonical knobs
cfg = MappingConfig(hmoe_config=GatingMoEConfig(
    moe_type="hierarchical_lora",
    num_levels=3, branch_factor=2,          # K = 4 leaves (use 8 on Fever)
    lora_rank=8, alpha=0.5, beta=0.7, tau=0.8,
    mapper_config=SimpleLinearMapperConfig(activation="selu", layer_num=4),
))

# 3. Fit f: A → B on the training corpus, then translate
mapper = HMoEMapper(cfg)
mapper.fit(src, tgt, reference_indices=np.arange(len(src)))
translated = mapper.transform(src)          # now in nemotron-space

# 4. Evaluate: search the translated corpus with the *real* target-model queries
from embedding_translation.evaluation import get_retrieval_list
tgt_q = get_embedding(EmbeddingRequest(dataset_name="fiqa", model_name="nemotron", type_="query"))
top100 = get_retrieval_list(tgt_q, translated, top_k=100, metric="cosine")  # (n_queries, 100) doc indices
# Recall@100 = fraction of queries whose gold answer index appears in its top-100 row.
```

### Translation Confidence

`δ(x)` is the distance to the nearest training point; `TC(x) = exp(−δ(x)/σ)` is a
pre-translation reliability score in `(0, 1]`. Lower `δ` → safer to translate.

```python
from embedding_translation.evaluation import TranslationConfidence

tc = TranslationConfidence.fit(src)            # builds a FAISS index over the training pool
scores = tc.score(src)                         # global σ_data variant
scores = tc.score(src, mode="local")           # local-kNN variant (heterogeneous pools)
risky  = src[tc.score(src) < 0.3]              # flag low-confidence inputs to re-embed
```

### CLI

The `etrans` Typer app ships a minimal surface today (the dataset/embedding/map
command namespaces are owned by `vectorbench` and return in a later phase):

```bash
etrans info            # version + wired-up mappers / clustering
etrans list-mappers    # available mapping strategies
```

---

## 📖 Examples

This release ships the core `embedding_translation` library — see the
[Quick Start](#-quick-start) above for the end-to-end translate-and-evaluate flow
against any model pair. The paper-reproduction scripts (one per experiment) are
released separately.

---

## 🧠 Core Concepts

### Problem setup

Two encoders `emb_A : O → ℝ^{d_A}` and `emb_B : O → ℝ^{d_B}` over a universe of
objects `O`. Train `f_{A→B}` on paired vectors `{(emb_A(o), emb_B(o)) | o ∈ O_train}`;
evaluate on a disjoint `O_test`. All embeddings are L2-normalized; the headline
metric is **Recall@100** of a real query against the *translated* corpus.

### Three scenarios

1. **Pairwise OOD** — train one `A→B` translator, apply it to a different corpus.
2. **Mixing** — several `src_i → target` translators merge disjoint corpus subsets
   into one shared search space.
3. **Chaining** — compose `A → Hub` and `Hub → B` when no direct translator exists.

### H-MoE architecture

A frozen 4-layer SELU MLP base translator `f_θbase`, plus a **per-node LoRA
adapter** `f_i(x) = f_θbase(x) + B_i A_i x` (rank `r = 8`) over a binary tree with
`K` leaves / `2K−1` nodes. At inference, **top-down cascade routing** descends to
the most specific node whose centroid is unambiguously closest (threshold
`τ = 0.8`), so capacity stays localized and each expert's Lipschitz constant is small.

Training is three-stage: (1) global base alignment with an L1 loss, then freeze;
(2) hierarchical clustering to build the tree; (3) per-expert LoRA specialization
with `L_reg + α·L_local + β·L_dir` (`α = 0.5`, `β = 0.7`).

Pre-translation **Translation Confidence** (`TC(x) = exp(−δ(x)/σ)`) gives a
per-input reliability signal — see the [Quick Start](#translation-confidence) for usage.

---

## 📊 Reproducing the paper

The per-experiment reproduction scripts are released separately from this core
library. Embeddings are loaded memmap-backed, so large-corpus translation and
evaluation stay incremental and resumable.

---

## 🗂️ Repository Layout

```
embedding-translation/
├── src/embedding_translation/
│   ├── core/                 # base classes, registries, types
│   ├── embedding/            # thin adapter over vectorbench.embeddings
│   ├── dataset/              # thin adapter over vectorbench.dataset
│   ├── mapper/               # one subfolder per strategy:
│   │   ├── procrustes/  cca/  simple_linear/  linear/
│   │   ├── nonlinear/   gromov_wasserstein/   la2m/
│   │   └── hmoe/             # hierarchical MoE + LoRA experts (headline)
│   ├── loss/                 # triplet, margin/lambda-rank, rcsls, spearman, ...
│   ├── clustering/           # kmeans, avg-linkage, hypergraph (reference + gating)
│   ├── reference/            # build (source, target) training pairs
│   ├── evaluation/           # Recall@k / NDCG / MRR + Translation Confidence
│   ├── analysis/             # geometric diagnostics (research-only)
│   ├── pipeline/             # ZenML pipelines (local-only)
│   ├── config/               # pydantic schemas + pydantic-settings + YAML loader
│   └── cli/                  # `etrans` Typer app
└── pyproject.toml
```

**Conventions** (enforced):
- **Config is pydantic v2 only.** Every knob is a validated `BaseModel` field;
  every env-driven setting goes through `pydantic-settings`.
- **Datasets & embeddings delegate to `vectorbench`** — never re-implement BEIR/MTEB
  loaders or the 14 embedding generators. (Never import `vectorbench.embedding_dataset`.)
- **One mapper, one folder.** A new strategy is a new sibling under `mapper/`,
  registered in `mapper/__init__.py` — not edits sprinkled across shared files.

---

## 📄 Citation

```bibtex
@inproceedings{yang2026embeddingtranslation,
  title     = {Generalizable and Composable Multi-Model Embedding Translation},
  author    = {Yang, Beining and Cao, Yang},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}

@article{Yang2025integrating,
  author    = {Beining Yang and
               Yang Cao and
               Yang Ren},
  title     = {Integrating Vector Databases across Embedding Models},
  journal   = {Proc. {ACM} Manag. Data},
  volume    = {3},
  number    = {6},
  pages     = {1--28},
  year      = {2025},
  note      = {Presented at SIGMOD 2026}
}
```

---

## ⚖️ License

See [`LICENSE`](LICENSE).
</content>
</invoke>
