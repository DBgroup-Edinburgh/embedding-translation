# Reproduction scripts

Self-contained scripts that reproduce the two papers backed by this repo:

- **`la2m/`** — *Integrating Vector Databases across Embedding Models* (SIGMOD'26):
  the LA2M cross-model **vector-database integration** task.
- **`hmoe/`** — *Generalizable and Composable Multi-Model Embedding Translation*
  (ICML'26): the H-MoE **embedding translation** task (pairwise OOD, mixing, chaining).

`harness.py` is shared by both: cached-embedding loading, BEIR metadata, cosine
Recall@k, and the H-MoE train/eval closures. Each script writes its raw metrics to a
JSON under `output/repro/`.

## Prerequisites

1. **Python 3.10+** with `numpy`, `scikit-learn`, `faiss-cpu`, `beir`, and — for the
   H-MoE scripts only — `torch` plus this repo's `embedding_translation` package
   (installed, or runnable from the repo's `src/`). The LA2M scripts need none of
   torch / `embedding_translation` (pure numpy / sklearn / faiss).
2. **Pre-computed embeddings** (see below) — set `VB_DIR` to the directory holding them.
3. **BEIR metadata** — set `BEIR_DIR` to a working directory for corpus ids / query ids
   / qrels; the datasets are auto-downloaded there from the public BEIR mirror on first
   use.
4. **Cap BLAS threads.** The Procrustes / k-means SVDs oversubscribe CPU otherwise
   (~20× slower). Always export
   `OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16`.

`harness.py` adds the repo's `src/` to `sys.path` automatically, so the H-MoE scripts
find `embedding_translation` without a manual `PYTHONPATH`.

### Embeddings (`VB_DIR`)

The scripts operate on **pre-computed embedding arrays, not raw text**. These are the
**VectorBenchmark** vectors, published as the HuggingFace dataset
[`DB-Edinburgh/VectorBenchmark`](https://huggingface.co/datasets/DB-Edinburgh/VectorBenchmark)
— download them and point `VB_DIR` at the directory. They are not vendored here (the
full set is hundreds of GB). Each file is a single 2-D `.npy` named:

```
{kind}_{model}_{dataset}.npy        # kind ∈ {corpus, query}
e.g.  corpus_kalm_fever.npy         (shape [n_docs,    d_model])
      query_openai_scidocs.npy      (shape [n_queries, d_model])
```

Row `i` is the embedding of the `i`-th BEIR document (or query) **in BEIR's native id
order** — this row↔id alignment is what `harness.load_embeddings` + `load_beir_meta`
rely on. Arrays are L2-normalized on load (paper Appendix B.1).

### What you need per experiment

A translation/integration cell `src → tgt` evaluated on a dataset needs three arrays:
`corpus_{src}_{train_ds}`, `corpus_{tgt}_{train_ds}` (to fit the translator) and, for
each OOD test dataset, `corpus_{src}_{test_ds}` + `query_{tgt}_{test_ds}` (the query
side is always the **target** model — search is "real target query into translated
corpus"). Models: the paper's 10 (kalm, nemotron, qwen, gemini, linq, e5, sfr, gritlm,
openai, mistral). Table 1 trains on Fever; LA2M integration uses a single dataset per
cell. All required arrays are in the VectorBenchmark dataset above.

## LA2M — `la2m/`

The integration task: partition a corpus into `O1 / O_cap / O2` (answers spread evenly
across `O1`,`O2`; `O_cap` is the answer-free reference overlap). `emb1` encodes
`O1∪O_cap`, `emb2` encodes `O2∪O_cap`; the overlap gives reference pairs. Learn a map
`T: emb1→emb2` from references only, build the integrated DB `T(D1)∪(D2\Y)`, query with
`emb2(q)`. Metrics: **R** = Recall@100 over all queries; **R1** = Recall@100[D1], only
queries whose answer is in the translated half (the metric that requires a good map).

| script | what it does | paper ref |
|---|---|---|
| `exp_la2m.py` | one (src,tgt,dataset) cell: Union / A2M (global isometry) / LA2M over an n-cluster sweep; `--centers 1,0` toggles the mean-centring ablation | Table 4 (+ Fig 5 n-sweep) |
| `exp_la2m_matrix.py` | 16-cell generality scan across 4 datasets, tuned-small-n, reports LA2M best-n R1 vs native ceiling | generality of the LA2M claim |

```bash
# common env prefix for every command below
export VB_DIR=/path/to/embeddings BEIR_DIR=/path/to/beir_work
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

# single cell + n-sweep (paper Table 4 proxy: Mistral->OpenAI, SciFact)
python reproduce-scripts/la2m/exp_la2m.py \
  --src mistral --tgt openai --dataset scifact --n-clusters 1,4,8,16,32 --seeds 3

# centring ablation (mean-centred vs un-centred isometry)
python reproduce-scripts/la2m/exp_la2m.py \
  --src mistral --tgt openai --dataset scifact --n-clusters 1,100 --centers 1,0 --seeds 3

# 16-cell generality matrix (resumable; writes output/repro/la2m_matrix.json)
python reproduce-scripts/la2m/exp_la2m_matrix.py
```

Each run prints, per (src, tgt, dataset) cell, the native target ceiling and Union /
A2M / LA2M Recall@100 (**R**) and Recall@100[D1] (**R1**), and saves them to JSON.
The cluster count `n` should be kept small relative to the reference-set size (each
cluster needs enough reference pairs to fit a stable isometry).

## H-MoE — `hmoe/`

Embedding translation `f_{A→B}` with a 4-layer SELU MLP base + per-node LoRA experts.
These scripts each load a canonical `hmoe_config` from `harness.py`.

| script | what it does | paper ref |
|---|---|---|
| `exp_table1_90.py` | Table 1 cross-data OOD: train on **Fever**, evaluate on OOD targets (SciDocs, ArguAna, FiQA-2018, NFCorpus, SciFact); 10×9 = 90 directed pairs; `METHOD=hmoe\|procrustes` (resumable, one method per run) | Table 1 (pairwise OOD) |
| `exp_mixing.py` | multi-model mixing (sources→shared target); reports absolute `mixed` R@100 **and** drop | §5.4 / Fig 9a |
| `exp_chaining.py` | two-hop `src→hub→tgt` vs direct; seed-averaged | §5.4 / Fig 9b |

```bash
# (same env prefix as above: VB_DIR / BEIR_DIR / thread caps)

# Table-1 cross-data OOD (Fever-trained -> OOD targets), one method per (resumable) invocation
METHOD=hmoe python reproduce-scripts/hmoe/exp_table1_90.py

# mixing / chaining
python reproduce-scripts/hmoe/exp_mixing.py   --out output/repro/mixing.json
python reproduce-scripts/hmoe/exp_chaining.py --out output/repro/chaining.json
```

Each script prints Recall@100 (mixing/chaining also report the drop vs the
direct/per-pair reference) and saves the raw numbers to its `--out` JSON.

## Notes

- Results are cached as JSON under `output/repro/`; the matrix and Table-1 scripts are
  **resumable** (already-computed cells are skipped).
- The numpy-heavy scripts are CPU-bound on the per-cluster / per-pair SVDs — the thread
  caps above are not optional for reasonable wall-clock.
- More exploratory / diagnostic scripts (Procrustes-residual mixing variants, ensembles,
  per-target diagnostics) remain under `scripts/repro/` and are not part of this curated
  reproduction set.
