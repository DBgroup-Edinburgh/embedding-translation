"""OOD (FiQA->SciFact) hyperparameter sweep to maximize H-MoE Recall@100.

Honest-eval protocol: configs are scored on a VAL set of model pairs and the
single best config is then re-evaluated on a DISJOINT EVAL set of pairs. We
never pick a config by the numbers we ultimately report.

Grid is given as a list of hmoe_config kwargs dicts (--grid presets below).

Run:
    PYTHONPATH=src:.  python scripts/repro/exp_sweep.py --grid base_loss --out output/repro/sweep_baseloss.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.repro.harness import hmoe_config, score_pairwise

# Disjoint model-pair splits (all 11 models have FiQA corpus + SciFact corpus+query).
VAL_PAIRS = [("gemini", "openai"), ("e5", "kalm"), ("mistral", "nemotron")]
EVAL_PAIRS = [("qwen", "sfr"), ("gte", "linq"), ("openai", "gemini"), ("sfr", "e5")]

GRIDS = {
    # isolate the base-translator loss (l1 / cos / mse_cos)
    "base_loss": [
        {"base_loss": "l1"},
        {"base_loss": "cos"},
        {"base_loss": "mse_cos"},
    ],
    # output normalization of the inner MLP, stacked on the cos loss
    "normalize": [
        {"base_loss": "cos", "inner_normalize_output": False},
        {"base_loss": "cos", "inner_normalize_output": True},
    ],
    # capacity, stacked on the cos loss
    "capacity": [
        {"base_loss": "cos", "num_levels": 3},                 # K=4
        {"base_loss": "cos", "num_levels": 4},                 # K=8
        {"base_loss": "cos", "lora_rank": 16},
    ],
    # Approach B: retrieval-contrastive training (in-batch InfoNCE, cos_retr)
    # on the base translator and/or the LoRA experts.
    "retrieval": [
        {"base_loss": "cos"},                                   # cos regression baseline
        {"base_loss": "cos_retr"},                              # base via in-batch InfoNCE
        {"base_loss": "cos", "inner_loss": "cos_retr"},         # experts via InfoNCE
        {"base_loss": "cos_retr", "inner_loss": "cos_retr"},    # both
    ],
}


def _mean_recall(pairs, base_kwargs):
    vals = []
    for src, tgt in pairs:
        cfg = hmoe_config(**base_kwargs)
        r = score_pairwise(src, tgt, train_dataset="fiqa", test_dataset="scifact", cfg=cfg)
        vals.append(r["recall_at_100"])
    return float(np.mean(vals)), vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="base_loss", choices=list(GRIDS))
    ap.add_argument("--out", default="output/repro/sweep.json")
    ap.add_argument("--base-epochs", type=int, default=80)
    ap.add_argument("--lora-epochs", type=int, default=60)
    args = ap.parse_args()

    common = dict(base_epochs=args.base_epochs, lora_epochs=args.lora_epochs)
    results = []
    for override in GRIDS[args.grid]:
        kwargs = {**common, **override}
        val_mean, val_each = _mean_recall(VAL_PAIRS, kwargs)
        print(f"VAL {override}  mean={val_mean:.4f}  each={[round(v,3) for v in val_each]}")
        results.append({"override": override, "val_mean": val_mean, "val_each": val_each})

    best = max(results, key=lambda r: r["val_mean"])
    print(f"\nBEST on val: {best['override']}  val_mean={best['val_mean']:.4f}")
    eval_kwargs = {**common, **best["override"]}
    eval_mean, eval_each = _mean_recall(EVAL_PAIRS, eval_kwargs)
    print(f"EVAL (disjoint) mean={eval_mean:.4f}  each={[round(v,3) for v in eval_each]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "grid": args.grid, "val_pairs": VAL_PAIRS, "eval_pairs": EVAL_PAIRS,
        "results": results, "best": best,
        "eval_mean": eval_mean, "eval_each": eval_each,
    }, indent=2))


if __name__ == "__main__":
    main()
