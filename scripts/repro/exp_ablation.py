"""Ablation of the faithfulness fixes on FiQA->SciFact OOD Recall@100.

For OOD pairwise translation L_dir is off (beta=0), so the relevant fixes are
the tau-cascade router (Algorithm 2) and training experts at all 2K-1 nodes.
We compare, over a set of representative directed model pairs:

    baseline   : leaf-only experts, no tau backoff (tau=1.0)           [original src]
    +internal  : experts at all 2K-1 nodes, no tau backoff (tau=1.0)
    full       : all-node experts + tau-cascade routing (tau=0.8)      [both fixes]

Run:
    PYTHONPATH=src:.  python scripts/repro/exp_ablation.py --out output/repro/ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.repro.harness import hmoe_config, score_pairwise

# Representative directed pairs; all 11 models have FiQA corpus + SciFact corpus+query.
DEFAULT_PAIRS = [
    ("gemini", "openai"),
    ("e5", "kalm"),
    ("mistral", "nemotron"),
    ("qwen", "sfr"),
    ("gte", "linq"),
]

CONFIGS = {
    "baseline":   dict(tau=1.0, train_internal_experts=False),
    "+internal":  dict(tau=1.0, train_internal_experts=True),
    "full":       dict(tau=0.8, train_internal_experts=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/repro/ablation.json")
    ap.add_argument("--base-epochs", type=int, default=80)
    ap.add_argument("--lora-epochs", type=int, default=60)
    ap.add_argument("--pairs", default=None, help="comma list of src>tgt, overrides defaults")
    args = ap.parse_args()

    pairs = DEFAULT_PAIRS
    if args.pairs:
        pairs = [tuple(p.split(">")) for p in args.pairs.split(",")]

    rows = []
    for name, knobs in CONFIGS.items():
        for src, tgt in pairs:
            cfg = hmoe_config(
                num_levels=3, branch_factor=2, lora_rank=8, alpha=0.5, beta=0.0,
                base_epochs=args.base_epochs, lora_epochs=args.lora_epochs,
                **knobs,
            )
            res = score_pairwise(src, tgt, train_dataset="fiqa", test_dataset="scifact", cfg=cfg)
            rows.append({"config": name, "src": src, "tgt": tgt,
                         "recall_at_100": res["recall_at_100"]})
            print(f"{name:10s} {src:>9s}->{tgt:<9s}  R@100={res['recall_at_100']:.4f}")

    # Per-config mean
    summary = {}
    for name in CONFIGS:
        vals = [r["recall_at_100"] for r in rows if r["config"] == name]
        summary[name] = sum(vals) / len(vals)
    print("\n=== mean Recall@100 over", len(pairs), "pairs ===")
    for name in CONFIGS:
        print(f"  {name:10s} {summary[name]:.4f}")
    base = summary["baseline"]
    print(f"\n  +internal delta: {summary['+internal']-base:+.4f}")
    print(f"  full     delta: {summary['full']-base:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary,
                                          "pairs": pairs}, indent=2))


if __name__ == "__main__":
    main()
