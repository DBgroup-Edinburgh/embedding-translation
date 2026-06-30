"""FiQA->SciFact OOD pairwise translation driver (paper Section 5.3).

Trains a translator on FiQA corpus pairs (emb_src, emb_tgt) and evaluates it
OOD on SciFact: translate emb_src(SciFact corpus) into target space, then
measure Recall@100 of the real emb_tgt(SciFact queries) against the translated
corpus. One directed model pair per invocation.

Run:
    PYTHONPATH=src:.  python scripts/repro/exp_ood_fiqa_scifact.py \
        --src gemini --tgt openai --out output/repro/ood_gemini_openai.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.repro.harness import hmoe_config, linear_config, score_pairwise


def run_pair(src_model: str, tgt_model: str, *, method: str = "hmoe", cfg=None) -> dict:
    """Train on FiQA, evaluate OOD on SciFact; returns the score dict."""
    if cfg is None:
        cfg = (hmoe_config(base_loss="cos", retr_weight=6.0, beta=0.0, beta_base=0.0)
               if method == "hmoe" else linear_config())
    return score_pairwise(
        src_model, tgt_model,
        train_dataset="fiqa", test_dataset="scifact",
        method=method, cfg=cfg, k=100,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--method", default="hmoe", choices=["hmoe", "linear"])
    ap.add_argument("--out", default=None)
    # H-MoE structural knobs (OOD uses beta=0; method = cos base + global-retr=6)
    ap.add_argument("--k-leaves", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--base-epochs", type=int, default=120)
    ap.add_argument("--lora-epochs", type=int, default=80)
    args = ap.parse_args()

    cfg = None
    if args.method == "hmoe":
        # K = branch_factor ** (num_levels - 1); branch_factor=2 => num_levels chosen for k-leaves
        num_levels = {2: 2, 4: 3, 8: 4, 16: 5}.get(args.k_leaves, 3)
        cfg = hmoe_config(
            num_levels=num_levels, branch_factor=2,
            lora_rank=args.lora_rank, tau=args.tau, alpha=args.alpha, beta=args.beta,
            base_loss="cos", retr_weight=6.0, beta_base=args.beta,
            base_epochs=args.base_epochs, lora_epochs=args.lora_epochs,
        )
    res = run_pair(args.src, args.tgt, method=args.method, cfg=cfg)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
