"""Multi-Model Chaining (paper Section 5.4 / Fig 9b, Appendix H).

Two-hop pipeline SRC -> HUB -> TGT (nemotron -> openai -> kalm). Train the hop
translators f1 = SRC->HUB and f2 = HUB->TGT on FiQA, plus a direct SRC->TGT
reference; evaluate OOD on SciFact. The upstream hop uses the u1 channel and the
downstream hop the u2 channel (u2 _|_ u1).

drop = direct Recall@100 - chained Recall@100. The H-MoE translators use our config.

Run:
    PYTHONPATH=src:.  python reproduce-scripts/hmoe/exp_chaining.py --out output/repro/chaining.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))   # reproduce-scripts/ (harness)
from harness import (
    hmoe_config, load_beir_meta, load_embeddings, recall_at_k, train_hmoe,
)
from embedding_translation.mapper.hmoe import HMoEMapper

SRC, HUB, TGT = "nemotron", "openai", "kalm"
TRAIN_DS, TEST_DS = "fiqa", "scifact"


# H-MoE translator config (ours); setup per paper Appendix H
# (nemotron->openai->kalm, FiQA->SciFact). Each = (label, base_loss, beta, retr_weight):
# cos base loss + global-retrieval objective (retr_weight=6), L_dir off (beta=0).
VARIANTS = [
    ("H-MoE (cos+retr6)", "cos", 0.0, 6.0),
]


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/repro/chaining.json")
    ap.add_argument("--base-epochs", type=int, default=80)
    ap.add_argument("--lora-epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3, help="repeats per variant for mean+/-std")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    corpus_ids, query_ids, qrels = load_beir_meta(TEST_DS)
    src_train = load_embeddings(SRC, TRAIN_DS, "corpus")
    hub_train = load_embeddings(HUB, TRAIN_DS, "corpus")
    tgt_train = load_embeddings(TGT, TRAIN_DS, "corpus")
    ref = np.arange(src_train.shape[0])
    src_test = load_embeddings(SRC, TEST_DS, "corpus")
    tgt_queries = load_embeddings(TGT, TEST_DS, "query")
    tgt_corpus = load_embeddings(TGT, TEST_DS, "corpus")
    ceiling = recall_at_k(tgt_corpus, tgt_queries, qrels, corpus_ids, query_ids, k=100)

    rows = []
    for variant, base_loss, beta, retr_w in VARIANTS:
        direct_runs, chain_runs, drop_runs = [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg = hmoe_config(base_loss=base_loss, beta=beta, beta_base=beta,
                              retr_weight=retr_w,
                              base_epochs=args.base_epochs, lora_epochs=args.lora_epochs)
            f1, _ = train_hmoe(src_train, hub_train, ref, cfg)      # upstream, u1 channel
            f2 = HMoEMapper(cfg, dir_channel="u2")                  # downstream, u2 channel
            f2.fit(hub_train, tgt_train, ref)
            f_direct, _ = train_hmoe(src_train, tgt_train, ref, cfg)

            chained = f2.transform(f1.transform(src_test))
            direct = f_direct.transform(src_test)
            r_chain = recall_at_k(chained, tgt_queries, qrels, corpus_ids, query_ids, k=100)
            r_direct = recall_at_k(direct, tgt_queries, qrels, corpus_ids, query_ids, k=100)
            drop = (r_direct - r_chain) * 100.0
            print(f"  seed{seed} {variant:24s} direct={r_direct:.4f} chained={r_chain:.4f} drop={drop:.2f}%")
            direct_runs.append(r_direct); chain_runs.append(r_chain); drop_runs.append(drop)
        dm, dsd = float(np.mean(direct_runs)), float(np.std(direct_runs))
        cm, csd = float(np.mean(chain_runs)), float(np.std(chain_runs))
        pm, psd = float(np.mean(drop_runs)), float(np.std(drop_runs))
        print(f"{variant:24s} direct={dm:.4f}+/-{dsd:.4f} chained={cm:.4f}+/-{csd:.4f} "
              f"drop={pm:.2f}+/-{psd:.2f}%  (n={len(seeds)})")
        rows.append({"variant": variant, "base_loss": base_loss, "beta": beta, "retr_weight": retr_w,
                     "direct_mean": dm, "direct_std": dsd, "chained_mean": cm, "chained_std": csd,
                     "drop_mean": pm, "drop_std": psd,
                     "direct_runs": direct_runs, "chained_runs": chain_runs, "drop_runs": drop_runs})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"ceiling": ceiling, "chain": f"{SRC}->{HUB}->{TGT}",
                                          "seeds": len(seeds), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
