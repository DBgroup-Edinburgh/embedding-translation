"""Linear (3-layer MLP) mapping strategy — ported from VectorTranslation.

A configurable neural mapper. Two training loss modes:
    - "mse": plain mean-squared error to target.
    - "cos": cosine similarity loss (default; matches VT's streaming loss).
    - "triplet": triplet loss using FAISS-mined neighbors.
    - "hierarchy": VT's VectorizedTripletRankingLoss.
    - "cos_triplet_hierarchy": all three combined, weighted by lambda_.

`fit()` accepts numpy arrays. For large reference sets (>100k rows) it
switches to an on-demand `_BatchLazyDataset` that fancy-indexes a whole batch
of source/target rows at once, so a memmap-backed reference set (e.g. full
5.4M-row Fever, ~80 GB/side) is streamed from disk one batch at a time instead
of being copied into RAM up-front — required to fit a memory-capped pod. Batch
(not per-row) reads + torch worker prefetch keep the GPU fed. The full-VT
MultiMemmapDatasetLoader is still not ported; this is the minimal lazy path
for the single-pair case.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from tqdm import trange


# Above this many reference rows, fit() streams via _BatchLazyDataset instead
# of materialising both sides in RAM (mirrors SimpleLinearMapper's threshold).
_LAZY_THRESHOLD = 100_000


class _BatchLazyDataset(Dataset):
    """On-demand (source, target) pairs read a WHOLE BATCH at a time.

    Used with a ``BatchSampler`` (so ``__getitem__`` receives a *list* of
    positions) and ``DataLoader(batch_size=None)``: each call gathers the batch
    rows from the (typically memmap) arrays in a single numpy fancy-index, then
    casts fp16→fp32. This is the key throughput fix — the previous per-sample
    dataset did ~batch_size separate Python ``__getitem__`` calls + per-row
    ``np.array``/``torch.from_numpy`` per batch, which dominated wall-clock on a
    Fever-scale memmap (same pathology the H-MoE node-loader fix addressed). One
    fancy-index per batch (C-level gather) + torch's worker prefetch keeps the
    GPU fed. fp16-on-disk stays half-size and page-cache resident.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, indices: np.ndarray):
        self.source = source
        self.target = target
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, batch_positions):
        # batch_positions is a list of positions into self.indices (BatchSampler).
        rows = self.indices[np.asarray(batch_positions)]
        src = np.ascontiguousarray(self.source[rows]).astype(np.float32, copy=False)
        tgt = np.ascontiguousarray(self.target[rows]).astype(np.float32, copy=False)
        return torch.from_numpy(src), torch.from_numpy(tgt)

from ...config import MappingConfig
from ...core.mapping import MappingStrategy
from ...loss import TripletLoss, VectorizedTripletRankingLoss, get_knn_faiss


def _cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = nn.functional.normalize(x, p=2, dim=1)
    y = nn.functional.normalize(y, p=2, dim=1)
    return (1 - torch.sum(x * y, dim=1)).mean()


class LinearModel(nn.Module):
    """Fixed 3-layer MLP: input → hidden → hidden → output, ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearMapper(MappingStrategy):
    """Linear (MLP) mapper. Configured via `MappingConfig.linear_config`."""

    def __init__(self, config: MappingConfig):
        super().__init__(config)
        self.lcfg = config.linear_config
        self.model: LinearModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    # MappingStrategy.fit() will call _fit then set is_fitted; we override fit
    # directly because we want lazy model construction (needs input/output dims).
    def fit(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        reference_indices: np.ndarray,
        query_emb_1: np.ndarray | None = None,
        query_emb_2: np.ndarray | None = None,
        **_: object,
    ) -> None:
        if len(reference_indices) == 0:
            raise ValueError("reference_indices cannot be empty")
        input_dim = source_embeddings.shape[1]
        output_dim = target_embeddings.shape[1]

        self.model = LinearModel(input_dim, self.lcfg.hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lcfg.learning_rate)

        # FAST PATH — GPU-resident data. The streaming loaders are bound by the
        # per-epoch CPU→GPU copy chain (cast+pin+H2D), which dominates wall-clock
        # because the model is small enough to leave the GPU underutilized. When
        # the whole fp16 reference pair fits in GPU memory (with headroom), upload
        # it ONCE and index on-device: the per-epoch transfer vanishes and
        # training becomes GPU-bound. Only the no-query cos/mse case (the
        # pairwise-OOD sweep) takes this path.
        if (
            self.device.type == "cuda"
            and query_emb_1 is None
            and self.lcfg.loss_type in {"cos", "mse", "cos_triplet_hierarchy"}
            and self._fit_gpu_resident(
                source_embeddings, target_embeddings, reference_indices,
                input_dim, output_dim,
            )
        ):
            return

        # For large reference sets, stream per-sample from (possibly memmap)
        # arrays so we never copy both full sides into RAM. The query-side
        # KNN-cache losses below need a stable batch order, which the lazy
        # path's shuffling doesn't provide, so only take it when no queries
        # were passed (the pairwise-OOD / cos-loss case).
        use_lazy = len(reference_indices) > _LAZY_THRESHOLD and query_emb_1 is None
        pin = self.device.type == "cuda"
        if use_lazy:
            import os

            n_workers = int(os.environ.get("LINEAR_LOADER_WORKERS", "8"))
            logger.info(
                f"LinearMapper: streaming {len(reference_indices):,} pairs via "
                f"batched lazy dataset (batch={self.lcfg.batch_size}, "
                f"workers={n_workers}; no full-array RAM copy)"
            )
            # BatchSampler hands a list of positions per draw; DataLoader with
            # batch_size=None treats each as one already-collated batch, so the
            # dataset does ONE fancy-index gather per batch (not one per row).
            # RandomSampler reshuffles every epoch.
            base = RandomSampler(range(len(reference_indices)))
            sampler = BatchSampler(base, batch_size=self.lcfg.batch_size, drop_last=False)
            loader = DataLoader(
                _BatchLazyDataset(source_embeddings, target_embeddings, reference_indices),
                batch_size=None,
                sampler=sampler,
                pin_memory=pin,
                num_workers=n_workers,
                persistent_workers=n_workers > 0,
                prefetch_factor=4 if n_workers > 0 else None,
            )
        else:
            # Keep reference tensors on CPU; move each batch to GPU on demand.
            # Loading the full reference set onto a 24 GB GPU OOMs at >~500k pairs.
            ref_src = torch.from_numpy(source_embeddings[reference_indices]).float()
            ref_tgt = torch.from_numpy(target_embeddings[reference_indices]).float()
            loader = DataLoader(
                TensorDataset(ref_src, ref_tgt),
                batch_size=self.lcfg.batch_size,
                shuffle=True,
                pin_memory=pin,
            )

        # Optional KNN cache for hierarchy/triplet losses
        knn_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        q1: torch.Tensor | None = None
        q2: torch.Tensor | None = None
        if query_emb_1 is not None and query_emb_2 is not None:
            q1 = torch.from_numpy(query_emb_1).float().to(self.device)
            q2 = torch.from_numpy(query_emb_2).float().to(self.device)
            for i, (_, batch_tgt) in enumerate(loader):
                knn_cache[i] = get_knn_faiss(q2, batch_tgt, k=self.lcfg.hierarchy_k)

        loss_type = self.lcfg.loss_type
        triplet = TripletLoss(margin=self.lcfg.triplet_margin)
        hierarchy = VectorizedTripletRankingLoss(
            margin=self.lcfg.rank_margin, weight_mode=self.lcfg.hierarchy_weight_mode
        )

        self.model.train()
        bar = trange(self.lcfg.num_epochs, desc="LinearMapper")
        last_loss = float("nan")
        for _ in bar:
            for batch_idx, (src_batch, tgt_batch) in enumerate(loader):
                src_batch = src_batch.to(self.device, non_blocking=True)
                tgt_batch = tgt_batch.to(self.device, non_blocking=True)
                pred = self.model(src_batch)
                self.optimizer.zero_grad()
                if loss_type == "mse":
                    loss = nn.functional.mse_loss(pred, tgt_batch)
                elif loss_type == "cos":
                    loss = _cosine_loss(pred, tgt_batch)
                elif loss_type == "cos_triplet_hierarchy" and q1 is not None:
                    score, index = knn_cache[batch_idx]
                    loss = (
                        _cosine_loss(pred, tgt_batch)
                        + self.lcfg.lambda_ * triplet(q2, pred, index, score)
                        + self.lcfg.lambda_ * hierarchy(q2, pred, index, score)
                    )
                elif loss_type in {"cos_triplet_hierarchy", "cos"}:
                    loss = _cosine_loss(pred, tgt_batch)
                elif loss_type == "triplet":
                    if q1 is None:
                        loss = _cosine_loss(pred, tgt_batch)
                    else:
                        score, index = knn_cache[batch_idx]
                        loss = triplet(q2, pred, index, score)
                elif loss_type == "hierarchy":
                    if q1 is None:
                        loss = _cosine_loss(pred, tgt_batch)
                    else:
                        score, index = knn_cache[batch_idx]
                        loss = hierarchy(q2, pred, index, score)
                else:
                    raise ValueError(f"Unknown loss_type {loss_type!r}")
                loss.backward()
                self.optimizer.step()
                last_loss = loss.item()
            bar.set_description(f"LinearMapper loss={last_loss:.4f}")

        self.metadata.update(
            {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": self.lcfg.hidden_dim,
                "loss_type": loss_type,
                "final_loss": last_loss,
            }
        )
        self.is_fitted = True
        logger.info(f"LinearMapper fit complete; final loss {last_loss:.6f}")

    def _fit_gpu_resident(
        self,
        source: np.ndarray,
        target: np.ndarray,
        reference_indices: np.ndarray,
        input_dim: int,
        output_dim: int,
    ) -> bool:
        """Train with both reference sides held fp16 on the GPU.

        Returns True if it ran (data fit in GPU memory), False to fall back to
        the streaming path. Data is uploaded ONCE in chunks (bounded CPU RAM);
        each epoch indexes the on-device tensors and casts the batch to fp32 on
        the GPU, so there is zero per-epoch host→device transfer.
        """
        import os

        N = len(reference_indices)
        need = N * (input_dim + output_dim) * 2  # fp16 bytes
        free, total = torch.cuda.mem_get_info()
        # Budget on TOTAL memory, not free: each pair is a fresh exp1 process so
        # the GPU is effectively empty at fit() time, but mem_get_info()'s `free`
        # dips a few GB below total (CUDA context + keepalive), and frac*free was
        # tripping 55-61GB pairs into the slow streaming path. frac*total is
        # stable; `free` is kept only as an OOM safety floor (won't admit a pair
        # that genuinely can't be allocated right now). 0.78 admits ≤66GB pairs
        # on an 85GB H100 (every non-big↔big pair, max 61GB) and 88GB big↔big on
        # a 143GB H200; excludes 85-88GB pairs on the H100 (they stream / go to
        # the H200). Working set (per-batch fp32 + grads) is a few GB on top.
        frac = float(os.environ.get("LINEAR_GPU_RESIDENT_FRAC", "0.78"))
        budget = frac * total
        cap = os.environ.get("LINEAR_GPU_RESIDENT_GB")
        if cap:
            budget = min(budget, float(cap) * 1e9)
        if need > budget or need > 0.92 * free:
            logger.info(
                f"LinearMapper: data {need/1e9:.0f}GB vs budget {budget/1e9:.0f}GB "
                f"(0.92*free {0.92*free/1e9:.0f}GB) → streaming path"
            )
            return False

        dev = self.device
        ref = np.asarray(reference_indices)
        logger.info(
            f"LinearMapper: GPU-resident fit — {need/1e9:.0f}GB fp16 on device "
            f"(free {free/1e9:.0f}GB), {N:,} pairs"
        )
        src_g = torch.empty((N, input_dim), dtype=torch.float16, device=dev)
        tgt_g = torch.empty((N, output_dim), dtype=torch.float16, device=dev)
        CH = 500_000
        for i in range(0, N, CH):
            j = min(i + CH, N)
            rows = ref[i:j]
            src_g[i:j] = torch.from_numpy(np.ascontiguousarray(source[rows])).to(dev)
            tgt_g[i:j] = torch.from_numpy(np.ascontiguousarray(target[rows])).to(dev)

        bs = self.lcfg.batch_size
        mse = self.lcfg.loss_type == "mse"
        self.model.train()
        bar = trange(self.lcfg.num_epochs, desc="LinearMapper(GPU-resident)")
        last = float("nan")
        for _ in bar:
            perm = torch.randperm(N, device=dev)
            last_t: torch.Tensor | None = None
            for k in range(0, N, bs):
                idx = perm[k : k + bs]
                src_batch = src_g[idx].float()
                tgt_batch = tgt_g[idx].float()
                pred = self.model(src_batch)
                self.optimizer.zero_grad()
                loss = (
                    nn.functional.mse_loss(pred, tgt_batch)
                    if mse
                    else _cosine_loss(pred, tgt_batch)
                )
                loss.backward()
                self.optimizer.step()
                last_t = loss.detach()  # avoid a per-step .item() GPU sync
            last = float(last_t.item()) if last_t is not None else float("nan")
            bar.set_description(f"LinearMapper(GPU-resident) loss={last:.4f}")

        del src_g, tgt_g
        torch.cuda.empty_cache()
        self.metadata.update(
            {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": self.lcfg.hidden_dim,
                "loss_type": self.lcfg.loss_type,
                "final_loss": last,
                "gpu_resident": True,
            }
        )
        self.is_fitted = True
        logger.info(f"LinearMapper GPU-resident fit complete; final loss {last:.6f}")
        return True

    def transform(self, embeddings: np.ndarray, **_: object) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("LinearMapper must be fit before transform")
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(embeddings).float().to(self.device)
            return self.model(x).cpu().numpy()
