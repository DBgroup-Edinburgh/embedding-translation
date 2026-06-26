"""Simple Linear MLP mapper — ported from VectorTranslation.

Differences vs `LinearMapper`:
    - Configurable number of layers (1..N) and activation (relu/gelu/tanh/leaky_relu).
    - AdamW + ReduceLROnPlateau scheduler.
    - Combined loss: MSE + weighted cosine (no triplet/hierarchy by default).
    - Optional local-distillation KL: align student-network neighborhood
      distributions to teacher (target) neighborhood distributions.

Streaming / memmap paths are not ported. fit() takes numpy arrays only.
"""

from __future__ import annotations

from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from ...config import MappingConfig
from ...core.mapping import MappingStrategy


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}


class SimpleLinearModel(nn.Module):
    """Configurable MLP: layer_num linears, single shared activation between."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        layer_num: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        if layer_num < 1:
            raise ValueError("layer_num must be >= 1")
        layers: list[nn.Module] = []
        if layer_num == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(layer_num - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        self.activation = _ACTIVATIONS[activation]()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class SimpleLinearMapper(MappingStrategy):
    """Configurable MLP mapper. Configured via `MappingConfig.simple_linear_config`."""

    def __init__(self, config: MappingConfig):
        super().__init__(config)
        self.scfg = config.simple_linear_config
        self.model: SimpleLinearModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        self._cosine_weight = 0.5

        # Local-distillation buffers
        self._ref_idx: np.ndarray | None = None
        self._teacher_knn_idx: np.ndarray | None = None
        self._teacher_knn_sim: np.ndarray | None = None
        self._gid2pos: dict[int, int] = {}

    def _combined_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        mse = nn.functional.mse_loss(pred, tgt)
        cos = 1 - nn.functional.cosine_similarity(pred, tgt, dim=1).mean()
        return mse + self._cosine_weight * cos

    @torch.no_grad()
    def _precompute_teacher_knn(
        self, target_embeddings: np.ndarray, reference_indices: np.ndarray
    ) -> None:
        ref = target_embeddings[reference_indices].astype(np.float32)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
        idx = faiss.IndexFlatIP(ref.shape[1])
        idx.add(ref)
        k = min(self.scfg.local_k, ref.shape[0])
        sim, idx_local = idx.search(ref, k)
        self._ref_idx = np.asarray(reference_indices, dtype=np.int64)
        self._teacher_knn_idx = self._ref_idx[idx_local]
        self._teacher_knn_sim = sim
        self._gid2pos = {int(g): i for i, g in enumerate(self._ref_idx.tolist())}
        logger.info(
            f"Teacher KNN precomputed: {len(reference_indices)} anchors, k={k}, "
            f"sim range [{sim.min():.3f}, {sim.max():.3f}]"
        )

    def _local_distill_kl(
        self,
        anchor_gids: np.ndarray,
        batch_student: torch.Tensor,
        source_embeddings: np.ndarray,
    ) -> torch.Tensor:
        assert self._teacher_knn_idx is not None and self._teacher_knn_sim is not None
        k = self._teacher_knn_idx.shape[1]
        pos = np.asarray([self._gid2pos[int(g)] for g in anchor_gids], dtype=np.int64)
        nbr_ids = self._teacher_knn_idx[pos]
        teacher_sim = torch.from_numpy(self._teacher_knn_sim[pos]).float().to(batch_student.device)

        # Forward neighbors through the current model
        flat_ids = nbr_ids.reshape(-1)
        nbr_src = torch.from_numpy(source_embeddings[flat_ids]).float().to(batch_student.device)
        nbr_out = self.model(nbr_src)  # type: ignore[union-attr]
        nbr_out = nn.functional.normalize(nbr_out, dim=-1).view(len(anchor_gids), k, -1)

        anchor_norm = nn.functional.normalize(batch_student, dim=-1)
        student_sim = torch.einsum("bd,bkd->bk", anchor_norm, nbr_out)

        tau = self.scfg.local_tau
        log_p_s = torch.log_softmax(student_sim / tau, dim=1)
        p_t = torch.softmax(teacher_sim / tau, dim=1)
        kl = (p_t * (torch.log(p_t + 1e-9) - log_p_s)).sum(dim=1).mean()
        return (tau ** 2) * kl

    def fit(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        reference_indices: np.ndarray,
        **_: object,
    ) -> None:
        if len(reference_indices) == 0:
            raise ValueError("reference_indices cannot be empty")

        in_dim = source_embeddings.shape[1]
        out_dim = target_embeddings.shape[1]
        self.model = SimpleLinearModel(
            input_dim=in_dim,
            output_dim=out_dim,
            hidden_dim=self.scfg.hidden_dim,
            layer_num=self.scfg.layer_num,
            activation=self.scfg.activation,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.scfg.learning_rate,
            weight_decay=self.scfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.scfg.scheduler_patience,
            factor=self.scfg.scheduler_factor,
        )

        if self.scfg.use_local_distill:
            self._precompute_teacher_knn(target_embeddings, reference_indices)

        src_t = torch.from_numpy(source_embeddings[reference_indices]).float()
        tgt_t = torch.from_numpy(target_embeddings[reference_indices]).float()
        gid_t = torch.from_numpy(np.asarray(reference_indices, dtype=np.int64))
        ds = (
            TensorDataset(src_t, tgt_t, gid_t)
            if self.scfg.use_local_distill
            else TensorDataset(src_t, tgt_t)
        )
        loader = DataLoader(ds, batch_size=self.scfg.batch_size, shuffle=True)

        self.model.train()
        bar = trange(self.scfg.num_epochs, desc="SimpleLinearMapper")
        last_loss = float("nan")
        for _ in bar:
            running = 0.0
            n = 0
            for batch in loader:
                if self.scfg.use_local_distill:
                    src_b, tgt_b, gid_b = batch
                    gid_np = gid_b.cpu().numpy()
                else:
                    src_b, tgt_b = batch
                    gid_np = None
                src_b = src_b.to(self.device)
                tgt_b = tgt_b.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(src_b)
                loss = self._combined_loss(pred, tgt_b)
                if self.scfg.use_local_distill and gid_np is not None:
                    loss = loss + self.scfg.local_weight * self._local_distill_kl(
                        gid_np, pred, source_embeddings
                    )
                loss.backward()
                if self.scfg.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.scfg.gradient_clip)
                self.optimizer.step()
                running += loss.item()
                n += 1
            avg = running / max(n, 1)
            self.scheduler.step(avg)
            last_loss = avg
            bar.set_description(f"SimpleLinearMapper loss={avg:.4f}")

        self.metadata.update(
            {
                "input_dim": in_dim,
                "output_dim": out_dim,
                "hidden_dim": self.scfg.hidden_dim,
                "layer_num": self.scfg.layer_num,
                "final_loss": last_loss,
            }
        )
        self.is_fitted = True
        logger.info(f"SimpleLinearMapper fit complete; final loss {last_loss:.6f}")

    def transform(self, embeddings: np.ndarray, **_: object) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("SimpleLinearMapper must be fit before transform")
        self.model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            batch_size = self.scfg.batch_size * 4
            for i in range(0, embeddings.shape[0], batch_size):
                chunk = embeddings[i : i + batch_size]
                x = torch.from_numpy(chunk).float().to(self.device)
                out.append(self.model(x).cpu().numpy())
        return np.concatenate(out, axis=0)
