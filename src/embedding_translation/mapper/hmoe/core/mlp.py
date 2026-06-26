"""
Simplified Linear Mapper for Large-Scale Training
Optimized for memory efficiency and training speed with large datasets.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional, Any
from tqdm import tqdm
from tqdm.auto import trange

import faiss
import math
from collections import defaultdict
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod


# Internal ABC mirroring VectorTranslation's VectorMapper. Kept private to
# hmoe/ — the public hmoe surface (HMoEMapper) re-wraps these via our
# embedding_translation.core.mapping.MappingStrategy at the package boundary.
class VectorMapper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, source_embeddings, target_embeddings, reference_indices) -> None:
        ...

    @abstractmethod
    def transform(self, embeddings) -> Any:
        ...


if TYPE_CHECKING:
    MultiMemmapDatasetLoader = Any  # type: ignore[assignment]
else:
    MultiMemmapDatasetLoader = Any


# Stub wandb so any leftover .log() calls don't blow up when run without it.
class _WandbStub:
    def log(self, *args, **kwargs):  # noqa: D401
        return None

    def __getattr__(self, name):
        return self.log


wandb = _WandbStub()

# Disable logging to avoid multiprocessing issues
import logging
import os

# Completely disable logging to avoid multiprocessing issues
logging.disable(logging.CRITICAL)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress all logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# Set logging level to critical only
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger().disabled = True

class LazyEmbeddingDataset(Dataset):
    """Dataset that loads embeddings on-demand to save memory."""
    
    def __init__(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray, indices: np.ndarray, return_global_id: bool = False):
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        self.indices = indices
        self.return_global_id = return_global_id
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Only load the specific embedding when needed
        actual_idx = self.indices[idx]
        source_emb = self.source_embeddings[actual_idx]
        target_emb = self.target_embeddings[actual_idx]
        
        # Create writable copies to avoid PyTorch warnings
        source_emb_copy = source_emb.copy()
        target_emb_copy = target_emb.copy()
        
        if self.return_global_id:
            return torch.from_numpy(source_emb_copy).float(), torch.from_numpy(target_emb_copy).float(), torch.tensor(actual_idx, dtype=torch.long)
        else:
            return torch.from_numpy(source_emb_copy).float(), torch.from_numpy(target_emb_copy).float()


class SimpleLinearModel(nn.Module):
    """Multi-Layer Perceptron model for large-scale training."""
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 512, 
        layer_num: int = 2,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        super(SimpleLinearModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        
        # Build layers
        layers = []
        
        if layer_num == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Hidden layers
            for _ in range(layer_num - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout layer
        # self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply activation and dropout to all layers except the last one
            if i < len(self.layers) - 1:
                x = self.activation(x)
                # x = self.dropout(x)
        
        return x


class SimpleLinearMapper(VectorMapper):
    """
    MLP-based Mapper optimized for large-scale datasets.
    Uses Multi-Layer Perceptron with configurable hidden dimensions and layer numbers.
    Supports various activation functions and dropout for regularization.
    
    Note: This is an expert model, not a MoE mapper. It implements the standard
    VectorMapper interface and can be used as an expert in FlatMoEMapper.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        layer_num: int = 2,
        activation: str = 'relu',
        dropout: float = 0.1,
        device: torch.device = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        batch_size: int = 4028,  # Larger batch size for efficiency
        gradient_clip: float = 1.0,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-6,
        use_local_distill: bool = False,
        local_k: int = 50,
        local_tau: float = 0.1,
        local_weight: float = 0.5,
        faiss_use_float32: bool = True,
        knn_recompute_epochs: int = 0,
        global_weight: float = 0.5,
        # Structure-Preserving parameters (SPNT)
        use_structure_preserving: bool = False,
        struct_lambda: float = 0.1,
        struct_k: int = 10,
        struct_pair_sampling: str = 'knn',  # 'knn', 'random', or 'anchor'
        # Loss/output knobs (added in embedding_translation port for tuning)
        cosine_weight: float = 0.5,  # MSE + cosine_weight * cos. Set to inf-like to make cos dominate.
        mse_weight: float = 1.0,     # Set to 0 for pure cosine loss.
        normalize_output: bool = False,  # L2-normalize the model's output before loss + transform.
        loss_kind: str = "mse_cos",      # "l1" | "cos" | "mse" | "mse_cos" (paper Stage-1 base uses "l1")
    ):
        super(SimpleLinearMapper, self).__init__()
        
        # Auto-detect device if not provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        
        # Local distillation parameters
        self.use_local_distill = use_local_distill
        self.local_k = local_k
        self.local_tau = local_tau
        self.local_weight = local_weight
        self.faiss_use_float32 = faiss_use_float32
        self.knn_recompute_epochs = knn_recompute_epochs
        self.global_weight = global_weight
        # Runtime buffers for local distillation
        self._ref_idx_np = None          # Global reference indices (copy of reference_indices, np.int64)
        self._teacher_ref = None         # Teacher/target vectors on reference set (L2 normalized)
        self._faiss_index = None
        self._knn_idx = None             # Shape (N_ref, k) global ids
        self._knn_sim = None             # Shape (N_ref, k) teacher similarities
        self._student_cache = {}         # {global_id: torch.Tensor(d)} batch neighbor forward cache
        
        # Structure-Preserving parameters (SPNT)
        self.use_structure_preserving = use_structure_preserving
        self.struct_lambda = struct_lambda
        self.struct_k = struct_k
        self.struct_pair_sampling = struct_pair_sampling
        # Runtime buffers for structure preservation
        self._struct_knn_idx = None      # Shape (N_ref, struct_k) global ids for structure loss
        self._struct_knn_dist = None     # Shape (N_ref, struct_k) source space distances
        self._source_ref = None          # Source vectors on reference set for distance computation
        
        # Cached embeddings for KNN recomputation (only used in embeddings mode)
        self._cached_source_embeddings = None
        self._cached_target_embeddings = None
        self._cached_reference_indices = None

        # Initialize model
        self.model = SimpleLinearModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layer_num=layer_num,
            activation=activation,
            dropout=dropout
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=scheduler_patience,
            factor=scheduler_factor
        )
        # Use combined loss like in Diffusion model: MSE + weighted Cosine loss
        self.mse_criterion = nn.MSELoss()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.normalize_output = normalize_output
        self.loss_kind = loss_kind
        self.distill_temperature = 1.0  # Temperature for distillation
        
        print(f"SimpleLinearMapper (MLP) initialized on device: {device}")
        print(f"MLP Architecture: {input_dim} -> {hidden_dim} (x{layer_num-1}) -> {output_dim}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def _compute_combined_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss dispatcher.

        - loss_kind="l1":     ‖f(x)-y‖_1 (paper Stage-1 base translator)
        - loss_kind="mse":    pure MSE
        - loss_kind="cos":    pure cosine (1 - cos_sim mean)
        - loss_kind="mse_cos" (default): mse_weight*MSE + cosine_weight*Cosine
        If normalize_output=True the model outputs are L2-normalized first.
        """
        if getattr(self, "normalize_output", False):
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        kind = getattr(self, "loss_kind", "mse_cos")
        if kind == "l1":
            return torch.nn.functional.l1_loss(outputs, targets)
        if kind == "mse":
            return torch.nn.functional.mse_loss(outputs, targets)
        if kind == "cos":
            return 1 - torch.nn.functional.cosine_similarity(outputs, targets, dim=1).mean()
        if kind == "cos_retr":
            # cos + λ·InfoNCE with NATIVE negatives: anchor=target y_a, positive
            # f(x_a), negatives {f(x_b)} ∪ {y_b}, b≠a. Putting the true native
            # docs in the negative set trains the translator not to outrank them
            # in a merged corpus — the multi-model mixing failure mode. λ,τ via env.
            o = torch.nn.functional.normalize(outputs, p=2, dim=1)
            t = torch.nn.functional.normalize(targets, p=2, dim=1)
            cos_loss = 1 - (o * t).sum(dim=1).mean()
            B = t.shape[0]
            if B < 8:
                return cos_loss
            lam = float(os.environ.get("HMOE_RETR_LAMBDA", "1.0"))
            tau = float(os.environ.get("HMOE_RETR_TAU", "0.05"))
            cand = torch.cat([o, t], dim=0)                       # (2B, d)
            logits = (t @ cand.t()) / tau                         # (B, 2B)
            idx = torch.arange(B, device=t.device)
            logits[idx, B + idx] = float("-inf")                  # mask anchor's own native
            retr = torch.nn.functional.cross_entropy(logits, idx)
            return cos_loss + lam * retr
        # default: mse_cos hybrid
        cos_loss = 1 - torch.nn.functional.cosine_similarity(outputs, targets, dim=1).mean()
        if self.mse_weight == 0.0:
            return self.cosine_weight * cos_loss
        mse_loss = self.mse_criterion(outputs, targets)
        return self.mse_weight * mse_loss + self.cosine_weight * cos_loss
    
    def _compute_distillation_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss using KL divergence between student and teacher outputs.
        
        Args:
            student_outputs: Output logits from student model (B, d)
            teacher_outputs: Output logits from teacher model (B, d)
            
        Returns:
            KL divergence loss scaled by temperature^2
        """
        # Apply temperature scaling and softmax to get probability distributions
        student_log_probs = torch.log_softmax(student_outputs / self.distill_temperature, dim=1)
        teacher_probs = torch.softmax(teacher_outputs / self.distill_temperature, dim=1)
        
        # KL divergence: KL(teacher || student)
        kl_div = torch.nn.functional.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        )
        
        # Scale by temperature^2 to maintain gradient magnitude
        return (self.distill_temperature ** 2) * kl_div
    
    def _build_faiss_index(self, vecs_np: np.ndarray):
        """
        Build FAISS index for teacher space KNN search.
        
        Args:
            vecs_np: (N_ref, d) float32, already L2-normalized
            
        Returns:
            FAISS IndexFlatIP index
        """
        index = faiss.IndexFlatIP(vecs_np.shape[1])
        index.add(vecs_np)
        return index

    @torch.no_grad()
    def _precompute_teacher_knn(self, target_embeddings: np.ndarray, reference_indices: np.ndarray):
        """
        Precompute teacher space KNN using FAISS.
        
        Args:
            target_embeddings: Full target embedding matrix
            reference_indices: Indices of reference samples
        """
        # Extract teacher vectors for reference subset and L2 normalize
        ref = target_embeddings[reference_indices]
        ref = ref.astype(np.float32) if self.faiss_use_float32 else ref
        # L2 normalize
        norms = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
        ref = ref / norms

        self._ref_idx_np = np.asarray(reference_indices, dtype=np.int64)
        self._teacher_ref = ref  # (N_ref, d)
        self._faiss_index = self._build_faiss_index(ref)
        
        # Perform KNN search with reference set itself (includes self, top-1 is self, keeping it is fine)
        k = min(self.local_k, ref.shape[0])
        sim, idx_local = self._faiss_index.search(ref, k)
        
        # Map "local indices in reference subset" back to "global indices"
        self._knn_idx = self._ref_idx_np[idx_local]           # (N_ref, k) global ids
        self._knn_sim = sim                                   # (N_ref, k) teacher similarities (IP≈cos)
        
        # Clear student cache
        self._student_cache.clear()
        
        print(f"Precomputed teacher KNN: {len(reference_indices)} references, k={k}")
        print(f"KNN similarity range: [{self._knn_sim.min():.4f}, {self._knn_sim.max():.4f}]")

    @torch.no_grad()
    def _precompute_structure_knn(self, source_embeddings: np.ndarray, reference_indices: np.ndarray):
        """
        Precompute source space KNN for structure-preserving loss.
        
        This follows the SPNT paper: we need pairs (i,j) to compute
        L_Struct = (1/|P|) * Σ (||T(x_1^i) - T(x_1^j)|| - ||x_1^i - x_1^j||)^2
        
        Args:
            source_embeddings: Full source embedding matrix
            reference_indices: Indices of reference samples
        """
        # Extract source vectors for reference subset
        src_ref = source_embeddings[reference_indices]
        src_ref = src_ref.astype(np.float32) if self.faiss_use_float32 else src_ref
        
        self._source_ref = src_ref  # (N_ref, d)
        
        # Build FAISS index for source space (using L2 distance, not normalized)
        index = faiss.IndexFlatL2(src_ref.shape[1])
        index.add(src_ref)
        
        # Perform KNN search
        k = min(self.struct_k, src_ref.shape[0])
        dist, idx_local = index.search(src_ref, k)
        
        # Map "local indices in reference subset" back to "global indices"
        self._struct_knn_idx = self._ref_idx_np[idx_local]    # (N_ref, k) global ids
        self._struct_knn_dist = np.sqrt(dist)                  # (N_ref, k) L2 distances in source space
        
        print(f"Precomputed structure KNN: {len(reference_indices)} references, k={k}")
        print(f"KNN distance range: [{self._struct_knn_dist.min():.4f}, {self._struct_knn_dist.max():.4f}]")

    def _map_global_to_refpos(self, global_ids: np.ndarray) -> np.ndarray:
        """
        Map "global ids" to "row indices in reference array" for indexing _knn_idx/_knn_sim.
        Assumes reference_indices is a unique and fixed set of global ids.
        
        Args:
            global_ids: Array of global ids
            
        Returns:
            Array of positions in reference array
        """
        # For efficiency, build dict mapping during _precompute_teacher_knn; cache here
        if not hasattr(self, "_gid2pos"):
            self._gid2pos = {int(g): i for i, g in enumerate(self._ref_idx_np.tolist())}
        return np.asarray([self._gid2pos[int(g)] for g in global_ids], dtype=np.int64)

    def _local_distill_kl(self,
                          anchor_global_ids: np.ndarray,
                          batch_student: torch.Tensor,
                          source_embeddings: np.ndarray) -> torch.Tensor:
        """
        Compute local distance distillation KL loss (listwise KL).
        
        Args:
            anchor_global_ids: (B,) Global ids of anchors in this batch (= values in reference_indices)
            batch_student: (B, d) Student outputs for this batch anchors (already torch, may not be L2 normalized)
            source_embeddings: Used to get source vectors from "global ids" for neighbors, then forward to get student neighbor representations
            
        Returns:
            KL divergence loss tensor
        """
        device = batch_student.device
        B, d = batch_student.shape
        k = min(self.local_k, self._knn_idx.shape[1])

        # Get all neighbor global ids, do unique to reduce redundant forward passes
        nbr_ids = self._knn_idx[self._map_global_to_refpos(anchor_global_ids)][:, :k]  # (B, k) global ids
        uniq_nbr_ids = np.unique(nbr_ids.reshape(-1))
        
        # Get student neighbors from cache; compute forward pass for cache misses and store in cache
        to_compute = [gid for gid in uniq_nbr_ids if gid not in self._student_cache]
        if to_compute:
            # Batch forward pass (no_grad)
            src_batch = torch.from_numpy(source_embeddings[to_compute]).float().to(device)
            with torch.no_grad():
                out = self.model(src_batch)                           # (M, d)
                out = torch.nn.functional.normalize(out, dim=-1)
            for gid, vec in zip(to_compute, out):  # vec: (d,)
                self._student_cache[gid] = vec.detach()

        # Assemble neighbor tensors for this batch (B, k, d)
        S_anchor = torch.nn.functional.normalize(batch_student, dim=-1)   # (B, d)
        S_nb = torch.stack([torch.stack([self._student_cache[int(g)]
                                         for g in nbr_ids[i]], dim=0)
                            for i in range(B)], dim=0).to(device)          # (B, k, d)

        # Student similarity distribution (softmax over neighborhood)
        s_sim = torch.einsum("bd,bkd->bk", S_anchor, S_nb)                 # (B, k)
        s_logp = torch.log_softmax(s_sim / self.local_tau, dim=1)

        # Teacher target distribution (using precomputed similarities)
        t_sim = torch.from_numpy(
            self._knn_sim[self._map_global_to_refpos(anchor_global_ids)][:, :k]
        ).to(device)                                                       # (B, k)
        t_p = torch.softmax(t_sim / self.local_tau, dim=1)

        # KL(student || teacher) in teacher->student form: sum p_t * (log p_t - log p_s)
        # For training, commonly use KL(p_t || p_s), where p_t is constant, gradient only flows through s_logp
        kl = torch.sum(t_p * (torch.log(t_p + 1e-9) - s_logp), dim=1).mean()
        # Temperature scaling consistency (optional): multiply by tau^2
        return (self.local_tau ** 2) * kl
    
    def _compute_structure_preserving_loss(self,
                                           anchor_global_ids: np.ndarray,
                                           batch_outputs: torch.Tensor,
                                           batch_source: torch.Tensor) -> torch.Tensor:
        """
        Compute structure-preserving loss based on distance distortion.
        
        Following SPNT paper:
        L_Struct = (1/|P|) * Σ_{(i,j)∈P} (||T(x_1^i) - T(x_1^j)|| - ||x_1^i - x_1^j||)^2
        
        This penalizes the model when the distance between translated points differs
        from the distance in the source space, thus controlling the Lipschitz constant.
        
        Args:
            anchor_global_ids: (B,) Global ids of anchors in this batch
            batch_outputs: (B, d) Translated outputs for this batch (in target space)
            batch_source: (B, d_src) Source embeddings for this batch
            
        Returns:
            Structure-preserving loss tensor
        """
        device = batch_outputs.device
        B = batch_outputs.shape[0]
        k = min(self.struct_k, self._struct_knn_idx.shape[1])
        
        # Get reference positions for these anchors
        ref_pos = self._map_global_to_refpos(anchor_global_ids)  # (B,)
        
        # Get k-NN global ids and precomputed source distances
        nbr_gids = self._struct_knn_idx[ref_pos][:, :k]          # (B, k)
        src_dists = self._struct_knn_dist[ref_pos][:, :k]        # (B, k) - source space distances
        
        # Convert source distances to torch tensor
        src_dists_t = torch.from_numpy(src_dists).float().to(device)  # (B, k)
        
        # Get neighbor embeddings in target space
        # We need to translate the source embeddings of neighbors
        uniq_nbr_gids = np.unique(nbr_gids.reshape(-1))
        
        # Use the same cache mechanism as local distillation to avoid redundant forward passes
        nbr_outputs = {}
        to_compute = [int(gid) for gid in uniq_nbr_gids if int(gid) not in self._student_cache]
        
        if to_compute:
            # Get source embeddings for neighbors and translate them
            # Note: we need the original source_embeddings here
            # For now, we'll use the precomputed source_ref and map from global to local indices
            nbr_src_embs = []
            for gid in to_compute:
                # Find position in reference array
                pos = self._gid2pos[gid]
                nbr_src_embs.append(self._source_ref[pos])
            
            nbr_src_batch = torch.from_numpy(np.array(nbr_src_embs)).float().to(device)
            with torch.no_grad():
                nbr_out = self.model(nbr_src_batch)  # (M, d)
                for gid, vec in zip(to_compute, nbr_out):
                    nbr_outputs[gid] = vec.detach()
        
        # Get all neighbor outputs (from cache or newly computed)
        for gid in uniq_nbr_gids:
            gid_int = int(gid)
            if gid_int in self._student_cache:
                # Denormalize if it was normalized for local distillation
                nbr_outputs[gid_int] = self._student_cache[gid_int] * \
                    torch.norm(self._student_cache[gid_int]) if torch.norm(self._student_cache[gid_int]) < 0.99 else self._student_cache[gid_int]
        
        # Build neighbor tensor (B, k, d)
        try:
            nbr_out_tensor = torch.stack([
                torch.stack([nbr_outputs[int(gid)] for gid in nbr_gids[i]], dim=0)
                for i in range(B)
            ], dim=0).to(device)  # (B, k, d)
        except KeyError as e:
            # Fallback: if some neighbors are not found, recompute them
            print(f"KeyError in structure loss: {e}, falling back to direct computation")
            all_nbr_gids = nbr_gids.reshape(-1)
            all_nbr_pos = [self._gid2pos[int(gid)] for gid in all_nbr_gids]
            all_nbr_src = torch.from_numpy(self._source_ref[all_nbr_pos]).float().to(device)
            with torch.no_grad():
                all_nbr_out = self.model(all_nbr_src)
            nbr_out_tensor = all_nbr_out.view(B, k, -1)
        
        # Compute target space distances: ||T(x_i) - T(x_j)||
        # batch_outputs: (B, d), nbr_out_tensor: (B, k, d)
        tgt_dists = torch.norm(
            batch_outputs.unsqueeze(1) - nbr_out_tensor,  # (B, 1, d) - (B, k, d) = (B, k, d)
            dim=2
        )  # (B, k)
        
        # Compute distance distortion: (||T(x_i) - T(x_j)|| - ||x_i - x_j||)^2
        distortion = (tgt_dists - src_dists_t) ** 2  # (B, k)
        
        # Average over all pairs
        struct_loss = distortion.mean()
        
        return struct_loss
    
    def _create_dataloader(
        self, 
        source_emb: np.ndarray, 
        target_emb: np.ndarray,
        indices: np.ndarray,
        shuffle: bool = True,
        return_global_id: bool = False
    ) -> DataLoader:
        """Create DataLoader for training data."""
        
        # Convert to tensors (keep on CPU for DataLoader workers)
        source_tensor = torch.from_numpy(source_emb[indices]).float()
        target_tensor = torch.from_numpy(target_emb[indices]).float()
        
        if return_global_id:
            # Create global_id tensor
            global_id_tensor = torch.from_numpy(indices).long()
            # Create dataset with global_id
            dataset = TensorDataset(source_tensor, target_tensor, global_id_tensor)
        else:
            # Create dataset without global_id
            dataset = TensorDataset(source_tensor, target_tensor)
        
        # Create dataloader with efficient settings for large datasets
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 to avoid multiprocessing overhead and CUDA issues
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        
        return dataloader
    
    def _create_lazy_dataloader(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        indices: np.ndarray,
        shuffle: bool = True,
        return_global_id: bool = False
    ) -> DataLoader:
        """Create DataLoader using lazy loading to save memory."""
        
        # Create lazy dataset that loads embeddings on-demand
        dataset = LazyEmbeddingDataset(source_embeddings, target_embeddings, indices, return_global_id=return_global_id)
        
        # Create dataloader with memory-efficient settings.
        # num_workers>0: parallel per-sample reads + prefetch overlap disk I/O
        # with GPU compute (the single-threaded num_workers=0 starved the GPU
        # on memmap-backed full-Fever — ~8.7 min/epoch). On Linux fork the
        # memmap is COW-shared (no pickle/copy), and __getitem__ touches no
        # CUDA, so neither the "pickling large arrays" nor "CUDA" caveat
        # applies. Overridable via HMOE_LOADER_WORKERS (0 restores old path).
        _nw = int(os.environ.get("HMOE_LOADER_WORKERS", "8"))
        _kw = dict(num_workers=_nw, pin_memory=torch.cuda.is_available(), drop_last=False)
        if _nw > 0:
            _kw.update(prefetch_factor=4, persistent_workers=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, **_kw)

        return dataloader
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_source, batch_target in val_loader:
                # non_blocking enables async H2D DMA when the loader's
                # tensors are pinned (ArrayDatasetLoader sets pin_memory=True
                # when CUDA is available, see base_mapper.fit).
                batch_source = batch_source.to(self.device, non_blocking=True)
                batch_target = batch_target.to(self.device, non_blocking=True)

                outputs = self.model(batch_source)
                loss = self._compute_combined_loss(outputs, batch_target)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    
    def fit(
        self, 
        source_embeddings: Optional[np.ndarray] = None,
        target_embeddings: Optional[np.ndarray] = None,
        reference_indices: Optional[np.ndarray] = None,
        train_loader: Optional[Any] = None,
        validation_split: float = 0.0,
        global_model: Optional[SimpleLinearModel] = None,
        **kwargs
    ) -> None:
        """
        Train the linear mapping on reference data.
        
        Support two modes:
        1. Pass source_embeddings, target_embeddings, and reference_indices
        2. Pass train_loader directly
        
        Args:
            source_embeddings: Source embedding matrix (mode 1)
            target_embeddings: Target embedding matrix (mode 1)
            reference_indices: Indices to use for training (mode 1)
            train_loader: Pre-built dataloader (mode 2)
            validation_split: Fraction of reference data to use for validation
            global_model: Optional global model for distillation
        """
        self._log_training_info(source_embeddings, target_embeddings, reference_indices)
        
        # Step 1: Prepare dataloader
        train_loader = self._prepare_dataloader(
            source_embeddings, target_embeddings, reference_indices, train_loader
        )
        
        # Step 2: Setup preprocessing (KNN, etc.)
        self._setup_preprocessing(source_embeddings, target_embeddings, reference_indices)
        
        # Step 3: Train
        self._train(train_loader, global_model)
    
    def _prepare_dataloader(
        self,
        source_embeddings: Optional[np.ndarray],
        target_embeddings: Optional[np.ndarray],
        reference_indices: Optional[np.ndarray],
        train_loader: Optional[Any]
    ) -> Any:
        """Prepare dataloader from either embeddings or use provided loader."""
        # Validate input
        has_embeddings = all(x is not None for x in [source_embeddings, target_embeddings, reference_indices])
        has_loader = train_loader is not None
        
        if not (has_embeddings or has_loader):
            raise ValueError(
                "Must provide either (source_embeddings, target_embeddings, reference_indices) or train_loader"
            )
        if has_embeddings and has_loader:
            raise ValueError("Cannot provide both embeddings and train_loader. Choose one.")
        
        # Return provided loader
        if has_loader:
            return train_loader
        
        # Build loader from embeddings
        n_samples = len(reference_indices)
        need_global_id = self.use_local_distill or self.use_structure_preserving
        
        create_fn = self._create_lazy_dataloader if n_samples > 1e5 else self._create_dataloader
        return create_fn(
            source_embeddings, target_embeddings, reference_indices,
            shuffle=True, return_global_id=need_global_id
        )
    
    def _setup_preprocessing(
        self,
        source_embeddings: Optional[np.ndarray],
        target_embeddings: Optional[np.ndarray],
        reference_indices: Optional[np.ndarray]
    ) -> None:
        """Setup preprocessing like KNN computation (only in embeddings mode)."""
        # Cache embeddings for KNN recomputation
        self._cached_source_embeddings = source_embeddings
        self._cached_target_embeddings = target_embeddings
        self._cached_reference_indices = reference_indices
        
        # Skip if no embeddings provided
        if source_embeddings is None:
            print("Dataloader mode: Skipping KNN precomputation")
            return
        
        # Precompute teacher KNN for local distillation
        if self.use_local_distill:
            print("Precomputing teacher KNN for local distance distillation...")
            self._precompute_teacher_knn(target_embeddings, reference_indices)
        
        # Precompute source KNN for structure-preserving loss
        if self.use_structure_preserving:
            print("Precomputing source KNN for structure-preserving loss...")
            if self._ref_idx_np is None:
                self._ref_idx_np = np.asarray(reference_indices, dtype=np.int64)
            self._precompute_structure_knn(source_embeddings, reference_indices)
    
    def _log_training_info(
        self,
        source_embeddings: Optional[np.ndarray],
        target_embeddings: Optional[np.ndarray],
        reference_indices: Optional[np.ndarray]
    ) -> None:
        """Log training information."""
        print(f"Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
        if source_embeddings is not None and target_embeddings is not None and reference_indices is not None:
            print(f"Training with {len(reference_indices)} reference samples")
            print(f"Source shape: {source_embeddings.shape}, Target shape: {target_embeddings.shape}")
        else:
            print(f"Training with pre-built dataloader")
    
    def _train(
        self,
        train_loader: Any,
        global_model: Optional[SimpleLinearModel]
    ) -> None:
        """Execute training loop."""
        # Initialize model from global model if provided
        if global_model is not None:
            self.model.load_state_dict(global_model.model.state_dict())
        
        self.model.train()
        
        import time
        for epoch in tqdm[int](range(self.num_epochs), desc="Training epochs"):
            epoch_loss = self._train_one_epoch(train_loader, global_model, epoch)
            
            # Log progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
                wandb.log({"loss": epoch_loss, "epoch": epoch})
            
            # Recompute KNN if needed
            self._maybe_recompute_knn(epoch)
        
        print("Training completed!")
    
    def _train_one_epoch(
        self, 
        train_loader: Any, 
        global_model: Optional[SimpleLinearModel],
        epoch: int
    ) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for ib, batch_data in enumerate(train_loader):
            # Skip invalid batches
            if self._batch_has_nan(batch_data):
                print("Found nan or inf in batch data, skipping")
                continue
            
            # Extract and prepare batch
            batch_source, batch_target, batch_gid = self._extract_batch(batch_data)
            
            # Debug info for first batch
            if epoch == 0 and ib == 0:
                print(f"First batch - device: {batch_source.device}, shape: {batch_source.shape}")
            
            # Compute loss and update
            loss = self._compute_batch_loss(batch_source, batch_target, batch_gid, global_model)
            
            if np.isnan(loss.item()):
                print("Batch loss is nan, skipping")
                continue
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches if num_batches > 0 else float('inf')
    
    def _batch_has_nan(self, batch_data: tuple) -> bool:
        """Check if batch contains NaN or Inf values."""
        return (torch.isnan(batch_data[0]).any() or torch.isinf(batch_data[0]).any() or
                torch.isnan(batch_data[1]).any() or torch.isinf(batch_data[1]).any())
    
    def _extract_batch(self, batch_data: tuple) -> tuple:
        """Extract source, target, and optional global_id from batch."""
        need_global_id = self.use_local_distill or self.use_structure_preserving
        
        if need_global_id:
            batch_source, batch_target, batch_gid = batch_data
            batch_gid = batch_gid.cpu().numpy()
        else:
            batch_source, batch_target = batch_data
            batch_gid = None
        
        return (batch_source.to(self.device, non_blocking=True),
                batch_target.to(self.device, non_blocking=True),
                batch_gid)
    
    def _compute_batch_loss(
        self,
        batch_source: torch.Tensor,
        batch_target: torch.Tensor,
        batch_gid: Optional[np.ndarray],
        global_model: Optional[SimpleLinearModel]
    ) -> torch.Tensor:
        """Compute total loss for a batch."""
        # Forward pass
        outputs = self.model(batch_source)
        
        # Base loss
        loss = self._compute_combined_loss(outputs, batch_target)
        
        # Global model distillation
        if global_model is not None and self.global_weight > 0:
            global_model.model.eval()
            with torch.no_grad():
                global_outputs = global_model.model(batch_source)
            distill_loss = self._compute_distillation_loss(outputs, global_outputs.detach())
            loss = (1 - self.global_weight) * loss + self.global_weight * distill_loss
        
        # Local distance distillation
        if self.use_local_distill and batch_gid is not None and self._cached_source_embeddings is not None:
            local_kl = self._local_distill_kl(
                anchor_global_ids=batch_gid,
                batch_student=outputs,
                source_embeddings=self._cached_source_embeddings
            )
            loss = loss + self.local_weight * local_kl
        
        # Structure-preserving loss
        if self.use_structure_preserving and batch_gid is not None:
            struct_loss = self._compute_structure_preserving_loss(
                anchor_global_ids=batch_gid,
                batch_outputs=outputs,
                batch_source=batch_source
            )
            loss = loss + self.struct_lambda * struct_loss
        
        return loss
    
    def _maybe_recompute_knn(self, epoch: int) -> None:
        """Recompute KNN if conditions are met."""
        should_recompute = (
            self.use_local_distill and 
            self.knn_recompute_epochs > 0 and 
            (epoch + 1) % self.knn_recompute_epochs == 0 and 
            self._cached_target_embeddings is not None and 
            self._cached_reference_indices is not None
        )
        
        if should_recompute:
            print(f"Recomputing teacher KNN at epoch {epoch + 1}")
            self._precompute_teacher_knn(self._cached_target_embeddings, self._cached_reference_indices)
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using the trained linear mapping.
        Handles large datasets efficiently by processing in batches.
        """
        self.model.eval()

        # Process in batches to handle large datasets
        n_samples = embeddings.shape[0]
        batch_size = self.batch_size * 4  # Use larger batch size for inference
        
        results = []
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Transforming embeddings"):
                end_idx = min(i + batch_size, n_samples)
                batch = embeddings[i:end_idx]
                
                # Convert to tensor and move to device
                batch_tensor = torch.from_numpy(batch).float().to(self.device)
                
                # Forward pass
                output = self.model(batch_tensor)
                if getattr(self, "normalize_output", False):
                    output = torch.nn.functional.normalize(output, p=2, dim=1)

                # Move back to CPU and convert to numpy
                output_np = output.cpu().numpy()
                results.append(output_np)
        
        # Concatenate all results
        return np.concatenate(results, axis=0)
    
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {path}")
    
    def fit_with_loader(self, train_loader: Any) -> None:
        """
        Train the model with a dataloader.
        """
        self.model.train()
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
        
        for epoch in trange(self.num_epochs):
            for batch_idx, (batch_source, batch_target) in enumerate(train_loader):
                # Move to device. non_blocking enables async H2D DMA when
                # the loader's tensors are pinned (ArrayDatasetLoader with
                # pin_memory=True; harmless otherwise).
                batch_source = batch_source.to(self.device, non_blocking=True)
                batch_target = batch_target.to(self.device, non_blocking=True)

                if batch_source.dtype != torch.float32:
                    batch_source = batch_source.float()
                if batch_target.dtype != torch.float32:
                    batch_target = batch_target.float()
                
                # Check for NaN/Inf in input data
                assert not torch.isnan(batch_source).any() and not torch.isinf(batch_source).any(), "Batch source contains NaN or Inf"
                assert not torch.isnan(batch_target).any() and not torch.isinf(batch_target).any(), "Batch target contains NaN or Inf"
                
                # Zero gradients BEFORE forward pass (correct order)
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_source)
                
                # Compute loss
                loss = self._compute_combined_loss(outputs, batch_target)
                
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss (NaN/Inf) at epoch {epoch}, batch {batch_idx}")
                    continue
                
                # Check if loss has gradients before backward
                if not loss.requires_grad:
                    print(f"Warning: Loss does not require gradients! Loss value: {loss.item()}")
                    print(f"Outputs requires_grad: {outputs.requires_grad}")
                    print(f"Model parameters require_grad: {any(p.requires_grad for p in self.model.parameters())}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent gradient explosion
                # if self.gradient_clip > 0:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                # Update parameters
                self.optimizer.step()
                
                if batch_idx == 0 or batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                    wandb.log({
                        "loss": loss.item(), 
                        "epoch": epoch, 
                        "batch": batch_idx,
                        "batch_loss": loss.item()
                    })
        print("Training completed!")
    
    def fit_multi(self, multi_dataloader) -> None:
        """
        Train the model with MultiMemmapDatasetLoader (streaming interface).
        
        Simply delegates to fit_with_loader for memory-efficient training.
        
        Args:
            multi_dataloader: MultiMemmapDatasetLoader instance
        """
        self.fit_with_loader(multi_dataloader)
