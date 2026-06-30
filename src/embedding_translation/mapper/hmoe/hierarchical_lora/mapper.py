"""
Hierarchical LoRA MoE Mapper implementation.

Three-step hierarchical construction with parameter-efficient LoRA:
1. MiniBatchKMeans to get leaf clusters
2. Bottom-up tree construction using KMeans on node centroids
3. Train shared base model + LoRA adapters for each leaf node

Key advantages:
- Parameter efficient: shared base model + lightweight LoRA adapters
- Memory friendly: only need base + one LoRA at a time during training/inference
- Hierarchical routing: cascade routing from root to leaf
"""

import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
from typing import Tuple, List, Optional, Dict

from ..base_mapper import BaseMoEMapper
from ..tree_structure import BottomUpHierarchyTree, TreeNode
from ..core.mlp import SimpleLinearModel, SimpleLinearMapper
from .lora_config import LoRAConfig
from .lora_expert import LoRAExpert


class HierarchicalLoRAMoEMapper(BaseMoEMapper):
    """
    Hierarchical Mixture of Experts with LoRA adapters.
    
    Multi-level tree-based expert system where all experts share a base model
    and use lightweight LoRA adapters for specialization.
    
    Construction process:
    1. MiniBatchKMeans streaming clustering to get leaf clusters
    2. Bottom-up tree construction using KMeans on node centroids
    3. Train shared base model on all data
    4. Train LoRA adapters for each leaf node using streaming data access
    
    Memory efficiency:
    - Shared base model (trained once, frozen)
    - Lightweight LoRA adapters (~0.1%-1% of base parameters)
    - Streaming training for both base and adapters
    """
    
    def __init__(
        self,
        num_levels: int = 3,
        branch_factor: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        share_base_model: bool = True,
        base_model_epochs: int = 200,
        lora_epochs: int = 100,
        lora_learning_rate: float = 1e-3,
        mapper_config = None,
        distance_metric: str = "cosine",
        transform_strategy: str = "cluster_then_route",
        # Paper-canonical knobs (ICML 2026; defaults match the paper)
        alpha: float = 0.5,                 # L_local weight (LoRA stage)
        beta: float = 0.0,                  # L_dir weight at LoRA stage
        beta_base: float = 0.0,             # L_dir weight at base stage (Stage 1).
                                            # Setting >0 directionally regularizes
                                            # the shared base before LoRA stage,
                                            # which matters for mixing/chaining.
        tau: float = 0.8,                   # routing ambiguity threshold
        local_nn_m: int = 100,              # NN count for L_local
        base_loss: str = "l1",              # Stage-1 base translator loss
        train_internal_experts: bool = True,  # train an expert at all 2K-1 nodes
                                            # (internal + leaf) so tau-cascade
                                            # backoff routes to a trained expert.
        u1: Optional[np.ndarray] = None,    # external PCA-1 direction of TARGET (mixing channel)
        u2: Optional[np.ndarray] = None,    # external PCA-2 direction of TARGET (chaining channel)
        # Which channel the L_dir penalty uses. "u1" (default, mixing channel)
        # vs "u2" (chaining channel for the second-hop translator in s→h→t).
        dir_channel: str = "u1",
        # L_dir normalization: "fixed" (raw off-axis energy / precomputed
        # target-variance scale, β monotone) vs "fraction" (‖orth‖²/‖e‖²).
        dir_norm: str = "fixed",
        # L_local anchor subsampling (0 = all rows, paper-faithful).
        local_anchors: int = 0,
        # H-MoE-specific mixing-aware retr (global native negatives); 0 disables.
        retr_weight: float = 0.0,
        retr_tau: float = 0.05,
        retr_pool_size: int = 2048,
        retr_hard_k: int = 0,
        **kwargs
    ):
        super().__init__()

        self.num_levels = num_levels
        self.branch_factor = branch_factor
        self.mapper_config = mapper_config
        self.distance_metric = distance_metric
        self.share_base_model = share_base_model

        # Calculate total number of leaf clusters
        self.num_leaf_clusters = branch_factor ** (num_levels - 1)

        # LoRA configuration
        self.lora_config = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )

        # Training hyperparameters
        self.base_model_epochs = base_model_epochs
        self.lora_epochs = lora_epochs
        self.lora_learning_rate = lora_learning_rate

        # Paper-canonical hyperparameters
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_base = float(beta_base)
        self.tau = float(tau)
        self.local_nn_m = int(local_nn_m)
        # L_local anchor subsampling: compute the local penalty on this many
        # random anchor rows per batch (0 = all, paper-faithful). The LoRA
        # stage is bottlenecked by L_local's O(B·k·d) gather/norm; subsampling
        # anchors cuts it ~linearly while preserving m (the neighbor count).
        self.local_anchors = int(local_anchors)
        self.base_loss = str(base_loss)
        self.train_internal_experts = bool(train_internal_experts)
        # PCA directions (R^{d_target}); set externally or auto-computed in Stage 2.5
        self._u1: Optional[torch.Tensor] = None if u1 is None else torch.from_numpy(np.asarray(u1, dtype=np.float32))
        self._u2: Optional[torch.Tensor] = None if u2 is None else torch.from_numpy(np.asarray(u2, dtype=np.float32))
        if dir_channel not in ("u1", "u2"):
            raise ValueError(f"dir_channel must be 'u1' or 'u2', got {dir_channel!r}")
        self.dir_channel = dir_channel
        if dir_norm not in ("fraction", "fixed"):
            raise ValueError(f"dir_norm must be 'fraction' or 'fixed', got {dir_norm!r}")
        self.dir_norm = dir_norm
        # Fixed L_dir scale σ² (mean target variance); set in _precompute_pca_directions.
        self._dir_scale: Optional[float] = None
        # H-MoE-specific mixing-aware retr loss. Each LoRA expert is trained with
        # an InfoNCE whose negatives are a GLOBAL pool of native target docs
        # sampled across ALL clusters — so an expert learns not to outrank native
        # docs from OTHER clusters (the multi-model mixing failure the generic
        # per-cluster cos+retr could not fix). retr_weight=0 disables.
        self.retr_weight = float(retr_weight)
        self.retr_tau = float(retr_tau)
        self.retr_pool_size = int(retr_pool_size)
        self.retr_hard_k = int(retr_hard_k)
        self._global_neg_pool: Optional[torch.Tensor] = None
        
        # Tree structure
        self.tree: Optional[BottomUpHierarchyTree] = None
        
        # Data storage - keep reference to train_loader for memmap access
        self.train_loader = None
        self.num_samples: Optional[int] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        
        # Leaf cluster mappings
        self.sample_to_leaf: Optional[np.ndarray] = None
        self.leaf_to_indices: Optional[List[np.ndarray]] = None
        
        # Shared base model and LoRA adapters
        self.base_model: Optional[nn.Module] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lora_adapters: Dict[int, LoRAExpert] = {}
        
        logger.info(
            f"Initialized HierarchicalLoRAMoEMapper: {self.num_levels} levels, "
            f"branch_factor={self.branch_factor}, "
            f"num_leaf_clusters={self.num_leaf_clusters}, "
            f"LoRA: rank={lora_rank}, alpha={lora_alpha}"
        )
    
    def _fit_from_loader(self, train_loader):
        """
        Train hierarchical LoRA MoE using four-step process.
        
        Steps:
        1. MiniBatchKMeans streaming clustering to get leaf clusters
        2. Bottom-up tree construction using KMeans on node centroids
        3. Train shared base model on all data
        4. Train LoRA adapters for each leaf node using streaming data access
        """
        logger.info(f"Training HierarchicalLoRAMoE ({self.num_levels} levels, "
                   f"branch_factor={self.branch_factor}, "
                   f"num_leaf_clusters={self.num_leaf_clusters})")
        
        # Step 0: Bind base dataset
        self._bind_base_dataset(train_loader)
        
        # Step 1: MiniBatchKMeans streaming clustering
        logger.info("=" * 70)
        logger.info("Step 1: MiniBatchKMeans streaming clustering...")
        logger.info("=" * 70)
        leaf_centroids = self._stream_cluster_bottom_layer(train_loader)
        logger.info(f"✓ Got {len(leaf_centroids)} leaf cluster centroids")
        
        # Step 2: Bottom-up tree construction
        logger.info("=" * 70)
        logger.info("Step 2: Bottom-up tree construction...")
        logger.info("=" * 70)
        self._build_hierarchy_bottom_up(leaf_centroids)
        logger.info(f"✓ Built hierarchy tree with {len(self.tree.nodes)} nodes")
        
        # Step 2.5: Precompute target-space PCA directions for L_dir.
        # Done up-front (before base training) so L_dir can be applied at the
        # base stage too when beta_base > 0, not just at the LoRA stage.
        if (self.beta > 0 or self.beta_base > 0) and self._u1 is None:
            logger.info("=" * 70)
            logger.info("Step 2.5: Precomputing target-space PCA directions u_1, u_2...")
            logger.info("=" * 70)
            self._precompute_pca_directions(train_loader)
            logger.info(f"✓ u_1, u_2 computed: dim={self._u1.shape[0]}")

        # Step 3: Train shared base model (with optional L_dir if beta_base>0)
        logger.info("=" * 70)
        logger.info("Step 3: Training shared base model on all data...")
        logger.info("=" * 70)
        self._train_base_model(train_loader)
        logger.info(f"✓ Trained and froze shared base model")
        
        # Step 4: Train LoRA adapters for leaf nodes
        logger.info("=" * 70)
        logger.info("Step 4: Training LoRA adapters for leaf nodes...")
        logger.info("=" * 70)
        self._train_lora_adapters_streaming()
        logger.info(f"✓ Trained {len(self.lora_adapters)} LoRA adapters")
        
        logger.info("=" * 70)
        logger.info("✓ HierarchicalLoRAMoE training completed")
        logger.info("=" * 70)
    
    def _bind_base_dataset(self, train_loader):
        """
        Bind train_loader and extract dimensions.
        
        Uses MultiMemmapDatasetLoader's datasets for direct memmap access.
        """
        self.train_loader = train_loader
        self.input_dim = train_loader.source_embedding_dim
        self.output_dim = train_loader.target_embedding_dim
        
        # Get total samples
        if hasattr(train_loader, 'total_samples'):
            self.num_samples = train_loader.total_samples
        else:
            logger.warning("Counting samples by iterating (slow)...")
            self.num_samples = 0
            for batch in train_loader:
                src_batch = batch[0] if isinstance(batch, tuple) else batch
                if isinstance(src_batch, torch.Tensor):
                    self.num_samples += src_batch.shape[0]
                else:
                    self.num_samples += len(src_batch)
        
        logger.info(
            f"Bound train_loader: {self.num_samples:,} samples, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        )
    
    def _stream_cluster_bottom_layer(
        self,
        train_loader,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Step 1: Streaming clustering with MiniBatchKMeans.
        
        Returns:
            leaf_centroids: (K, d_in) array of leaf cluster centers
        """
        # Pass 1: Streaming clustering (only learn cluster centers)
        logger.info(f"Pass 1: Streaming clustering with {self.num_leaf_clusters} clusters...")
        mbk = MiniBatchKMeans(
            n_clusters=self.num_leaf_clusters,
            batch_size=batch_size,
            random_state=42,
            n_init=3,
            verbose=0
        )
        
        for src_batch, tgt_batch in tqdm(train_loader, desc="Clustering"):
            if isinstance(src_batch, torch.Tensor):
                src_batch = src_batch.cpu().numpy()
            mbk.partial_fit(src_batch)
        
        leaf_centroids = mbk.cluster_centers_
        logger.info(f"Learned {len(leaf_centroids)} leaf cluster centers")
        
        # Pass 2: Assign each sample to a leaf cluster
        logger.info("Pass 2: Assigning samples to leaf clusters...")
        N = self.num_samples
        sample_to_leaf = np.empty(N, dtype=np.int32)
        
        global_idx = 0
        for src_batch, tgt_batch in tqdm(train_loader, desc="Assigning"):
            if isinstance(src_batch, torch.Tensor):
                src_batch = src_batch.cpu().numpy()
            
            labels = mbk.predict(src_batch)
            B = len(labels)
            if global_idx + B > N:
                actual_B = N - global_idx
                sample_to_leaf[global_idx : global_idx + actual_B] = labels[:actual_B]
                global_idx = N
                break
            else:
                sample_to_leaf[global_idx : global_idx + B] = labels
                global_idx += B
        
        if global_idx != N:
            logger.warning(
                f"Sample count mismatch: processed {global_idx}, expected {N}. "
                f"Truncating sample_to_leaf array."
            )
            sample_to_leaf = sample_to_leaf[:global_idx]
            self.num_samples = global_idx
        
        # Reverse mapping: group samples by leaf cluster
        leaf_to_indices: List[List[int]] = [[] for _ in range(self.num_leaf_clusters)]
        for gid, leaf_id in enumerate(sample_to_leaf):
            leaf_to_indices[leaf_id].append(gid)
        
        # Convert to numpy arrays
        self.leaf_to_indices = [
            np.array(lst, dtype=np.int64) for lst in leaf_to_indices
        ]
        self.sample_to_leaf = sample_to_leaf
        
        # Log cluster sizes
        cluster_sizes = [len(indices) for indices in self.leaf_to_indices]
        logger.info(f"Leaf cluster sizes: min={min(cluster_sizes)}, "
                   f"max={max(cluster_sizes)}, "
                   f"mean={np.mean(cluster_sizes):.1f}")
        
        return leaf_centroids
    
    def _build_hierarchy_bottom_up(self, leaf_centroids: np.ndarray):
        """
        Step 2: Build hierarchy tree bottom-up using KMeans on node centroids.
        
        Args:
            leaf_centroids: (K, d_in) array of leaf cluster centers
        """
        # Initialize tree
        self.tree = BottomUpHierarchyTree(self.num_levels, self.branch_factor)
        
        # Step 2.1: Create leaf nodes
        logger.info("Creating leaf nodes...")
        leaf_level = self.num_levels - 1
        K = leaf_centroids.shape[0]
        
        for leaf_id in range(K):
            node_id = len(self.tree.nodes)
            node = TreeNode(
                node_id=node_id,
                level=leaf_level,
                centroid=leaf_centroids[leaf_id].copy()
            )
            node.data_indices = self.leaf_to_indices[leaf_id].copy()
            self.tree.nodes.append(node)
            self.tree.level_nodes[leaf_level].append(node_id)
        
        logger.info(f"Created {K} leaf nodes")
        
        # Step 2.2: Build tree bottom-up
        current_level_ids = self.tree.level_nodes[leaf_level].copy()
        
        for level in range(self.num_levels - 2, -1, -1):
            logger.info(f"Building level {level} from {len(current_level_ids)} children...")
            
            num_children = len(current_level_ids)
            num_parents = max(1, num_children // self.branch_factor)
            
            # Get centroids of current level nodes
            node_centroids = np.stack(
                [self.tree.nodes[nid].centroid for nid in current_level_ids],
                axis=0
            )
            
            # Cluster children into parents
            if num_parents == 1:
                labels = np.zeros(num_children, dtype=np.int32)
                parent_centroids = [node_centroids.mean(axis=0)]
            else:
                kmeans = KMeans(
                    n_clusters=num_parents,
                    random_state=42,
                    n_init=10,
                    verbose=0
                )
                labels = kmeans.fit_predict(node_centroids)
                parent_centroids = kmeans.cluster_centers_
            
            # Create parent nodes
            parent_ids = []
            for p in range(num_parents):
                parent_node_id = len(self.tree.nodes)
                parent = TreeNode(
                    node_id=parent_node_id,
                    level=level,
                    centroid=parent_centroids[p].copy()
                )
                
                # Collect children belonging to this parent
                child_ids = [
                    current_level_ids[i]
                    for i, lab in enumerate(labels)
                    if lab == p
                ]
                
                # Collect all data indices from children
                all_indices = []
                for cid in child_ids:
                    child = self.tree.nodes[cid]
                    child.parent_id = parent_node_id
                    parent.child_ids.append(cid)
                    if child.data_indices is not None and len(child.data_indices) > 0:
                        all_indices.append(child.data_indices)
                
                # Concatenate all child data indices
                if len(all_indices) > 0:
                    parent.data_indices = np.concatenate(all_indices)
                else:
                    parent.data_indices = np.zeros(0, dtype=np.int64)
                
                self.tree.nodes.append(parent)
                parent_ids.append(parent_node_id)
            
            self.tree.level_nodes[level] = parent_ids
            current_level_ids = parent_ids
            
            logger.info(f"Created {len(parent_ids)} parent nodes at level {level}")
        
        # Set root
        if len(self.tree.level_nodes[0]) > 0:
            self.tree.root_id = self.tree.level_nodes[0][0]
            logger.info(f"Root node ID: {self.tree.root_id}")
        else:
            raise ValueError("Failed to create root node")
        
        # Log tree statistics
        self._log_tree_statistics()
    
    def _log_tree_statistics(self):
        """Log statistics about the constructed tree."""
        logger.info("Tree statistics:")
        for level in range(self.num_levels):
            nodes = self.tree.get_nodes_at_level(level)
            total_samples = sum(
                len(node.data_indices) if node.data_indices is not None else 0
                for node in nodes
            )
            logger.info(
                f"  Level {level}: {len(nodes)} nodes, "
                f"{total_samples:,} total samples"
            )
    
    def _train_base_model(self, train_loader):
        """
        Step 3: Train shared base model on all data.

        Paper's Algorithm 1 line 2 uses an L1 regression loss. If beta_base > 0
        we additionally apply the L_dir directional penalty (u_1 by default)
        during base training so the base translator's residuals are already
        directionally regularized before LoRA fine-tuning runs.
        """
        logger.info("Training shared base model on all training data...")
        logger.info(f"  Input dim: {self.input_dim}, Output dim: {self.output_dim}")
        logger.info(f"  Epochs: {self.base_model_epochs}  beta_base={self.beta_base}")

        # Build the SimpleLinearMapper to inherit its model architecture +
        # optimizer scaffolding, then run our own training loop so we can
        # inject the L_dir penalty.
        base_config = self.mapper_config.model_dump()
        base_config['num_epochs'] = self.base_model_epochs
        base_config['loss_kind'] = self.base_loss
        base_trainer = SimpleLinearMapper(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            **base_config,
        )

        if self.beta_base == 0.0:
            # Fast path: plain SimpleLinearMapper training (paper's default).
            base_trainer.fit_with_loader(train_loader)
        else:
            # Custom training loop with L1 (or configured base_loss) + β_base · L_dir.
            self._fit_base_with_dir(base_trainer, train_loader)

        # Extract, freeze, place on device, log
        self.base_model = base_trainer.model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.to(self.device)
        self.base_model.eval()
        total_params = sum(p.numel() for p in self.base_model.parameters())
        logger.info(f"✓ Base model trained and frozen: {total_params:,} parameters")

    def _fit_base_with_dir(self, base_trainer, train_loader) -> None:
        """Stage-1 training with L1 (or chosen base_loss) + β_base · L_dir.

        We inherit base_trainer's model + optimizer + LR scheduler so all
        SimpleLinearMapper hyperparameters (lr, weight_decay, grad clip,
        scheduler) still apply.
        """
        from tqdm.auto import trange
        model = base_trainer.model
        opt = base_trainer.optimizer
        sched = getattr(base_trainer, "scheduler", None)
        device = self.device

        # Tag what the base loss should look like.
        kind = self.base_loss
        beta_b = self.beta_base
        u_channel = self.dir_channel
        u = (self._u1 if u_channel == "u1" else self._u2)
        if u is None:
            logger.warning(
                f"_fit_base_with_dir: u_{u_channel[-1]} is None; falling back to plain base loss."
            )
            base_trainer.fit_with_loader(train_loader)
            return
        u = u.to(device)

        model.train()
        bar = trange(self.base_model_epochs, desc=f"base+L_dir (β={beta_b})")
        last_loss = float("nan")
        for epoch in bar:
            total = 0.0
            n = 0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    src_b, tgt_b = batch[0], batch[1]
                else:
                    continue
                if not isinstance(src_b, torch.Tensor):
                    src_b = torch.from_numpy(src_b)
                if not isinstance(tgt_b, torch.Tensor):
                    tgt_b = torch.from_numpy(tgt_b)
                src_b = src_b.float().to(device, non_blocking=True)
                tgt_b = tgt_b.float().to(device, non_blocking=True)
                opt.zero_grad()
                pred = model(src_b)
                # Primary regression loss
                if kind == "l1":
                    loss = nn.functional.l1_loss(pred, tgt_b)
                elif kind == "mse":
                    loss = nn.functional.mse_loss(pred, tgt_b)
                elif kind == "cos":
                    loss = 1 - nn.functional.cosine_similarity(pred, tgt_b, dim=1).mean()
                else:  # mse_cos hybrid
                    loss = (
                        nn.functional.mse_loss(pred, tgt_b)
                        + 0.5 * (1 - nn.functional.cosine_similarity(pred, tgt_b, dim=1).mean())
                    )
                # L_dir off-axis penalty (dir_norm="fraction" legacy / "fixed"
                # raw-energy form); shared with the LoRA-stage _l_dir.
                l_dir = self._off_axis_penalty(pred - tgt_b, u)
                loss = loss + beta_b * l_dir
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            avg = total / max(n, 1)
            if sched is not None:
                try:
                    sched.step(avg)
                except TypeError:
                    sched.step()
            bar.set_description(f"base+L_dir loss={avg:.4f}")
            last_loss = avg
        logger.info(f"  Base+L_dir final loss: {last_loss:.6f}")
    
    def _train_lora_adapters_streaming(self, min_samples: int = 10):
        """
        Step 4: Train LoRA adapter for each leaf node using streaming data access.
        
        For each leaf node:
        1. Create a DataLoader that streams only that node's samples (memmap-based)
        2. Create a LoRAExpert wrapping the frozen base model
        3. Train only the LoRA parameters
        4. Store the LoRA adapter
        
        Args:
            min_samples: Minimum samples required to train an adapter
        """
        leaf_level = self.num_levels - 1
        if self.train_internal_experts:
            # All 2K-1 nodes (internal + leaf); each internal node trains on the
            # union of its descendant leaves so tau-cascade backoff has a real
            # expert at every node it can stop on (paper Algorithm 1 / line 364).
            leaf_nodes = [
                n for n in self.tree.nodes
                if n.data_indices is not None and len(n.data_indices) >= min_samples
            ]
        else:
            leaf_nodes = self.tree.get_nodes_at_level(leaf_level)

        logger.info(f"Training LoRA adapters for {len(leaf_nodes)} nodes "
                    f"(internal+leaf={self.train_internal_experts})...")
        logger.info(f"  LoRA config: rank={self.lora_config.rank}, "
                   f"alpha={self.lora_config.alpha}, dropout={self.lora_config.dropout}")
        logger.info(f"  Epochs per adapter: {self.lora_epochs}")
        logger.info(f"  Learning rate: {self.lora_learning_rate}")
        
        # Build the GLOBAL native-target negative pool once (mixing-aware retr).
        if self.retr_weight > 0 and self._global_neg_pool is None:
            try:
                ds = self.train_loader.datasets[0]
                Y_all = ds.target_embeddings
                N = Y_all.shape[0]
                psize = min(self.retr_pool_size, N)
                # deterministic, contiguous sample (sorted indices → memmap-friendly)
                pool_idx = np.sort(np.linspace(0, N - 1, psize).astype(np.int64))
                pool = np.ascontiguousarray(Y_all[pool_idx], dtype=np.float32)
                pool_t = nn.functional.normalize(
                    torch.from_numpy(pool).float().to(self.device), p=2, dim=1)
                self._global_neg_pool = pool_t
                logger.info(f"  retr: global native negative pool = {pool_t.shape[0]} docs "
                            f"(weight={self.retr_weight}, tau={self.retr_tau})")
            except Exception as e:
                logger.warning(f"  retr: could not build global pool ({e}); disabling retr.")
                self.retr_weight = 0.0

        trained_count = 0
        skipped_count = 0

        for node in tqdm(leaf_nodes, desc="Training LoRA adapters"):
            # Check if node has enough samples
            if node.data_indices is None or len(node.data_indices) < min_samples:
                logger.warning(
                    f"Leaf node {node.node_id}: too few samples "
                    f"({len(node.data_indices) if node.data_indices is not None else 0}), "
                    f"skipping"
                )
                skipped_count += 1
                continue
            
            # Create streaming DataLoader for this node's samples
            node_loader = self._create_node_loader(
                node.data_indices,
                batch_size=min(1024, len(node.data_indices)),
                shuffle=True
            )
            
            # Create LoRA expert wrapping a *copy* of the frozen base model.
            # The shared-reference version mutates `self.base_model` in place
            # (LoRAExpert.__init__ -> _inject_lora_layers does setattr on every
            # nn.Linear), so all experts end up sharing the same nested LoRA
            # stack and produce indistinguishable output at inference. Deep
            # copy + freeze gives each expert its own independent LoRA layer
            # over a shared (but untouched) base.
            import copy as _copy
            base_for_expert = _copy.deepcopy(self.base_model)
            for p in base_for_expert.parameters():
                p.requires_grad = False
            lora_expert = LoRAExpert(
                base_model=base_for_expert,
                lora_config=self.lora_config,
                freeze_base=True
            )
            lora_expert.to(self.device)
            
            # Train LoRA adapter
            logger.debug(f"Training LoRA adapter for node {node.node_id} ({len(node.data_indices):,} samples)...")
            self._train_single_lora_adapter(lora_expert, node_loader)
            
            # Store LoRA adapter
            self.lora_adapters[node.node_id] = lora_expert
            trained_count += 1
            
            # Clean up to free memory
            del node_loader
        
        # Log final statistics
        logger.info(f"LoRA adapter training complete:")
        logger.info(f"  Trained: {trained_count}")
        logger.info(f"  Skipped: {skipped_count}")
        
        # Calculate total parameters
        if len(self.lora_adapters) > 0:
            base_params = sum(p.numel() for p in self.base_model.parameters())
            lora_params_per_adapter = sum(
                p.numel() for p in next(iter(self.lora_adapters.values())).get_lora_parameters()
            )
            total_lora_params = lora_params_per_adapter * trained_count
            total_params = base_params + total_lora_params
            
            logger.info(f"Parameter statistics:")
            logger.info(f"  Base model: {base_params:,}")
            logger.info(f"  LoRA per adapter: {lora_params_per_adapter:,}")
            logger.info(f"  Total LoRA: {total_lora_params:,} ({trained_count} adapters)")
            logger.info(f"  Grand total: {total_params:,}")
            logger.info(f"  Compression ratio: {base_params * trained_count / total_params:.1f}x vs full experts")
    
    def _precompute_pca_directions(
        self,
        train_loader,
        max_samples: int = 50_000,
    ) -> None:
        """Compute u_1, u_2 = top-2 right-singular vectors of the centered
        target embedding matrix (== PCA-1 and PCA-2 of the target distribution).

        u_1 and u_2 are stored as unit float32 tensors and used by L_dir at
        per-batch loss time. Subsamples up to max_samples for memory.
        """
        ys: list[np.ndarray] = []
        seen = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                _, tgt = batch[0], batch[1]
            else:
                continue
            if isinstance(tgt, torch.Tensor):
                tgt = tgt.detach().cpu().numpy()
            ys.append(np.asarray(tgt, dtype=np.float32))
            seen += ys[-1].shape[0]
            if seen >= max_samples:
                break
        Y = np.concatenate(ys, axis=0)[:max_samples]
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        # Fixed L_dir scale σ² = mean squared distance of targets from their mean
        # (total target variance, in target-units²). A fixed positive constant —
        # the natural denominator for the raw off-axis energy ‖orth‖² so the
        # "fixed" L_dir form is dimensionless and β stays controllable, without
        # the per-sample ‖e‖² denominator that lets the model game the ratio.
        self._dir_scale = float((Y_centered ** 2).sum(axis=1).mean()) + 1e-8
        # Top-2 PCA directions via SVD on centered matrix.
        _, _, Vt = np.linalg.svd(Y_centered, full_matrices=False)
        u1 = Vt[0].astype(np.float32)
        u2 = Vt[1].astype(np.float32)
        # Unit-norm + deterministic sign: fix the sign so the largest-|coord|
        # element of each direction is positive. SVD's sign is arbitrary; if
        # two translators with the same target space compute PCA independently
        # they may get +u vs -u, which silently breaks cross-source residual
        # alignment in mixing.
        u1 = u1 / (np.linalg.norm(u1) + 1e-12)
        u2 = u2 / (np.linalg.norm(u2) + 1e-12)
        if u1[np.argmax(np.abs(u1))] < 0:
            u1 = -u1
        if u2[np.argmax(np.abs(u2))] < 0:
            u2 = -u2
        self._u1 = torch.from_numpy(u1)
        self._u2 = torch.from_numpy(u2)

    @staticmethod
    def _l_local(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        src_batch: torch.Tensor,
        m: int = 100,
        n_anchors: int = 0,
    ) -> torch.Tensor:
        """Paper L_local — penalize discrepancy in pairwise distances between
        a sample and its m nearest neighbors in the source space.

        Implementation note: the paper defines NN over the full cluster C_i.
        We approximate with within-batch nearest neighbors which is exact when
        a batch covers the full cluster and a faithful proxy for large batches.

        ``n_anchors`` > 0 evaluates the penalty on a random subset of that many
        anchor rows per batch instead of all B — a stochastic estimate that
        averages over batches/epochs. This preserves ``m`` (the neighbor count)
        but cuts the O(B·k·d) gather/norm — the LoRA-stage bottleneck — roughly
        linearly (benchmarked ~5x at 1024/4096). 0 = all rows (paper-faithful).
        """
        B = src_batch.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=outputs.device)
        k = min(m, B - 1)
        # Pairwise source distances; mask self to inf so it's not in top-k.
        # (cdist+topk is only ~3% of L_local's cost — the gather/norm below
        # dominates, hence anchor subsampling rather than touching the cdist.)
        with torch.no_grad():
            src_dist = torch.cdist(src_batch, src_batch)
            diag = torch.eye(B, device=src_dist.device, dtype=torch.bool)
            src_dist = src_dist.masked_fill(diag, float("inf"))
            _, nn_idx = torch.topk(src_dist, k=k, dim=1, largest=False)  # (B, k)
            if 0 < n_anchors < B:
                sel = torch.randperm(B, device=outputs.device)[:n_anchors]
                nn_idx = nn_idx[sel]                                         # (A, k)
            else:
                sel = None
        anc_out = outputs if sel is None else outputs[sel]                  # (A, d_out)
        anc_tgt = targets if sel is None else targets[sel]
        f_nn = outputs[nn_idx]                                              # (A, k, d_out)
        y_nn = targets[nn_idx]
        # anc[i,j] = anc[i] for every j → broadcast the anchor instead of
        # materialising the redundant (A, k, d_out) tensors (~6 GB each at
        # A=4096,k=100,d=3840). Halves peak activation memory; numerically
        # identical. The gather of f_nn/y_nn is the irreducible cost.
        f_pair = (anc_out.unsqueeze(1) - f_nn).norm(dim=-1)                 # (A, k)
        y_pair = (anc_tgt.unsqueeze(1) - y_nn).norm(dim=-1)
        return (f_pair - y_pair).abs().mean()

    def _l_dir(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Paper L_dir — penalize residual energy orthogonal to chosen PCA direction.

        - dir_channel="u1": L_dir^mix from Section 4.1, concentrate residual
          along u_1 (the dominant target-space PCA direction). Use for mixing
          translators and the upstream (first-hop) translator in chaining.
        - dir_channel="u2": L_dir^chain from Section 4.2, concentrate residual
          along u_2 ⊥ u_1. Use for the downstream (second-hop) translator in
          chaining s → Hub → t so its residual decouples from the upstream
          channel.

        Normalization is controlled by ``dir_norm`` (see ``_off_axis_penalty``):
        the default "fixed" form divides the raw off-axis energy by a precomputed
        target-variance scale, keeping β interpretable across batches/translators.
        """
        u = self._u1 if self.dir_channel == "u1" else self._u2
        if u is None:
            return torch.tensor(0.0, device=outputs.device)
        u = u.to(outputs.device)
        return self._off_axis_penalty(outputs - targets, u)

    def _off_axis_penalty(self, e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Penalty on residual energy orthogonal to direction ``u``.

        - dir_norm="fixed" (default): ``‖orth‖²/σ²`` with σ² a precomputed
          constant (mean target variance). Penalizes the raw off-axis energy
          directly — the paper's L_dir form at a fixed scale, monotone in the
          off-axis component and interpretable across batches/translators.
        - dir_norm="fraction": ``‖orth‖²/‖e‖²`` per sample, in [0,1]. Scale-free
          but lowerable by inflating the on-axis component rather than reducing
          the off-axis energy.
        """
        e_sq = e.pow(2).sum(dim=1)                                           # (B,)
        proj_sq = (e @ u).pow(2)                                             # (B,)
        off = (e_sq - proj_sq).clamp_min(0.0)                                # (B,)
        if self.dir_norm == "fixed" and self._dir_scale:
            return (off / self._dir_scale).mean()
        return (off / (e_sq + 1e-8)).mean()

    def _l_retr_global(self, outputs: torch.Tensor, tgt_batch: torch.Tensor) -> torch.Tensor:
        """H-MoE-specific mixing-aware retr: InfoNCE with a GLOBAL native pool.

        anchor a = this cluster's native target y_a (acts as a query); positive =
        the expert's translation f(x_a); negatives = other in-batch translations
        AND a global pool of native docs from ALL clusters. Trains the expert to
        place its outputs so they don't outrank true native docs anywhere in the
        merged corpus — directly attacking the cross-cluster fragmentation that a
        per-cluster loss can't see.
        """
        pool = self._global_neg_pool
        o = nn.functional.normalize(outputs, p=2, dim=1)
        a = nn.functional.normalize(tgt_batch, p=2, dim=1)
        B = o.shape[0]
        idx = torch.arange(B, device=o.device)
        if self.retr_hard_k > 0:
            # Hard-negative mining: per anchor, keep only the top-k native pool
            # docs it is most confusable with (excluding its own near-dup). These
            # are the docs most likely to crowd out true natives in the merged
            # corpus — concentrate the contrastive there instead of on a uniform
            # pool (which over-regularizes, as the pool-size sweep showed).
            k = min(self.retr_hard_k, pool.shape[0])
            with torch.no_grad():
                sims = a @ pool.t()                              # (B, P)
                sims = sims.masked_fill(sims > 0.999, float("-inf"))
                topk = sims.topk(k, dim=1).indices               # (B, k)
            neg = pool[topk]                                     # (B, k, d)
            o_logits = (a @ o.t()) / self.retr_tau              # (B, B), positive on diag
            neg_logits = torch.einsum("bd,bkd->bk", a, neg) / self.retr_tau   # (B, k)
            logits = torch.cat([o_logits, neg_logits], dim=1)   # (B, B+k)
            return nn.functional.cross_entropy(logits, idx)
        cand = torch.cat([o, pool], dim=0)                       # (B+P, d)
        logits = (a @ cand.t()) / self.retr_tau                  # (B, B+P)
        # mask pool columns that are the anchor's own (near-dup) native doc
        with torch.no_grad():
            dup = (a @ pool.t()) > 0.999                         # (B, P)
        logits[:, B:] = logits[:, B:].masked_fill(dup, float("-inf"))
        return nn.functional.cross_entropy(logits, idx)

    def _train_single_lora_adapter(self, lora_expert: LoRAExpert, node_loader):
        """
        Train a single LoRA adapter with the paper's L_reg + α·L_local + β·L_dir loss.

        - L_reg uses the mapper_config's loss_kind (paper Stage-3 uses regression).
        - L_local is the m=100 NN local-structure loss (this implementation does
          within-batch NN; mapping it across full clusters is left as an
          extension when batches don't cover the cluster).
        - L_dir is the directional residual penalty along u_1 (mixing channel).
        """
        # Setup optimizer (only for LoRA parameters!)
        optimizer = torch.optim.Adam(
            lora_expert.get_lora_parameters(),
            lr=self.lora_learning_rate
        )

        # Loss function — honour the mapper_config's loss/output knobs so the
        # LoRA-fitting objective matches the base model's training objective.
        mse_weight = float(getattr(self.mapper_config, "mse_weight", 1.0))
        cosine_weight = float(getattr(self.mapper_config, "cosine_weight", 0.5))
        normalize_output = bool(getattr(self.mapper_config, "normalize_output", False))
        mse_fn = nn.MSELoss()
        # Paper L_local + L_dir
        alpha = self.alpha
        beta = self.beta
        m_nn = self.local_nn_m

        # Training loop
        lora_expert.train()
        for epoch in range(self.lora_epochs):
            total_loss = 0.0
            num_batches = 0

            for src_batch, tgt_batch in node_loader:
                # Move to device
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass (base + LoRA)
                outputs = lora_expert(src_batch)
                if normalize_output:
                    outputs = nn.functional.normalize(outputs, p=2, dim=1)

                # L_reg: pointwise alignment (cos + optional MSE, as before)
                cos_loss = 1 - nn.functional.cosine_similarity(outputs, tgt_batch, dim=1).mean()
                loss = cosine_weight * cos_loss
                if mse_weight > 0:
                    loss = loss + mse_weight * mse_fn(outputs, tgt_batch)
                # L_local: within-batch NN distance preservation
                if alpha > 0:
                    loss = loss + alpha * self._l_local(
                        outputs, tgt_batch, src_batch, m=m_nn, n_anchors=self.local_anchors
                    )
                # L_dir: directional residual alignment to u_1
                if beta > 0 and self._u1 is not None:
                    loss = loss + beta * self._l_dir(outputs, tgt_batch)
                # H-MoE-specific mixing-aware retr (global native negatives)
                if self.retr_weight > 0 and self._global_neg_pool is not None:
                    loss = loss + self.retr_weight * self._l_retr_global(outputs, tgt_batch)
                
                # Backward pass (only LoRA parameters get gradients)
                loss.backward()
                
                # Update LoRA parameters
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            if epoch == 0 or (epoch + 1) % 5 == 0:
                logger.debug(f"  Epoch {epoch+1}/{self.lora_epochs}, Loss: {avg_loss:.6f}")
        
        lora_expert.eval()
    
    def _create_node_loader(
        self,
        indices: np.ndarray,
        batch_size: int = 1024,
        shuffle: bool = True
    ):
        """
        Create a DataLoader for a subset of indices using memmap access.
        
        Uses MultiMemmapDatasetLoader's datasets for efficient memmap access.
        
        Args:
            indices: Array of global sample indices
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader that yields (src_batch, tgt_batch) tuples
        """
        from torch.utils.data import Dataset, DataLoader

        if self.train_loader is None:
            raise ValueError("train_loader not bound. Call _bind_base_dataset() first.")

        indices = np.asarray(indices)
        # Fast path (our single-dataset ArrayDatasetLoader shim): return a
        # batched, threaded-prefetch loader that fancy-indexes the leaf's rows
        # one numpy gather per batch. The previous per-sample, num_workers=0
        # Dataset issued ~batch_size serial Python row-reads per step, making
        # the LoRA stage loader-bound (~75% of each batch's wall time). Reusing
        # ArrayDatasetLoader with `indices=` keeps it lazy (no leaf-subset copy)
        # and parallelises the gather/copy/pin (HMOE_ARRAY_WORKERS).
        if len(getattr(self.train_loader, "datasets", [])) == 1:
            ds = self.train_loader.datasets[0]
            from .._array_loader import ArrayDatasetLoader
            return ArrayDatasetLoader(
                ds.source_embeddings,
                ds.target_embeddings,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=torch.cuda.is_available(),
                indices=indices,
            )

        # Fallback: original per-sample loader for (untested) multi-dataset loaders.
        class MemmapNodeDataset(Dataset):
            """Dataset wrapper using MultiMemmapDatasetLoader's memmap access."""

            def __init__(self, loader, indices: np.ndarray):
                self.loader = loader
                self.indices = indices

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                global_idx = self.indices[idx]
                ds_idx, local_idx = self.loader._find_dataset_idx(global_idx)
                dataset = self.loader.datasets[ds_idx]
                src_emb = dataset.source_embeddings[local_idx]
                tgt_emb = dataset.target_embeddings[local_idx]
                return (
                    torch.from_numpy(src_emb.copy()).float(),
                    torch.from_numpy(tgt_emb.copy()).float(),
                )

        dataset = MemmapNodeDataset(self.train_loader, indices)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    
    def _compute_distances(
        self,
        embeddings: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between embeddings and centroids.
        
        Args:
            embeddings: Input embeddings (N x D)
            centroids: Centroids to compare against (K x D)
            
        Returns:
            Distance matrix (N x K)
        """
        if self.distance_metric == "cosine":
            from sklearn.metrics.pairwise import cosine_distances
            return cosine_distances(embeddings, centroids)
        elif self.distance_metric in ("euclidean", "l2"):
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(embeddings, centroids)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _route_single(self, embedding: np.ndarray) -> int:
        """Route one embedding via paper Algorithm 2 (tau-cascade routing).

        Descend from the root, but at each level stop early when the child
        decision is ambiguous: with d the distances to the children, let
        rho = d_min / d_next (the two nearest children). If rho > tau the
        boundary is ambiguous, so we stop at the current (coarser) node and
        return its id; otherwise we descend to the nearest child. tau=1.0
        disables the backoff (always reaches a leaf), recovering the plain
        nearest-centroid cascade.
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call fit() first.")
        node_id = self.tree.root_id
        x = embedding.reshape(1, -1)
        while True:
            node = self.tree.nodes[node_id]
            if len(node.child_ids) == 0:
                return node_id
            cents = np.stack(
                [self.tree.nodes[c].centroid for c in node.child_ids], axis=0
            )
            d = self._compute_distances(x, cents)[0]
            order = np.argsort(d)
            d_min = float(d[order[0]])
            d_next = float(d[order[1]]) if len(d) > 1 else d_min
            rho = (d_min / d_next) if d_next > 0 else 0.0
            if d_next > 0 and rho > self.tau:
                return node_id
            node_id = node.child_ids[int(order[0])]

    def _route_single_to_leaf(self, embedding: np.ndarray) -> int:
        """
        Route a single embedding through the hierarchy tree to a leaf node.
        
        Uses cascade routing: at each level, finds the nearest child based on
        centroid distance, until reaching a leaf node.
        
        Args:
            embedding: Single embedding vector (D,)
            
        Returns:
            Leaf node ID (expert ID)
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call fit() first.")
        
        # Start from root
        current_node_id = self.tree.root_id
        embedding_reshaped = embedding.reshape(1, -1)  # (1, D)
        
        # Traverse tree from root to leaf
        while True:
            node = self.tree.nodes[current_node_id]
            
            # If leaf node, return its ID
            if len(node.child_ids) == 0:
                return current_node_id
            
            # Get centroids of child nodes
            child_centroids = np.stack(
                [self.tree.nodes[cid].centroid for cid in node.child_ids],
                axis=0
            )  # (num_children, D)
            
            # Compute distances to all children
            distances = self._compute_distances(embedding_reshaped, child_centroids)[0]  # (num_children,)
            
            # Find nearest child
            nearest_child_idx = np.argmin(distances)
            current_node_id = node.child_ids[nearest_child_idx]
    
    def get_expert_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get expert assignments for embeddings using hierarchical routing.
        
        Routes each embedding through the tree from root to leaf, returning
        the leaf node ID (expert ID) for each embedding.
        
        Uses cascade routing: at each level, finds the nearest child node
        based on centroid distance, until reaching a leaf node.
        
        Args:
            embeddings: Input embeddings (N x D)
            
        Returns:
            Array of expert IDs (leaf node IDs) for each embedding (N,)
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call fit() first.")
        
        n_samples = embeddings.shape[0]
        expert_ids = np.empty(n_samples, dtype=np.int32)
        
        # Route each embedding through the tree (tau-cascade, paper Algorithm 2)
        for i in range(n_samples):
            expert_ids[i] = self._route_single(embeddings[i])
        
        return expert_ids
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using hierarchical routing with LoRA adapters.
        
        For each embedding:
        1. Route through tree from root to leaf (cascade routing)
        2. Transform with base model + the leaf node's LoRA adapter
        
        Args:
            embeddings: Source embeddings to transform (N x D)
            
        Returns:
            Transformed embeddings (N x D_out)
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call fit() first.")
        
        if self.base_model is None:
            raise ValueError("Base model not trained. Call fit() first.")
        
        n_samples = embeddings.shape[0]
        results = np.zeros((n_samples, self.output_dim), dtype=embeddings.dtype)
        
        logger.info(f"Transforming {n_samples:,} embeddings using hierarchical LoRA routing...")
        
        # Route embeddings to leaf nodes (experts)
        expert_ids = self.get_expert_assignments(embeddings)
        
        # Count assignments per expert
        unique_experts, counts = np.unique(expert_ids, return_counts=True)
        logger.info(f"Assignment distribution: {len(unique_experts)} experts used")
        for expert_id, count in zip(unique_experts, counts):
            logger.debug(f"  Expert {expert_id}: {count:,} samples ({count/n_samples*100:.1f}%)")
        
        # Transform each group of embeddings with their assigned expert
        normalize_output = bool(getattr(self.mapper_config, "normalize_output", False))
        self.base_model.eval()
        with torch.no_grad():
            for expert_id in unique_experts:
                mask = (expert_ids == expert_id)
                expert_embeddings = embeddings[mask]

                # Convert to tensor
                expert_embeddings_tensor = torch.from_numpy(expert_embeddings).float().to(self.device)

                # Get the LoRA adapter for this leaf node
                if expert_id in self.lora_adapters:
                    # Use base model + LoRA adapter
                    lora_expert = self.lora_adapters[expert_id]
                    lora_expert.eval()
                    transformed_tensor = lora_expert(expert_embeddings_tensor)
                    if normalize_output:
                        transformed_tensor = nn.functional.normalize(transformed_tensor, p=2, dim=1)
                else:
                    # Fallback: use only base model if no adapter trained
                    logger.warning(
                        f"Leaf node {expert_id} has no LoRA adapter. "
                        f"Using base model only."
                    )
                    transformed_tensor = self.base_model(expert_embeddings_tensor)
                    if normalize_output:
                        transformed_tensor = nn.functional.normalize(transformed_tensor, p=2, dim=1)

                # Convert back to numpy
                transformed = transformed_tensor.cpu().numpy()
                results[mask] = transformed
        
        return results
    
    def save_lora_adapters(self, save_dir: str):
        """
        Save all LoRA adapters to disk.
        
        Only saves the lightweight LoRA parameters, not the base model.
        
        Args:
            save_dir: Directory to save adapters
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for node_id, lora_expert in self.lora_adapters.items():
            adapter_path = os.path.join(save_dir, f"lora_adapter_{node_id}.pt")
            lora_expert.save_lora_weights(adapter_path)
        
        logger.info(f"Saved {len(self.lora_adapters)} LoRA adapters to {save_dir}")
    
    def load_lora_adapters(self, save_dir: str):
        """
        Load LoRA adapters from disk.
        
        Args:
            save_dir: Directory containing saved adapters
        """
        import os
        
        for node_id in self.lora_adapters.keys():
            adapter_path = os.path.join(save_dir, f"lora_adapter_{node_id}.pt")
            if os.path.exists(adapter_path):
                self.lora_adapters[node_id].load_lora_weights(adapter_path)
        
        logger.info(f"Loaded LoRA adapters from {save_dir}")
