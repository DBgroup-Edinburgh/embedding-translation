"""
Hierarchical MoE Mapper implementation.

Three-step hierarchical construction:
1. MiniBatchKMeans to get leaf clusters
2. Bottom-up tree construction using KMeans on node centroids
3. Train experts for each leaf node using streaming data access
"""

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
from typing import Tuple, List, Optional

from ..base_mapper import BaseMoEMapper
from ..tree_structure import BottomUpHierarchyTree, TreeNode
from ..core.mlp import SimpleLinearMapper


class HierarchicalMoEMapper(BaseMoEMapper):
    """
    Hierarchical Mixture of Experts Mapper.
    
    Multi-level tree-based expert system with cascade routing.
    
    Construction process:
    1. MiniBatchKMeans streaming clustering to get leaf clusters
    2. Bottom-up tree construction using KMeans on node centroids
    3. Train experts for each leaf node using memmap-based streaming
    """
    
    def __init__(
        self,
        num_levels: int = 3,
        branch_factor: int = 4,
        mapper_config = None,
        distance_metric: str = "cosine",
        transform_strategy: str = "cluster_then_route",
        **kwargs
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.branch_factor = branch_factor
        self.mapper_config = mapper_config
        self.distance_metric = distance_metric
        
        # Calculate total number of leaf clusters
        # For a tree: num_leaves = branch_factor^(num_levels-1)
        self.num_leaf_clusters = branch_factor ** (num_levels - 1)
        
        # Tree structure
        self.tree: Optional[BottomUpHierarchyTree] = None
        
        # Data storage - keep reference to train_loader for memmap access
        self.train_loader = None  # MultiMemmapDatasetLoader instance
        self.num_samples: Optional[int] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        
        # Leaf cluster mappings
        self.sample_to_leaf: Optional[np.ndarray] = None  # sample_to_leaf[i] = leaf_id
        self.leaf_to_indices: Optional[List[np.ndarray]] = None  # leaf_to_indices[leaf_id] = [i1, i2, ...]
        
        # Experts (one per leaf node)
        self.experts = {}  # node_id -> VectorMapper
        
        logger.info(
            f"Initialized HierarchicalMoEMapper: {self.num_levels} levels, "
            f"branch_factor={self.branch_factor}, "
            f"num_leaf_clusters={self.num_leaf_clusters}"
        )
    
    def _fit_from_loader(self, train_loader):
        """
        Train hierarchical MoE using three-step process.
        
        Steps:
        1. MiniBatchKMeans streaming clustering to get leaf clusters
        2. Bottom-up tree construction using KMeans on node centroids
        3. Train experts for each leaf node using streaming data access
        """
        logger.info(f"Training HierarchicalMoE ({self.num_levels} levels, "
                   f"branch_factor={self.branch_factor}, "
                   f"num_leaf_clusters={self.num_leaf_clusters})")
        
        # Step 0: Bind base dataset
        self._bind_base_dataset(train_loader)
        
        # Step 1: MiniBatchKMeans streaming clustering
        logger.info("Step 1: MiniBatchKMeans streaming clustering...")
        leaf_centroids = self._stream_cluster_bottom_layer(train_loader)
        logger.info(f"✓ Got {len(leaf_centroids)} leaf cluster centroids")
        
        # Step 2: Bottom-up tree construction
        logger.info("Step 2: Bottom-up tree construction...")
        self._build_hierarchy_bottom_up(leaf_centroids)
        logger.info(f"✓ Built hierarchy tree with {len(self.tree.nodes)} nodes")
        
        # Step 3: Train experts for leaf nodes
        logger.info("Step 3: Training experts for leaf nodes...")
        self._train_leaf_experts()
        logger.info(f"✓ Trained {len(self.experts)} leaf experts")
        
        logger.info("✓ HierarchicalMoE training completed")
    
    def _bind_base_dataset(self, train_loader):
        """
        Bind train_loader and extract dimensions.
        
        Uses MultiMemmapDatasetLoader's datasets for direct memmap access.
        """
        # Store loader reference for memmap access
        self.train_loader = train_loader
        self.input_dim = train_loader.source_embedding_dim
        self.output_dim = train_loader.target_embedding_dim
        
        # Get total samples
        if hasattr(train_loader, 'total_samples'):
            self.num_samples = train_loader.total_samples
        else:
            # Fallback: count by iterating
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
        from sklearn.cluster import MiniBatchKMeans
        
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
        
        leaf_centroids = mbk.cluster_centers_  # (K, d_in)
        logger.info(f"Learned {len(leaf_centroids)} leaf cluster centers")
        
        # Pass 2: Assign each sample to a leaf cluster
        logger.info("Pass 2: Assigning samples to leaf clusters...")
        N = self.num_samples
        sample_to_leaf = np.empty(N, dtype=np.int32)
        
        global_idx = 0
        for src_batch, tgt_batch in tqdm(train_loader, desc="Assigning"):
            if isinstance(src_batch, torch.Tensor):
                src_batch = src_batch.cpu().numpy()
            
            labels = mbk.predict(src_batch)  # Each sample gets a cluster id
            B = len(labels)
            if global_idx + B > N:
                # Handle last batch that might be smaller
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
        from sklearn.cluster import KMeans
        
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
            )  # (num_children, d_in)
            
            # Cluster children into parents
            if num_parents == 1:
                # Only one parent: all children go to it
                labels = np.zeros(num_children, dtype=np.int32)
                parent_centroids = [node_centroids.mean(axis=0)]
            else:
                # Use KMeans to cluster children
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
    
    def _train_leaf_experts(self, min_samples: int = 10):
        """
        Train expert for each leaf node using streaming data access.
        
        For each leaf node:
        1. Create a DataLoader that streams only that node's samples (memmap-based)
        2. Train a SimpleLinearMapper expert
        3. Store the expert in self.experts dict
        
        Args:
            min_samples: Minimum samples required to train an expert
        """
        leaf_level = self.num_levels - 1
        leaf_nodes = self.tree.get_nodes_at_level(leaf_level)
        
        logger.info(f"Training {len(leaf_nodes)} leaf experts (streaming, low memory)...")
        
        trained_count = 0
        skipped_count = 0
        
        for node in tqdm(leaf_nodes, desc="Training leaf experts"):
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
            
            # Create expert
            expert = SimpleLinearMapper(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                **self.mapper_config.model_dump()
            )
            
            # Train expert using fit_multi (streaming training)
            logger.debug(f"Training expert for node {node.node_id} ({len(node.data_indices):,} samples)...")
            expert.fit_multi(node_loader)
            
            # Store expert
            self.experts[node.node_id] = expert
            trained_count += 1
            
            # Clean up to free memory
            del node_loader
        
        logger.info(
            f"Expert training complete: {trained_count} trained, {skipped_count} skipped"
        )
    
    def _get_sample_by_index(self, global_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get source and target embeddings for a single global index.
        
        Uses MultiMemmapDatasetLoader's memmap access for zero-copy efficiency.
        
        Args:
            global_idx: Global sample index
            
        Returns:
            Tuple of (source_embedding, target_embedding)
        """
        if self.train_loader is None:
            raise ValueError("train_loader not bound. Call _bind_base_dataset() first.")
        
        # Use loader's method to find which dataset and local index
        ds_idx, local_idx = self.train_loader._find_dataset_idx(global_idx)
        
        # Direct memmap access - zero copy!
        dataset = self.train_loader.datasets[ds_idx]
        src_emb = dataset.source_embeddings[local_idx]
        tgt_emb = dataset.target_embeddings[local_idx]
        
        return src_emb, tgt_emb
    
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
        
        class MemmapNodeDataset(Dataset):
            """Dataset wrapper using MultiMemmapDatasetLoader's memmap access."""
            
            def __init__(self, loader, indices: np.ndarray):
                self.loader = loader
                self.indices = indices
            
            def __len__(self) -> int:
                return len(self.indices)
            
            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                """Load a single sample using memmap access."""
                global_idx = self.indices[idx]
                ds_idx, local_idx = self.loader._find_dataset_idx(global_idx)
                
                # Direct memmap access - zero copy!
                dataset = self.loader.datasets[ds_idx]
                src_emb = dataset.source_embeddings[local_idx]
                tgt_emb = dataset.target_embeddings[local_idx]
                
                # Convert to torch tensors (creates a copy, but necessary for PyTorch)
                return (
                    torch.from_numpy(src_emb.copy()).float(),
                    torch.from_numpy(tgt_emb.copy()).float()
                )
        
        dataset = MemmapNodeDataset(self.train_loader, indices)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Single-threaded for memmap
            pin_memory=torch.cuda.is_available(),
            drop_last=False
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
        elif self.distance_metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(embeddings, centroids)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _route_single_to_leaf(self, embedding: np.ndarray) -> int:
        """
        Route a single embedding through the hierarchy tree to a leaf node.
        
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
        
        # Route each embedding through the tree
        # Note: This could be optimized for batch processing, but for now
        # we route one by one to handle variable tree structure
        for i in range(n_samples):
            expert_ids[i] = self._route_single_to_leaf(embeddings[i])
        
        return expert_ids
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using hierarchical routing.
        
        For each embedding:
        1. Route through tree from root to leaf (cascade routing)
        2. Transform with the leaf node's expert
        
        Args:
            embeddings: Source embeddings to transform (N x D)
            
        Returns:
            Transformed embeddings (N x D_out)
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call fit() first.")
        
        n_samples = embeddings.shape[0]
        results = np.zeros((n_samples, self.output_dim), dtype=embeddings.dtype)
        
        logger.info(f"Transforming {n_samples:,} embeddings using hierarchical routing...")
        
        # Route embeddings to leaf nodes (experts)
        expert_ids = self.get_expert_assignments(embeddings)
        
        # Count assignments per expert
        unique_experts, counts = np.unique(expert_ids, return_counts=True)
        logger.info(f"Assignment distribution: {len(unique_experts)} experts used")
        for expert_id, count in zip(unique_experts, counts):
            logger.debug(f"  Expert {expert_id}: {count:,} samples ({count/n_samples*100:.1f}%)")
        
        # Transform each group of embeddings with their assigned expert
        for expert_id in unique_experts:
            mask = (expert_ids == expert_id)
            expert_embeddings = embeddings[mask]
            
            # Get the expert for this leaf node
            if expert_id in self.experts:
                # Use trained expert
                expert = self.experts[expert_id]
                transformed = expert.transform(expert_embeddings)
            else:
                # Fallback: use identity if no expert trained
                logger.warning(
                    f"Leaf node {expert_id} has no trained expert. "
                    f"Using zero/identity transformation."
                )
                transformed = np.zeros(
                    (len(expert_embeddings), self.output_dim),
                    dtype=embeddings.dtype
                )
            
            results[mask] = transformed
        
        return results

