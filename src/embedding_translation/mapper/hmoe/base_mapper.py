"""
Base mapper abstract class for all MoE implementations.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional, Dict, Any, Union, overload, TYPE_CHECKING
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from .core.mlp import VectorMapper

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class BaseMoEMapper(VectorMapper):
    """
    Abstract base class for all MoE mappers.
    
    Inherits from VectorMapper to maintain compatibility with the existing pipeline,
    but uses a loader-based training interface for better efficiency.
    """
    
    def __init__(self):
        super().__init__()
        self._is_fitted = False
    
    def transform_dataset(self, config, dataloader: "DataLoader", cache_path: str) -> None:
        """
        Transform dataset based on config strategy.
        
        Add new strategy here by adding elif branch.
        
        Args:
            config: Configuration object
            dataloader: DataLoader for input data
            cache_path: Output path for transformed embeddings
        """
        # Get strategy from config (with fallback)
        strategy = getattr(config.mapper, 'transform_strategy', 'cluster_then_route')
        
        if strategy == 'cluster_then_route':
            # Strategy 1: Cluster first, then route each cluster to nearest expert
            num_clusters = getattr(config.mapper, 'transform_num_clusters', 16)
            self.cluster_and_transform(
                dataloader=dataloader,
                cache_path=cache_path,
                num_clusters=num_clusters
            )
        
        elif strategy == 'direct_route':
            # Strategy 2: Route each vector directly to nearest expert
            self.direct_route_transform(
                dataloader=dataloader,
                cache_path=cache_path
            )
        
        else:
            raise ValueError(
                f"Unknown transform strategy: '{strategy}'. "
                f"Available: ['cluster_then_route', 'direct_route']"
            )
    
    # Type hints for overloaded fit method
    @overload
    def fit(self, train_data: np.ndarray, target_data: np.ndarray, 
            reference_indices: np.ndarray) -> None: ...
    
    @overload
    def fit(self, train_data: "DataLoader") -> None: ...
    
    def fit(self, train_data: Union[np.ndarray, "DataLoader"], 
            target_data: Optional[np.ndarray] = None, 
            reference_indices: Optional[np.ndarray] = None) -> None:
        """
        Unified fit interface - accepts embeddings or DataLoader.
        
        Usage:
            # Mode 1: numpy arrays (VectorMapper interface)
            mapper.fit(source_emb, target_emb, reference_indices)
            
            # Mode 2: DataLoader (MoE optimized interface)  
            mapper.fit(multi_memmap_dataloader)
        
        Args:
            train_data: Source embeddings (np.ndarray) OR DataLoader
            target_data: Target embeddings (required if train_data is np.ndarray)
            reference_indices: Reference indices (required if train_data is np.ndarray)
        """
        # Mode detection: check if first arg is numpy array
        if isinstance(train_data, np.ndarray):
            if target_data is None or reference_indices is None:
                raise ValueError("target_data and reference_indices required when using numpy arrays")

            from ._array_loader import ArrayDatasetLoader

            logger.info(f"Training with numpy arrays ({len(reference_indices)} reference samples)")
            # Fast path: when reference_indices == np.arange(len(train_data))
            # (the common harness.train_hmoe call), skip the fancy-indexing
            # copy. On a memmap-backed Fever array (5.4M × 4096 fp32 = 88 GB)
            # train_data[np.arange(N)] would materialise the whole file into
            # RAM up-front; the identity-select check costs O(N) once but
            # saves 176 GB of paired source+target copy.
            n = len(train_data)
            ri = reference_indices
            if (
                len(ri) == n
                and int(ri[0]) == 0
                and int(ri[-1]) == n - 1
                and np.array_equal(ri, np.arange(n))
            ):
                source_ref = train_data
                target_ref = target_data
            else:
                source_ref = train_data[ri]
                target_ref = target_data[ri]
            # Base-stage batch is env-tunable: a larger batch gives the GPU
            # more work per step (raising utilisation) and fewer Python-loop
            # round-trips; combined with ArrayDatasetLoader's threaded prefetch
            # (HMOE_ARRAY_WORKERS) it keeps the GPU fed instead of starved.
            import os as _os
            _base_bs = int(_os.environ.get("HMOE_BASE_BATCH", "4096"))
            loader = ArrayDatasetLoader(
                source_ref, target_ref, batch_size=_base_bs, shuffle=True,
                pin_memory=torch.cuda.is_available(),
            )
            self._fit_from_loader(loader)
        else:
            # Loader mode
            logger.info("Training with DataLoader (optimized for large datasets)")
            self._fit_from_loader(train_data)
        
        self._is_fitted = True
        logger.info("✓ Training completed")
    
    def fit_multi(self, multi_dataloader) -> None:
        """
        Preferred training interface for MoE mappers.
        
        Uses MultiMemmapDatasetLoader for efficient multi-dataset training
        without loading everything into memory.
        
        Args:
            multi_dataloader: MultiMemmapDatasetLoader instance
        """
        logger.info("Training with MultiMemmapDatasetLoader (recommended for MoE)")
        self._fit_from_loader(multi_dataloader)
        self._is_fitted = True
        logger.info("✓ Training completed via fit_multi() interface")
    
    @abstractmethod
    def _fit_from_loader(self, train_loader):
        """
        Internal training method that subclasses must implement.
        
        This method receives a DataLoader (either from fit() or fit_multi())
        and performs the actual training logic.
        
        Args:
            train_loader: DataLoader or MultiMemmapDatasetLoader
        """
        pass
    
    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using the trained mapper.
        
        Args:
            embeddings: Input embeddings to transform
            
        Returns:
            Transformed embeddings
        """
        pass
    
    def get_expert_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get expert assignments for embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Array of expert IDs (one per embedding)
        """
        if hasattr(self, 'router') and self.router is not None:
            return self.router.route(embeddings)
        raise NotImplementedError("get_expert_assignments must be implemented by subclass or have a router")
    
    def _find_nearest_experts_for_clusters(self, cluster_centroids: np.ndarray) -> Dict[int, int]:
        """
        Find the nearest expert for each cluster centroid.
        
        Args:
            cluster_centroids: Array of cluster centroids (num_clusters x D)
            
        Returns:
            Dictionary mapping cluster_id -> expert_id
        """
        if not hasattr(self, 'clusterer') or self.clusterer is None:
            raise ValueError("Clusterer not initialized. Call fit() first.")
        
        if not hasattr(self.clusterer, 'centroids') or self.clusterer.centroids is None:
            raise ValueError("Expert cluster centers not available. Call fit() first.")
        
        # Get expert centroids from the clusterer
        expert_centroids = self.clusterer.centroids  # (num_experts x D)
        
        # Get distance metric (default to cosine if not set)
        distance_metric = getattr(self, 'distance_metric', 'cosine')
        
        # For each cluster centroid, find the nearest expert centroid
        cluster_to_expert = {}
        
        for cluster_id, cluster_centroid in enumerate(cluster_centroids):
            # Calculate distances to all expert centroids
            if distance_metric == "cosine":
                # Cosine similarity (higher is better)
                cluster_norm = np.linalg.norm(cluster_centroid)
                expert_norms = np.linalg.norm(expert_centroids, axis=1)
                
                # Avoid division by zero
                if cluster_norm < 1e-10:
                    expert_id = 0
                else:
                    similarities = np.dot(expert_centroids, cluster_centroid) / (expert_norms * cluster_norm + 1e-10)
                    expert_id = int(np.argmax(similarities))
            else:
                # Euclidean distance (lower is better)
                distances = np.linalg.norm(expert_centroids - cluster_centroid, axis=1)
                expert_id = int(np.argmin(distances))
            
            cluster_to_expert[cluster_id] = expert_id
        
        return cluster_to_expert

    def cluster_and_transform(
        self,
        dataloader: "DataLoader",
        cache_path: str,
        num_clusters: int = 16,
        batch_size: int = 10000
    ) -> None:
        """
        Cluster data first, then transform each cluster with its nearest expert.
        
        Args:
            dataloader: DataLoader that yields (source, target) batches
            cache_path: Path to save transformed embeddings (.npy file)
            num_clusters: Number of clusters
            batch_size: Batch size for clustering
        """
        import os
        
        if not self._is_fitted:
            raise ValueError("Mapper not fitted. Call fit() first.")
        
        logger.info(f"Cluster-and-transform: {num_clusters} clusters -> {cache_path}")
        
        # Pass 1: Incremental clustering
        clusterer = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=42,
            batch_size=min(batch_size, 1024),
            verbose=0,
            compute_labels=False,
            n_init=3
        )
        
        n_samples = 0
        for batch in tqdm(dataloader, desc="Clustering"):
            src_batch = batch[0] if isinstance(batch, tuple) else batch
            if isinstance(src_batch, torch.Tensor):
                src_batch = src_batch.cpu().numpy()
            clusterer.partial_fit(src_batch)
            n_samples += len(src_batch)
        
        # Map clusters to experts
        cluster_to_expert = self._find_nearest_experts_for_clusters(clusterer.cluster_centers_)
        logger.info(f"Mapped {num_clusters} clusters to experts")
        
        # Pass 2: Transform and save
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        output_memmap = np.lib.format.open_memmap(
            cache_path,
            mode='w+',
            dtype=np.float32,
            shape=(n_samples, self.output_dim)
        )
        
        processed = 0
        for batch in tqdm(dataloader, desc="Transforming"):
            src_batch = batch[0] if isinstance(batch, tuple) else batch
            if isinstance(src_batch, torch.Tensor):
                src_batch = src_batch.cpu().numpy()
            
            batch_clusters = clusterer.predict(src_batch)
            
            # Transform by cluster
            for cluster_id in range(num_clusters):
                mask = (batch_clusters == cluster_id)
                if not mask.any():
                    continue
                
                expert_id = cluster_to_expert[cluster_id]
                cluster_samples = src_batch[mask]
                
                # Use specific expert or fallback to transform
                if hasattr(self, 'experts') and expert_id in self.experts:
                    transformed = self.experts[expert_id].transform(cluster_samples)
                else:
                    transformed = self.transform(cluster_samples)
                
                # Write to output
                global_indices = processed + np.where(mask)[0]
                output_memmap[global_indices] = transformed
            
            processed += len(src_batch)
        
        output_memmap.flush()
        del output_memmap

        assert processed == n_samples, f"Processed {processed} samples, expected {n_samples}"
        
        logger.info(f"✓ Saved {n_samples:,} samples to {cache_path}")
    
    def direct_route_transform(
        self,
        dataloader: "DataLoader",
        cache_path: str
    ) -> None:
        """
        Direct routing: each vector is routed to its nearest expert.
        
        Args:
            dataloader: DataLoader for input data
            cache_path: Output path for transformed embeddings
        """
        import os
        
        if not self._is_fitted:
            raise ValueError("Mapper not fitted. Call fit() first.")
        
        logger.info(f"Direct route transform -> {cache_path}")
        
        # Count samples first
        n_samples = 0
        for batch in tqdm(dataloader, desc="Counting samples"):
            src_batch = batch[0] if isinstance(batch, tuple) else batch
            if isinstance(src_batch, torch.Tensor):
                src_batch = src_batch.cpu().numpy()
            n_samples += len(src_batch)
        
        logger.info(f"Total samples: {n_samples:,}")
        
        # Create output file
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        output_memmap = np.lib.format.open_memmap(
            cache_path,
            mode='w+',
            dtype=np.float32,
            shape=(n_samples, self.output_dim)
        )
        
        # Transform with direct routing
        processed = 0
        for batch in tqdm(dataloader, desc="Transforming"):
            src_batch = batch[0] if isinstance(batch, tuple) else batch
            if isinstance(src_batch, torch.Tensor):
                src_batch = src_batch.cpu().numpy()
            
            batch_size = len(src_batch)
            
            # Get expert assignment for each sample
            expert_ids = self.get_expert_assignments(src_batch)
            
            # Group by expert and transform
            for expert_id in np.unique(expert_ids):
                mask = (expert_ids == expert_id)
                if not mask.any():
                    continue
                
                samples = src_batch[mask]
                
                # Use specific expert or fallback
                if hasattr(self, 'experts') and expert_id in self.experts:
                    transformed = self.experts[expert_id].transform(samples)
                else:
                    transformed = self.transform(samples)
                
                # Write to output
                local_indices = np.where(mask)[0]
                global_indices = processed + local_indices
                output_memmap[global_indices] = transformed
            
            processed += batch_size
        
        output_memmap.flush()
        del output_memmap
        
        assert processed == n_samples, f"Processed {processed} samples, expected {n_samples}"
        
        logger.info(f"✓ Saved {n_samples:,} samples to {cache_path}")
