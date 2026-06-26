"""
Flat MoE Mapper implementation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
from loguru import logger
from tqdm import tqdm

from ..base_mapper import BaseMoEMapper
from ..expert_clusterer import ExpertClusterer
from ..router import FlatRouter
from ..core.mlp import SimpleLinearMapper


class FlatMoEMapper(BaseMoEMapper):
    """
    Flat Mixture of Experts Mapper.
    
    Single-level expert system with simple nearest-neighbor routing.
    Each expert is trained on a cluster of similar data points.
    """
    
    def __init__(
        self,
        num_experts: int = 8,
        mapper_config = None,
        distance_metric: str = "cosine",
        clustering_method: str = "kmeans",
        random_state: int = 42,
        clustering_sample_size: int = 100000,
        use_soft_routing: bool = False,
        gating_temperature: float = 1.0,
        **kwargs
    ):
        """
        Initialize Flat MoE Mapper.
        
        Args:
            num_experts: Number of experts
            mapper_config: Configuration for expert mappers
            distance_metric: Distance metric for routing
            clustering_method: Clustering algorithm
            random_state: Random seed
            clustering_sample_size: Max samples for clustering
            use_soft_routing: Use soft routing (weighted combination)
            gating_temperature: Temperature for soft routing
        """
        self.num_experts = num_experts
        self.mapper_config = mapper_config
        self.distance_metric = distance_metric
        self.clustering_method = clustering_method
        self.random_state = random_state
        self.clustering_sample_size = clustering_sample_size
        self.use_soft_routing = use_soft_routing
        self.gating_temperature = gating_temperature
        
        # Components (initialized during fit)
        self.clusterer = ExpertClusterer(
            num_experts=num_experts,
            clustering_method=clustering_method,
            distance_metric=distance_metric,
            random_state=random_state
        )
        self.router: Optional[FlatRouter] = None
        self.experts: Dict[int, SimpleLinearMapper] = {}
        
        # Model state
        self.is_fitted = False
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        
        logger.info(f"Initialized FlatMoEMapper with {num_experts} experts")
    
    def _fit_from_loader(self, train_loader):
        """
        Train the Flat MoE system from a DataLoader.
        
        This is the internal implementation called by fit() or fit_multi().
        
        Steps:
        1. Perform incremental clustering on all training data
        2. Build expert mappers for each cluster
        3. Train each expert on its assigned data
        4. Initialize router with cluster centroids
        
        Args:
            train_loader: DataLoader or MultiMemmapDatasetLoader instance
        """
        self.input_dim = train_loader.source_embedding_dim
        self.output_dim = train_loader.target_embedding_dim
        
        logger.info(f"Starting Flat MoE training with {self.num_experts} experts")
        logger.info(f"Input dim: {self.input_dim}, Output dim: {self.output_dim}")
        
        # Step 1: Incremental clustering on ALL data
        logger.info("Step 1: Incremental clustering on all data...")
        cluster_result = self.clusterer.cluster_references(
            train_loader,
            batch_size=10000
        )
        logger.info(f"Clustering completed: {cluster_result['num_clusters']} clusters")
        
        # Log cluster distribution
        cluster_sizes = cluster_result['cluster_sizes']
        logger.info(f"Cluster sizes: {cluster_sizes}")
        for exp_id, count in enumerate(cluster_sizes):
            total = sum(cluster_sizes)
            logger.info(f"  Expert {exp_id}: {count:,} samples ({count/total*100:.1f}%)")
        
        # Step 2: Build and train experts
        logger.info("Step 2: Building and training experts...")
        
        for expert_id in range(self.num_experts):
            # Get indices assigned to this expert
            expert_indices = np.array(
                self.clusterer.expert_assignments.get(expert_id, [])
            )
            
            if len(expert_indices) == 0:
                logger.warning(f"Expert {expert_id} has no samples, skipping...")
                continue
            
            logger.info(f"\nTraining Expert {expert_id}/{self.num_experts} "
                       f"with {len(expert_indices):,} samples...")
            
            # Create expert mapper
            expert = SimpleLinearMapper(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                **self.mapper_config.model_dump()
            )
            
            # Create sub-loader for this expert
            expert_loader = self._create_expert_loader(train_loader, expert_indices)
            
            # Train this expert
            expert.fit_with_loader(expert_loader)
            
            # Store trained expert
            self.experts[expert_id] = expert
            
            logger.info(f"Expert {expert_id} training completed")
        
        # Step 3: Initialize router
        logger.info("Step 3: Building router...")
        self.router = FlatRouter(
            centroids=self.clusterer.centroids,
            distance_metric=self.distance_metric
        )
        logger.info(f"Router built with {len(self.clusterer.centroids)} centroids")
        
        self.is_fitted = True
        logger.info(f"Flat MoE training completed! Trained {len(self.experts)} experts.")
    
    def _create_expert_loader(self, train_loader, expert_indices: np.ndarray) -> DataLoader:
        """
        Create a DataLoader for a specific expert using subset of indices.
        
        Args:
            train_loader: MultiMemmapDatasetLoader instance
            expert_indices: Global indices assigned to this expert
            
        Returns:
            PyTorch DataLoader for this expert
        """
        class ExpertDataset(Dataset):
            def __init__(self, loader, indices: np.ndarray):
                self.loader = loader
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                global_idx = self.indices[idx]
                src, tgt = self.loader.load_batch(
                    global_idx, 
                    global_idx + 1, 
                    return_target=True
                )
                return (
                    torch.from_numpy(src[0]).float(),
                    torch.from_numpy(tgt[0]).float()
                )
        
        dataset = ExpertDataset(train_loader, expert_indices)
        
        return DataLoader(
            dataset,
            batch_size=self.mapper_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using the trained experts.
        
        For each embedding:
        1. Route to nearest expert using centroids
        2. Transform with that expert
        
        Args:
            embeddings: Source embeddings to transform (N x D)
            
        Returns:
            Transformed embeddings (N x D_out)
        """
        if not self.is_fitted:
            raise ValueError("Flat MoE has not been fitted. Call fit() first.")
        
        if not self.experts:
            raise ValueError("No experts were trained. Check clustering results.")
        
        if self.router is None:
            raise ValueError("Router has not been initialized. Call fit() first.")
        
        n_samples = embeddings.shape[0]
        results = np.zeros((n_samples, self.output_dim), dtype=embeddings.dtype)
        
        logger.info(f"Transforming {n_samples:,} embeddings using {len(self.experts)} experts...")
        
        # Route embeddings to experts
        expert_ids, distances = self.router.route(embeddings)
        
        # Count assignments per expert
        assignment_counts = np.bincount(expert_ids, minlength=self.num_experts)
        logger.info(f"Assignment distribution: {assignment_counts.tolist()}")
        
        # Transform each group of embeddings with their assigned expert
        for expert_id in range(self.num_experts):
            expert_mask = (expert_ids == expert_id)
            n_assigned = expert_mask.sum()
            
            if n_assigned == 0:
                continue
            
            if expert_id not in self.experts:
                logger.warning(f"Expert {expert_id} not found, using expert 0 as fallback")
                expert_id = 0
            
            # Get embeddings assigned to this expert
            expert_inputs = embeddings[expert_mask]
            
            # Transform using this expert
            expert_outputs = self.experts[expert_id].transform(expert_inputs)
            
            # Store results
            results[expert_mask] = expert_outputs
            
            if expert_id < 3:  # Log first few experts
                logger.info(f"  Expert {expert_id}: transformed {n_assigned:,} samples")
        
        logger.info("Transformation completed")
        return results
    
    def stream_transform(
        self, 
        dataloader: DataLoader,
        use_clustering: bool = True,
        num_clusters: int = 16
    ) -> np.ndarray:
        """
        Stream transform embeddings from a dataloader.
        
        Two modes:
        1. use_clustering=True: First cluster the data, then transform each cluster
           with its nearest expert (more efficient for large batches)
        2. use_clustering=False: Direct transform batch by batch
        
        Args:
            dataloader: DataLoader instance
            use_clustering: Whether to cluster first before transforming
            num_clusters: Number of clusters to use (only if use_clustering=True)
            
        Returns:
            Transformed embeddings (N x D_out)
        """
        if not self.is_fitted:
            raise ValueError("Flat MoE has not been fitted. Call fit() first.")
        
        if not self.experts:
            raise ValueError("No experts were trained. Check clustering results.")
        
        if self.router is None:
            raise ValueError("Router has not been initialized. Call fit() first.")
        
        if use_clustering:
            # Use cluster-and-transform strategy from base class
            logger.info(f"Using cluster-and-transform with {num_clusters} clusters")
            return self.cluster_and_transform(
                dataloader=dataloader,
                num_clusters=num_clusters
            )
        else:
            # Direct batch-by-batch transform
            logger.info("Using direct batch-by-batch transform")
            all_results = []
            
            for src_batch, tgt_batch in dataloader:
                src_np = src_batch.cpu().numpy()
                results = self.transform(src_np)
                all_results.append(results)
            
            return np.concatenate(all_results, axis=0)
    
    def get_expert_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get expert assignments for embeddings without transformation.
        
        Args:
            embeddings: Source embeddings (N x D)
            
        Returns:
            Expert IDs for each embedding (N,)
        """
        if not self.is_fitted or self.router is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        expert_ids, _ = self.router.route(embeddings)
        return expert_ids

