"""
K-means clustering strategy for VectorMerge.

This module implements K-means clustering strategy
for clustering reference points and assigning target points to clusters.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from loguru import logger

from ..base import ClusteringStrategy, ClusterData, ClusteringResult
from ...config import ClusteringConfig

class KMeansClusteringStrategy(ClusteringStrategy):
    """K-means clustering strategy for reference point clustering."""
    
    def __init__(self, config: ClusteringConfig):
        """Initialize K-means clustering strategy.
        
        Args:
            config: Configuration object containing clustering parameters
        """
        super().__init__(config)
        self.kmeans_model: Optional[KMeans] = None
        self.cluster_centers: Optional[np.ndarray] = None
        
        logger.info(f"K-means clustering initialized with {config.kmeans_config.n_clusters} clusters")
    
    def fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit K-means clustering to reference embeddings.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points
            
        Returns:
            ClusteringResult object with clustering information
        """
        logger.info(f"Fitting K-means clustering with {len(reference_indices)} reference points")
        
        # Extract reference embeddings
        reference_embeddings = embeddings[reference_indices]
        
        # Adjust number of clusters if we have fewer reference points
        n_clusters = min(self.config.kmeans_config.n_clusters, len(reference_indices))
        if n_clusters < self.config.kmeans_config.n_clusters:
            logger.warning(f"Reducing number of clusters from {self.config.kmeans_config.n_clusters} to {n_clusters} "
                          f"due to insufficient reference points")
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.kmeans_config.random_state,
            max_iter=self.config.kmeans_config.max_iter,
            tol=self.config.kmeans_config.tol,
            init="k-means++"
        )
        
        cluster_labels = self.kmeans_model.fit_predict(reference_embeddings)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        
        # Create cluster data structures
        cluster_data_list = []
        for cluster_id in range(n_clusters):
            # Find reference points in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            cluster_ref_indices = reference_indices[cluster_mask]
            cluster_ref_embeddings = reference_embeddings[cluster_mask]
            
            # Create cluster data using new structure
            cluster_data = ClusterData(
                reference_indices=cluster_ref_indices.tolist(),
                linked_target_indices=[],  # Will be populated later
                reference_embeddings=cluster_ref_embeddings.copy(),
                center_embedding=self.cluster_centers[cluster_id].copy() if self.cluster_centers is not None else None,
                linked_target_embeddings=None
            )
            cluster_data_list.append(cluster_data)
        
        # Create clustering result
        clustering_result = ClusteringResult(
            cluster_data_list=cluster_data_list,
            cluster_labels=cluster_labels,
            cluster_centers=self.cluster_centers.copy() if self.cluster_centers is not None else None,
            metadata={
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'inertia': float(self.kmeans_model.inertia_) if self.kmeans_model.inertia_ is not None else 0.0,
                'n_iter': int(self.kmeans_model.n_iter_) if self.kmeans_model.n_iter_ is not None else 0,
                'reference_size': len(reference_indices)
            }
        )
        
        # Compute quality metrics if requested
        if self.config.compute_metrics:
            clustering_result.quality_metrics = self.compute_quality_metrics(embeddings, clustering_result)
        
        self.is_fitted = True
        
        logger.info(f"K-means clustering completed. "
                   f"Cluster sizes: {[len(cluster.reference_indices) for cluster in cluster_data_list]}")
        
        return clustering_result
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings.
        
        Args:
            embeddings: New embeddings to assign to clusters
            
        Returns:
            Cluster assignments for each embedding
        """
        if not self.is_fitted or self.kmeans_model is None:
            raise ValueError("Must call fit() first")
        
        return self.kmeans_model.predict(embeddings)  # type: ignore
    
    def _fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Private fit method (delegates to public fit method).
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points
            
        Returns:
            ClusteringResult object with clustering information
        """
        return self.fit(embeddings, reference_indices)
    
    def _predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Private predict method (delegates to public predict method).
        
        Args:
            embeddings: New embeddings to assign to clusters
            
        Returns:
            Cluster assignments for each embedding
        """
        return self.predict(embeddings)
    
    def fit_predict(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> Tuple[ClusteringResult, np.ndarray]:
        """Fit K-means clustering and predict cluster assignments for all embeddings.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points
            
        Returns:
            Tuple of (ClusteringResult, cluster_assignments_for_all_embeddings)
        """
        # Fit the model
        clustering_result = self.fit(embeddings, reference_indices)
        
        # Predict assignments for all embeddings
        all_assignments = self.predict(embeddings)
        
        return clustering_result, all_assignments
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the fitted K-means model.
        
        Returns:
            Dictionary with cluster information
        """
        if self.kmeans_model is None:
            return {}
        
        return {
            'n_clusters': self.kmeans_model.n_clusters,
            'inertia': float(self.kmeans_model.inertia_)  # type: ignore
            if self.kmeans_model.inertia_ is not None else 0.0,
            'n_iter': int(self.kmeans_model.n_iter_)  # type: ignore
            if self.kmeans_model.n_iter_ is not None else 0,
            'cluster_centers_shape': self.cluster_centers.shape if self.cluster_centers is not None else None
        }
    
    def _compute_silhouette_score(self, embeddings: np.ndarray, 
                                 cluster_data_list: List[ClusterData]) -> Optional[float]:
        """Compute silhouette score for the clustering.
        
        Args:
            embeddings: All embeddings
            cluster_data_list: List of cluster data
            
        Returns:
            Silhouette score or None if computation fails
        """
        try:
            from sklearn.metrics import silhouette_score
            
            # Collect all reference points and their cluster labels
            all_ref_indices = []
            all_labels = []
            
            for cluster_id, cluster_data in enumerate(cluster_data_list):
                for ref_idx in cluster_data.reference_indices:
                    all_ref_indices.append(ref_idx)
                    all_labels.append(cluster_id)
            
            if len(all_ref_indices) < 2:
                return None
                
            # Compute silhouette score
            ref_embeddings = embeddings[all_ref_indices]
            score = silhouette_score(ref_embeddings, all_labels)
            
            return float(score)
            
        except Exception as e:
            logger.warning(f"Failed to compute silhouette score: {e}")
            return None
    
    def compute_quality_metrics(self, embeddings: np.ndarray, clustering_result: ClusteringResult) -> Dict[str, float]:
        """Compute quality metrics for the clustering.
        
        Args:
            embeddings: All embeddings
            clustering_result: Clustering result
            
        Returns:
            Dictionary with quality metrics
        """
        # Get base metrics
        base_metrics = super().compute_quality_metrics(embeddings, clustering_result)
        
        # Add K-means specific metrics
        kmeans_metrics = {}
        
        # Inertia (within-cluster sum of squares)
        if self.kmeans_model is not None:
            kmeans_metrics['inertia'] = float(self.kmeans_model.inertia_)  # type: ignore
            kmeans_metrics['n_iter'] = int(self.kmeans_model.n_iter_)  # type: ignore
        
        # Silhouette score
        silhouette_score = self._compute_silhouette_score(embeddings, clustering_result.cluster_data_list)
        if silhouette_score is not None:
            kmeans_metrics['silhouette_score'] = silhouette_score
        
        # Combine all metrics
        all_metrics = {**base_metrics, **kmeans_metrics}
        
        return all_metrics 