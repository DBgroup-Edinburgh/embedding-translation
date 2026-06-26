"""
LA2M clustering strategy implementation.

This module implements the clustering strategy used in LA2M (Local Anchor to Multiple)
where each point has a neighborhood of closest reference points, and these neighborhoods
are optimized and merged to form final clusters.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm

from ..base import ClusteringStrategy, ClusteringResult, ClusterData
from ...config import ClusteringConfig

logger = logging.getLogger(__name__)


class LA2MClusteringStrategy(ClusteringStrategy):
    """LA2M clustering strategy.
    
    This strategy creates neighborhoods around each point based on nearest neighbors.
    """
    
    def __init__(self, config: ClusteringConfig):
        super().__init__(config)
        self.d_prime = config.la2m_config.d_prime
        self.is_fitted = False
        
    def fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit clustering strategy to reference embeddings."""
        return self._fit(embeddings, reference_indices)
    
    def _fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit the LA2M clustering strategy."""
        logger.info(f"Starting LA2M clustering with d_prime={self.d_prime}")
        
        # Convert to torch tensors
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        reference_embeddings = embeddings_tensor[reference_indices]
        
        # Create neighborhoods for each reference point
        neighborhoods = self._compute_neighborhoods(reference_embeddings, reference_indices)
        
        # Create clustering result
        result = self._create_clustering_result(neighborhoods, embeddings, reference_indices)
        
        self.is_fitted = True
        logger.info(f"LA2M clustering completed with {len(neighborhoods)} neighborhoods")
        
        return result
    
    def _predict(self, result: ClusteringResult, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings."""
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before prediction")
    
        if not hasattr(self, 'cluster_centers_') or self.cluster_centers_ is None:
            self.cluster_centers_ = self._concat_clustering_centers(result.cluster_data_list)
        
        # For simplicity, assign each point to the nearest cluster center
        if hasattr(self, 'cluster_centers_') and self.cluster_centers_ is not None:
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            centers_tensor = torch.tensor(self.cluster_centers_, dtype=torch.float32)
            
            # Compute distances to all centers
            distances = torch.cdist(embeddings_tensor, centers_tensor)
            
            # Assign to nearest center
            cluster_assignments = torch.argmin(distances, dim=1)
            
            return cluster_assignments.numpy()
        else:
            # Fallback: assign all to cluster 0
            raise ValueError("Strategy must be fitted before prediction")
    
    def predict(self, clustering_result: ClusteringResult, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings."""
        return self._predict(clustering_result, embeddings)

    def _compute_neighborhoods(self, embeddings: torch.Tensor, 
                             reference_indices: np.ndarray) -> List[ClusterData]:
        """Compute neighborhoods for individual points."""
        neighborhoods = []
        
        # Compute reference-to-reference distances
        distances = torch.cdist(embeddings, embeddings)
        
        for i in range(len(embeddings)):
            # Get top-d neighbors (including the point itself)
            point_distances = distances[i]
            top_indices = torch.argsort(point_distances)[:self.d_prime]
            
            # Extract embeddings for this neighborhood
            neighborhood_embeddings = embeddings[top_indices].cpu().numpy()
            
            # Create neighborhood cluster data
            cluster_data = ClusterData(
                reference_indices=[int(reference_indices[j]) for j in top_indices],
                linked_target_indices=[],
                reference_embeddings=neighborhood_embeddings.copy(),
                center_embedding=np.mean(neighborhood_embeddings, axis=0) if len(neighborhood_embeddings) > 0 else None,
                linked_target_embeddings=None
            )
            
            neighborhoods.append(cluster_data)
        
        return neighborhoods
    
    
    def _create_clustering_result(self, clusters: List[ClusterData], 
                                 embeddings: np.ndarray,
                                 reference_indices: np.ndarray) -> ClusteringResult:
        """Create ClusteringResult from neighborhoods."""
        # Create cluster labels
        cluster_labels = np.arange(len(reference_indices), dtype=int)
        
        # Compute cluster centers
        cluster_centers = []
        for cluster_data in clusters:
            if cluster_data.center_embedding is not None:
                cluster_centers.append(cluster_data.center_embedding)
            else:
                # Fallback: use first reference point embedding
                if cluster_data.reference_indices:
                    cluster_centers.append(embeddings[cluster_data.reference_indices[0]])
                else:
                    cluster_centers.append(np.zeros(embeddings.shape[1]))
        
        cluster_centers = np.array(cluster_centers) if cluster_centers else None
        self.cluster_centers_ = cluster_centers
        
        return ClusteringResult(
            cluster_data_list=clusters,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            metadata={
                'method': 'la2m-cluster',
                'd_prime': self.config.la2m_config.d_prime,
                'num_reference_points': len(reference_indices)
            }
        ) 