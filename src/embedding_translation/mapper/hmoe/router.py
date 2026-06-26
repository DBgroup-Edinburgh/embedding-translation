"""
Routers for expert selection in MoE systems.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


class BaseRouter(ABC):
    """Base router class."""
    
    @abstractmethod
    def route(self, embeddings: np.ndarray, **kwargs):
        """Route embeddings to experts."""
        pass


class FlatRouter(BaseRouter):
    """Flat router for single-level MoE."""
    
    def __init__(self, centroids: np.ndarray, distance_metric: str = "cosine"):
        self.centroids = centroids
        self.distance_metric = distance_metric
    
    def route(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route embeddings to nearest expert.
        
        Args:
            embeddings: Input embeddings (N x D)
            
        Returns:
            Tuple of (expert_ids, distances) where:
            - expert_ids: Assigned expert ID for each embedding (N,)
            - distances: Distance to assigned expert centroid (N,)
        """
        # Compute distances to all centroids
        if self.distance_metric == "cosine":
            distances = cosine_distances(embeddings, self.centroids)
        elif self.distance_metric == "euclidean":
            distances = euclidean_distances(embeddings, self.centroids)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Find nearest centroid for each embedding
        expert_ids = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        return expert_ids, min_distances


class CascadeRouter(BaseRouter):
    """Cascade router for hierarchical MoE."""
    
    def __init__(self, tree, distance_metric: str = "cosine"):
        self.tree = tree
        self.distance_metric = distance_metric
    
    def route(self, embedding: np.ndarray):
        """Route from root to leaf node."""
        pass

