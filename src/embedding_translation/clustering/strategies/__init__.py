"""
Clustering strategy implementations for VectorMerge.

This module provides concrete implementations of different clustering strategies
including K-means and LA2M clustering.
"""

from .kmeans import KMeansClusteringStrategy
from .la2m_cluster import LA2MClusteringStrategy
SUPPORTED_CLUSTERING_METHODS = ["kmeans", "la2m-cluster"]

__all__ = [
    'KMeansClusteringStrategy',
    'LA2MClusteringStrategy',
    'SUPPORTED_CLUSTERING_METHODS'
] 