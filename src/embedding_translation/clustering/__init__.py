"""Clustering package.

Provides clustering strategies used both for reference-anchor selection and as
the gating layer for the hierarchical-MoE mapper.

Quick usage:

    from embedding_translation.clustering import ClusterManager
    from embedding_translation.config import ClusteringConfig

    manager = ClusterManager(
        dataset_name="scifact",
        model="mistral",
        reference_key="la2m_split_scifact_0.50",
        reference_path="./data/processed/references",
        cluster_path="./data/processed/clusters",
        embedding_path="./.cache/embeddings",
        strategy_name="kmeans",
        strategy_config=ClusteringConfig(clustering_method="kmeans"),
    )
"""

from .base import ClusterData, ClusteringResult, ClusteringStrategy
from .manage import ClusterManager
from .strategies import (
    SUPPORTED_CLUSTERING_METHODS,
    KMeansClusteringStrategy,
    LA2MClusteringStrategy,
)
from .utils import (
    compute_cluster_metrics,
    load_cluster_data,
    save_cluster_data,
    visualize_clusters,
)

__all__ = [
    "ClusteringStrategy",
    "ClusteringResult",
    "ClusterManager",
    "ClusterData",
    "SUPPORTED_CLUSTERING_METHODS",
    "KMeansClusteringStrategy",
    "LA2MClusteringStrategy",
    "load_cluster_data",
    "save_cluster_data",
    "compute_cluster_metrics",
    "visualize_clusters",
]
