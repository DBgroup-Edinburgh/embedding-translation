"""Base classes and interfaces for clustering.

ClusteringConfig, KMeansConfig, and LA2MClusteringConfig now live in
`embedding_translation.config.models` as pydantic models. ClusterData and
ClusteringResult remain dataclasses because they carry numpy arrays and
mutate state — they are internal data carriers, not user-facing configs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import joblib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from loguru import logger

from ..config import ClusteringConfig, KMeansConfig, LA2MClusteringConfig

@dataclass
class ClusterData:
    """Data structure to store cluster information."""
    
    reference_indices: List[int] = field(default_factory=list) 
    linked_target_indices: List[int] = field(default_factory=list)
    reference_embeddings: Optional[np.ndarray] = None
    center_embedding: Optional[np.ndarray] = None
    linked_target_embeddings: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        """Return total number of points in cluster."""
        return len(self.reference_indices) + len(self.linked_target_indices)
    
    @property
    def ref_size(self) -> int:
        """Return number of reference points."""
        return len(self.reference_indices)
    
    @property
    def bound_size(self) -> int:
        """Return number of bound points."""
        return len(self.linked_target_indices)
    
    @property
    def total_size(self) -> int:
        """Return total number of points in cluster."""
        return len(self)
    
    @property
    def has_center(self) -> bool:
        """Check if cluster has a center."""
        return self.center_embedding is not None
    
    def compute_center(self, embeddings: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Compute cluster center from reference embeddings."""
        if self.reference_embeddings is None or len(self.reference_embeddings) == 0:
            return None
        if embeddings is None:
            embeddings = self.reference_embeddings
        
        center = np.mean(embeddings, axis=0)
        self.center_embedding = center
        return center
    
    def add_reference_point(self, index: int, embedding: Optional[np.ndarray] = None) -> None:
        """Add a reference point to the cluster."""
        if index not in self.reference_indices:
            self.reference_indices.append(index)
            if embedding is not None:
                if self.reference_embeddings is None:
                    self.reference_embeddings = embedding.reshape(1, -1)
                else:
                    self.reference_embeddings = np.vstack([self.reference_embeddings, embedding.reshape(1, -1)])
    
    def add_target_point(self, index: int, embedding: Optional[np.ndarray] = None) -> None:
        """Add a target point to the cluster."""
        if index not in self.linked_target_indices:
            self.linked_target_indices.append(index)
            if embedding is not None:
                if self.linked_target_embeddings is None:
                    self.linked_target_embeddings = embedding.reshape(1, -1)
                else:
                    self.linked_target_embeddings = np.vstack([self.linked_target_embeddings, embedding.reshape(1, -1)])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster data to dictionary for serialization.
        
        Note: Embeddings are not serialized to save space.
        They can be recomputed from the original data when needed.
        """
        return {
            'reference_indices': self.reference_indices,
            'linked_target_indices': self.linked_target_indices,
            'total_size': self.total_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterData':
        """Create cluster data from dictionary.
        
        Note: Embeddings are not loaded and need to be computed separately.
        """
        return cls(
            reference_indices=data.get('reference_indices', []),
            linked_target_indices=data.get('linked_target_indices', []),
            reference_embeddings=None,
            center_embedding=None,
            linked_target_embeddings=None
        )
    
    # Backward compatibility properties
    @property
    def ref_index(self) -> List[int]:
        """Backward compatibility: return reference indices."""
        return self.reference_indices
    
    @property
    def bound_index(self) -> List[int]:
        """Backward compatibility: return linked target indices."""
        return self.linked_target_indices
    
    @property
    def center(self) -> Optional[np.ndarray]:
        """Backward compatibility: return center embedding."""
        return self.center_embedding



@dataclass
class ClusteringResult:
    """Result container for clustering operations."""
    
    cluster_data_list: List[ClusterData]
    cluster_labels: np.ndarray
    cluster_centers: Optional[np.ndarray] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_clusters(self) -> int:
        """Return number of clusters."""
        return len(self.cluster_data_list)
    
    @property
    def has_embeddings(self) -> bool:
        """Check if cluster result has embeddings."""
        return all(cluster.reference_embeddings is not None for cluster in self.cluster_data_list)
    
    def reload_reference_embeddings(self, embeddings: np.ndarray) -> None:
        """Recompute cluster centers."""

        for index, cluster in enumerate(self.cluster_data_list):
            self.cluster_data_list[index].reference_embeddings = embeddings[cluster.reference_indices]
            self.cluster_data_list[index].compute_center()

        # for cluster in self.cluster_data_list:
        #     if cluster.center_embedding is None:
        #         cluster.compute_center(embeddings=embeddings[cluster.reference_indices])
        
        self.cluster_centers = self.get_cluster_centers()
    
    @property
    def cluster_sizes(self) -> List[int]:
        """Return sizes of all clusters."""
        return [len(cluster.reference_indices) for cluster in self.cluster_data_list]
    
    @property
    def reference_cluster_sizes(self) -> List[int]:
        """Return sizes of reference points in each cluster."""
        return [cluster.ref_size for cluster in self.cluster_data_list]
    
    @property
    def target_cluster_sizes(self) -> List[int]:
        """Return sizes of target points in each cluster."""
        return [cluster.bound_size for cluster in self.cluster_data_list]
    
    @property
    def total_reference_points(self) -> int:
        """Return total number of reference points across all clusters."""
        reference_indices = []
        for cluster in self.cluster_data_list:
            reference_indices.extend(cluster.reference_indices)
        return len(set(reference_indices))
    
    @property
    def total_target_points(self) -> int:
        """Return total number of target points across all clusters."""
        return sum(cluster.bound_size for cluster in self.cluster_data_list)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers from cluster data."""
        centers = []
        for cluster in self.cluster_data_list:
            if cluster.center_embedding is not None:
                centers.append(cluster.center_embedding)
            elif cluster.reference_embeddings is not None and len(cluster.reference_embeddings) > 0:
                # Compute center if not available
                center = np.mean(cluster.reference_embeddings, axis=0)
                centers.append(center)
            else:
                return None  # Cannot compute centers
        
        return np.array(centers) if centers else None
    
    def compute_all_centers(self) -> bool:
        """Compute centers for all clusters that don't have them."""
        success = True
        for cluster in self.cluster_data_list:
            if cluster.center_embedding is None:
                computed_center = cluster.compute_center()
                if computed_center is None:
                    success = False
        
        # Update cluster_centers array
        if success:
            self.cluster_centers = self.get_cluster_centers()
        
        return success
    
    def save(self, path: Path, save_embeddings: bool = False) -> None:
        """Save the clustering result to a file."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        cluster_data_list = self.cluster_data_list
        if not save_embeddings:
            for index, _ in enumerate(cluster_data_list):
                cluster_data_list[index].reference_embeddings = None
                cluster_data_list[index].linked_target_embeddings = None
                cluster_data_list[index].center_embedding = None
        self.cluster_data_list = cluster_data_list
        joblib.dump(self, save_path / "cluster.pkl")
        logger.info(f"Saved clustering result to {save_path / 'cluster.pkl'}")
    
    @classmethod
    def load(cls, path: Path) -> 'ClusteringResult':
        """Load the clustering result from a file."""
        load_path = Path(path)
        data = joblib.load(load_path / "cluster.pkl")
        return data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'cluster_data_list': [cluster.to_dict() for cluster in self.cluster_data_list],
            'cluster_labels': self.cluster_labels.tolist(),
            'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None,
            'quality_metrics': self.quality_metrics,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringResult':
        """Create result from dictionary."""
        cluster_data_list = [ClusterData.from_dict(cluster_data) for cluster_data in data['cluster_data_list']]
        cluster_labels = np.array(data['cluster_labels'])
        cluster_centers = np.array(data['cluster_centers']) if data.get('cluster_centers') is not None else None
        
        return cls(
            cluster_data_list=cluster_data_list,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            quality_metrics=data.get('quality_metrics', {}),
            metadata=data.get('metadata', {})
        )


class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""
    
    def __init__(self, config: ClusteringConfig):
        """Initialize clustering strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" 
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.is_fitted = False
        
        logger.info(f"Initialized {self.__class__.__name__} with {config.clustering_method} clustering method")

    def predict(self, clustering_result: ClusteringResult, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings.
        
        Args:
            clustering_result: Clustering result to use for prediction
            embeddings: New embeddings to assign to clusters

        Returns:
            Cluster assignments for each embedding
        """
        return self._predict(clustering_result, embeddings)
        

    
    @abstractmethod
    def fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit the clustering strategy to the data.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points to cluster
            
        Returns:
            ClusteringResult object with clustering information
        """
        pass
    
    @abstractmethod
    def _fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit the clustering strategy to the data.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points to cluster
            
        Returns:
            ClusteringResult object with clustering information
        """
        pass

    def _concat_clustering_centers(self, clusters: List[ClusterData]) -> np.ndarray:
        """Concatenate clustering centers."""
        centers = []
        for cluster in clusters:
            if cluster.center_embedding is not None:
                centers.append(cluster.center_embedding)
        return np.array(centers)

    def _predict(self, clustering_result: ClusteringResult, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings."""
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before prediction")
    
        if not hasattr(self, 'cluster_centers_') or self.cluster_centers_ is None:
            self.cluster_centers_ = self._concat_clustering_centers(clustering_result.cluster_data_list)
        
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
    
    def assign_to_clusters(self, embeddings: np.ndarray, target_indices: np.ndarray,
                          clustering_result: ClusteringResult) -> ClusteringResult:
        """Assign target points to existing clusters.
        
        Args:
            embeddings: All embeddings
            target_indices: Indices of target points to assign
            clustering_result: Existing clustering result
            
        Returns:
            Updated clustering result with assigned target points
        """
        # Get cluster centers
        cluster_centers = clustering_result.get_cluster_centers()
        if cluster_centers is None:
            raise ValueError("Cannot assign points without cluster centers")
        
        logger.info(f"Assigning {len(target_indices)} target points to clusters")
        
        # Extract target embeddings
        target_embeddings = embeddings[target_indices]
        
        # Predict cluster assignments
        cluster_assignments = self.predict(clustering_result, target_embeddings)
        
        # Update cluster data with assigned target points
        updated_clusters = []
        for cluster_id, cluster_data in enumerate(clustering_result.cluster_data_list):
            # Find target points assigned to this cluster
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_target_indices = target_indices[cluster_mask]
            cluster_target_embeddings = target_embeddings[cluster_mask] if len(cluster_target_indices) > 0 else None
            
            # Create updated cluster data
            updated_cluster = ClusterData(
                reference_indices=cluster_data.reference_indices.copy(),
                linked_target_indices=cluster_target_indices.tolist(),
                reference_embeddings=cluster_data.reference_embeddings.copy() if cluster_data.reference_embeddings is not None else None,
                center_embedding=cluster_data.center_embedding.copy() if cluster_data.center_embedding is not None else None,
                linked_target_embeddings=cluster_target_embeddings.copy() if cluster_target_embeddings is not None else None
            )
            updated_clusters.append(updated_cluster)
        
        # Create updated result
        updated_result = ClusteringResult(
            cluster_data_list=updated_clusters,
            cluster_labels=clustering_result.cluster_labels,
            cluster_centers=clustering_result.cluster_centers,
            quality_metrics=clustering_result.quality_metrics,
            metadata=clustering_result.metadata
        )
        
        # Log assignment statistics
        assignment_counts = [len(cluster.linked_target_indices) for cluster in updated_clusters]
        logger.info(f"Target point assignment completed. Points per cluster: {assignment_counts}")
        
        return updated_result
    
    def compute_quality_metrics(self, embeddings: np.ndarray, clustering_result: ClusteringResult) -> Dict[str, float]:
        """Compute quality metrics for the clustering.
        
        Args:
            embeddings: All embeddings
            clustering_result: Clustering result
            
        Returns:
            Dictionary with quality metrics
        """
        if not clustering_result.cluster_data_list:
            return {}
        
        # Compute within-cluster sum of squares (WCSS)
        total_wcss = 0.0
        cluster_wcss = []
        
        for cluster_data in clustering_result.cluster_data_list:
            if len(cluster_data.reference_indices) > 0:
                cluster_embeddings = embeddings[cluster_data.reference_indices]
                center = cluster_data.center_embedding
                
                if center is not None:
                    # Compute sum of squared distances to center
                    distances_sq = np.sum((cluster_embeddings - center) ** 2, axis=1)
                    wcss = np.sum(distances_sq)
                    cluster_wcss.append(wcss)
                    total_wcss += wcss
                else:
                    cluster_wcss.append(0.0)
            else:
                cluster_wcss.append(0.0)
        
        # Compute average cluster size
        cluster_sizes = [len(cluster.reference_indices) for cluster in clustering_result.cluster_data_list]
        avg_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        
        # Compute cluster size variance
        cluster_size_variance = float(np.var(cluster_sizes)) if cluster_sizes else 0.0
        
        return {
            'total_wcss': total_wcss,
            'avg_wcss': total_wcss / len(clustering_result.cluster_data_list) if clustering_result.cluster_data_list else 0,
            'avg_cluster_size': avg_cluster_size,
            'cluster_size_variance': cluster_size_variance,
            'num_clusters': len(clustering_result.cluster_data_list),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0
        }
    
    def save(self, path: Path, clustering_result: ClusteringResult) -> None:
        """Save the clustering result and strategy.
        
        Args:
            path: Path to save the clustering result
            clustering_result: Clustering result to save
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save clustering result
        import json
        with open(save_path / "clustering_result.json", "w") as f:
            json.dump(clustering_result.to_dict(), f, indent=2)
        
        # Save configuration
        with open(save_path / "clustering_config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)
        
        # Save cluster centers if available
        if clustering_result.cluster_centers is not None:
            np.save(save_path / "cluster_centers.npy", clustering_result.cluster_centers)
        
        logger.info(f"Saved clustering result to {save_path}")
    
    @classmethod
    def load(cls, path: Path) -> Tuple['ClusteringStrategy', ClusteringResult]:
        """Load clustering strategy and result.
        
        Args:
            path: Path to load from
            
        Returns:
            Tuple of (clustering_strategy, clustering_result)
        """
        load_path = Path(path)
        
        # Load configuration
        import json
        with open(load_path / "clustering_config.json", "r") as f:
            config_data = json.load(f)
        
        config = ClusteringConfig.model_validate(config_data)
        
        # Load clustering result
        with open(load_path / "clustering_result.json", "r") as f:
            result_data = json.load(f)
        
        clustering_result = ClusteringResult.from_dict(result_data)
        
        # Load cluster centers if available
        centers_path = load_path / "cluster_centers.npy"
        if centers_path.exists():
            clustering_result.cluster_centers = np.load(centers_path)
        
        # Create strategy instance
        strategy = cls(config)
        strategy.is_fitted = True
        
        return strategy, clustering_result 