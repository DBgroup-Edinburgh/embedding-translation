"""
Expert clustering and assignment module.

This module handles clustering of reference points and assignment of experts
to cluster centroids for localized vector translation.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, TYPE_CHECKING
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from src.embeddings.memmap_dataset import MultiMemmapDatasetLoader


class ExpertClusterer:
    """
    Expert clustering and assignment system for localized vector translation.
    
    This class clusters reference points and assigns experts to cluster centroids,
    enabling localized modeling with small K values (e.g., 8 or 16).
    """
    
    def __init__(self, 
                 num_experts: int = 8,
                 clustering_method: str = "kmeans",
                 distance_metric: str = "cosine",
                 random_state: int = 42,
                 max_iter: int = 300,
                 n_init: int = 10):
        """
        Initialize the expert clusterer.
        
        Args:
            num_experts: Number of experts (K value), typically small (8 or 16)
            clustering_method: Clustering algorithm to use ("kmeans", "mini_batch_kmeans")
            distance_metric: Distance metric for clustering ("cosine", "euclidean")
            random_state: Random state for reproducibility
            max_iter: Maximum iterations for clustering
            n_init: Number of initialization attempts for clustering
        """
        self.num_experts = num_experts
        self.clustering_method = clustering_method
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        
        # Clustering results
        self.centroids: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.expert_assignments: Optional[Dict[int, List[int]]] = None
        
        logger.info(f"Initialized ExpertClusterer with {num_experts} experts, "
                   f"method={clustering_method}, metric={distance_metric}")
    
    def stream_cluster_references(self, 
                                  train_loader: 'MultiMemmapDatasetLoader',
                                  batch_size: int = 10000) -> Dict[str, Any]:
        """
        Stream cluster reference points using incremental clustering on the entire dataset.
        
        Uses MiniBatchKMeans with partial_fit for incremental learning:
        - Processes data in batches without loading all data into memory
        - Works on the full dataset, no sampling
        - Memory efficient for large datasets
        
        Args:
            train_loader: MultiMemmapDatasetLoader instance for loading training data
            batch_size: Batch size for incremental clustering (default: 10000)
            
        Returns:
            Dictionary containing clustering results
        """
        from sklearn.cluster import MiniBatchKMeans
        
        total_samples = train_loader.total_samples
        actual_num_clusters = min(self.num_experts, total_samples)
        
        logger.info(f"Incremental clustering on ALL {total_samples} samples (batch_size={batch_size})")
        
        if actual_num_clusters < self.num_experts:
            logger.warning(f"Reducing number of clusters from {self.num_experts} "
                          f"to {actual_num_clusters} due to insufficient samples")
        
        # Initialize MiniBatchKMeans for incremental learning
        clusterer = MiniBatchKMeans(
            n_clusters=actual_num_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            batch_size=min(batch_size, 1024),  # Internal batch size for algorithm
            verbose=0,
            compute_labels=False,  # Don't compute labels during fit
            init='k-means++',
            n_init=3,
            reassignment_ratio=0.01
        )
        
        # Stream through data and incrementally update clusters
        logger.info(f"Streaming through {total_samples} samples in batches of {batch_size}...")
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, total_samples, batch_size), 
                     total=num_batches,
                     desc="Incremental clustering"):
            end_idx = min(i + batch_size, total_samples)
            src_batch = train_loader.load_batch(i, end_idx, return_target=False)
            
            # Incrementally update cluster centroids
            clusterer.partial_fit(src_batch)
        
        # Get final centroids
        centroids = clusterer.cluster_centers_
        self.centroids = centroids
        
        logger.info(f"Incremental clustering completed with {len(centroids)} centroids")
        logger.info("Now assigning all samples to final clusters...")
        
        # Assign all samples to clusters (in batches to save memory)
        all_labels = []
        for i in tqdm(range(0, total_samples, batch_size), 
                     total=num_batches,
                     desc="Assigning labels"):
            end_idx = min(i + batch_size, total_samples)
            src_batch = train_loader.load_batch(i, end_idx, return_target=False)
            batch_labels = clusterer.predict(src_batch)
            all_labels.append(batch_labels)
        
        cluster_labels = np.concatenate(all_labels, axis=0)
        self.cluster_labels = cluster_labels
        
        # Create expert assignments mapping
        self.expert_assignments = {}
        cluster_sizes = []
        
        for expert_id in range(actual_num_clusters):
            cluster_indices = np.where(cluster_labels == expert_id)[0]
            self.expert_assignments[expert_id] = cluster_indices.tolist()
            cluster_sizes.append(len(cluster_indices))
        
        # Log clustering statistics
        logger.info(f"Clustering completed. Cluster sizes: {cluster_sizes}")
        logger.info(f"Min cluster size: {min(cluster_sizes)}, "
                   f"Max cluster size: {max(cluster_sizes)}, "
                   f"Mean cluster size: {np.mean(cluster_sizes):.2f}")
        
        return {
            "centroids": centroids,
            "cluster_labels": cluster_labels,
            "expert_assignments": self.expert_assignments,
            "cluster_sizes": cluster_sizes,
            "num_clusters": actual_num_clusters
        }
    
    def cluster_references(self, 
                          reference_data: Union[np.ndarray, 'MultiMemmapDatasetLoader'],
                          batch_size: int = 10000) -> Dict[str, Any]:
        """
        Cluster reference points and return centroids and assignments.
        
        Accepts either:
        1. NumPy array: Direct clustering on provided embeddings
        2. MultiMemmapDatasetLoader: Incremental clustering on all data
        
        Args:
            reference_data: Either reference embeddings (N x D) or MultiMemmapDatasetLoader
            batch_size: Batch size for incremental clustering (used with loader)
            
        Returns:
            Dictionary containing clustering results:
            - centroids: Cluster centroids (K x D)
            - cluster_labels: Assignment of each reference to cluster (N,)
            - expert_assignments: Mapping from expert_id to reference indices
            - cluster_sizes: Number of references per cluster
        """
        if isinstance(reference_data, np.ndarray):
            # Direct clustering on numpy array
            reference_embeddings = reference_data
            logger.info(f"Clustering {reference_embeddings.shape[0]} reference points "
                       f"into {self.num_experts} clusters")
            return self._perform_clustering(reference_embeddings)
        else:
            # Incremental clustering on MultiMemmapDatasetLoader
            logger.info("Detected MultiMemmapDatasetLoader, using incremental clustering...")
            return self.stream_cluster_references(reference_data, batch_size=batch_size)
    
    def _perform_clustering(self, reference_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Perform clustering on a given set of embeddings.
        
        Args:
            reference_embeddings: Reference embeddings to cluster (N x D)
            
        Returns:
            Dictionary containing clustering results
        """
        logger.info(f"Performing {self.clustering_method} clustering on "
                   f"{reference_embeddings.shape[0]} embeddings")
        
        # Ensure we don't have more clusters than references
        actual_num_clusters = min(self.num_experts, reference_embeddings.shape[0])
        
        if actual_num_clusters < self.num_experts:
            logger.warning(f"Reducing number of clusters from {self.num_experts} "
                          f"to {actual_num_clusters} due to insufficient references")
        
        # Perform clustering
        if self.clustering_method == "kmeans":
            clusterer = KMeans(
                n_clusters=actual_num_clusters,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init,
                algorithm='lloyd'
            )
        elif self.clustering_method == "mini_batch_kmeans":
            from sklearn.cluster import MiniBatchKMeans
            clusterer = MiniBatchKMeans(
                n_clusters=actual_num_clusters,
                random_state=self.random_state,
                max_iter=self.max_iter,
                batch_size=1000
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        # Fit clustering model
        cluster_labels = clusterer.fit_predict(reference_embeddings)
        centroids = clusterer.cluster_centers_
        
        # Store results
        self.centroids = centroids
        self.cluster_labels = cluster_labels
        
        # Create expert assignments mapping
        self.expert_assignments = {}
        cluster_sizes = []
        
        for expert_id in range(actual_num_clusters):
            cluster_indices = np.where(cluster_labels == expert_id)[0]
            self.expert_assignments[expert_id] = cluster_indices.tolist()
            cluster_sizes.append(len(cluster_indices))
        
        # Log clustering statistics
        logger.info(f"Clustering completed. Cluster sizes: {cluster_sizes}")
        logger.info(f"Min cluster size: {min(cluster_sizes)}, "
                   f"Max cluster size: {max(cluster_sizes)}, "
                   f"Mean cluster size: {np.mean(cluster_sizes):.2f}")
        
        return {
            "centroids": centroids,
            "cluster_labels": cluster_labels,
            "expert_assignments": self.expert_assignments,
            "cluster_sizes": cluster_sizes,
            "num_clusters": actual_num_clusters
        }
    
    def assign_to_expert(self, 
                        query_embedding: np.ndarray) -> Tuple[int, float]:
        """
        Assign a query vector to the nearest expert.
        
        Args:
            query_embedding: Query vector to assign (D,)
            
        Returns:
            Tuple of (expert_id, distance_to_centroid)
        """
        if self.centroids is None:
            raise ValueError("Must call cluster_references() first")
        
        # Reshape for distance computation
        query_reshaped = query_embedding.reshape(1, -1)
        
        # Compute distances to all centroids
        if self.distance_metric == "cosine":
            distances = cosine_distances(query_reshaped, self.centroids)[0]
        elif self.distance_metric == "euclidean":
            distances = euclidean_distances(query_reshaped, self.centroids)[0]
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Find nearest centroid
        expert_id = np.argmin(distances)
        min_distance = distances[expert_id]
        
        return expert_id, min_distance
    
    def batch_assign_to_experts(self, 
                               query_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign multiple query vectors to experts in batch.
        
        Args:
            query_embeddings: Query vectors to assign (N x D)
            
        Returns:
            Tuple of (expert_ids, distances) for each query
        """
        if self.centroids is None:
            raise ValueError("Must call cluster_references() first")
        
        # Compute distances to all centroids
        if self.distance_metric == "cosine":
            distances = cosine_distances(query_embeddings, self.centroids)
        elif self.distance_metric == "euclidean":
            distances = euclidean_distances(query_embeddings, self.centroids)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Find nearest centroids
        expert_ids = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        return expert_ids, min_distances
    
    def get_expert_references(self, expert_id: int) -> List[int]:
        """
        Get reference indices assigned to a specific expert.
        
        Args:
            expert_id: ID of the expert
            
        Returns:
            List of reference indices assigned to this expert
        """
        if self.expert_assignments is None:
            raise ValueError("Must call cluster_references() first")
        
        if expert_id not in self.expert_assignments:
            raise ValueError(f"Expert {expert_id} not found")
        
        return self.expert_assignments[expert_id]
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clustering results.
        
        Returns:
            Dictionary containing clustering statistics
        """
        if self.expert_assignments is None:
            raise ValueError("Must call cluster_references() first")
        
        cluster_sizes = [len(indices) for indices in self.expert_assignments.values()]
        
        return {
            "num_experts": len(self.expert_assignments),
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "mean_cluster_size": np.mean(cluster_sizes),
            "std_cluster_size": np.std(cluster_sizes),
            "total_references": sum(cluster_sizes)
        }
