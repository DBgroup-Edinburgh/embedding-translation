"""
Gating mechanism for routing inputs to experts.

This module implements the gating mechanism that routes input vectors
to the nearest expert based on cluster centroids.
"""

from typing import Tuple, Optional, Dict, Literal
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np
from loguru import logger
from pprint import pprint


# wandb stub — VT code logs liberally to wandb but we don't require it.
class _WandbStub:
    def log(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return self.log


wandb = _WandbStub()


class GatingMechanism:
    """
    Gating mechanism for routing input vectors to experts.
    
    This class implements the routing logic that assigns each input vector
    to its nearest expert based on cluster centroids.
    """
    
    def __init__(self, 
                 centroids: np.ndarray,
                 distance_metric: str = "cosine",
                 temperature: float = 1.0,
                 use_soft_routing: bool = False):
        """
        Initialize the gating mechanism.
        
        Args:
            centroids: Cluster centroids for expert routing (K x D)
            distance_metric: Distance metric for routing ("cosine", "euclidean")
            temperature: Temperature for soft routing (higher = more uniform)
            use_soft_routing: Whether to use soft routing instead of hard assignment
        """
        self.centroids = centroids
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.use_soft_routing = use_soft_routing
        self.num_experts = centroids.shape[0]
        
        logger.info(f"Initialized GatingMechanism with {self.num_experts} experts, "
                   f"metric={distance_metric}, soft_routing={use_soft_routing}")
    
    def route_input(self, 
                   input_vector: np.ndarray) -> Tuple[int, float, Optional[np.ndarray], Optional[dict]]:
        """
        Route a single input vector to the appropriate expert.
        
        Args:
            input_vector: Input vector to route (D,)
            
        Returns:
            Tuple of (expert_id, distance, routing_weights)
            - expert_id: ID of the assigned expert
            - distance: Distance to the assigned expert centroid
            - routing_weights: Soft routing weights (None if hard routing)
        """
        # Reshape for distance computation
        input_reshaped = input_vector.reshape(1, -1)
        
        # Compute distances to all centroids
        distances = self._compute_distances(input_reshaped, self.centroids)[0]
        
        if self.use_soft_routing:
            # Soft routing using temperature-scaled softmax
            routing_weights = self._compute_soft_weights(distances)
            expert_id = np.argmax(routing_weights)
            distance = distances[expert_id]
            return expert_id, distance, routing_weights, None
        else:
            # Hard routing to nearest centroid
            expert_id = np.argmin(distances)
            distance = distances[expert_id]
            return expert_id, distance, None, None
    
    
    def batch_route(self, 
                   input_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[dict]]:
        """
        Route multiple input vectors to experts in batch.
        
        Args:
            input_vectors: Input vectors to route (N x D)
            
        Returns:
            Tuple of (expert_ids, distances, routing_weights)
            - expert_ids: IDs of assigned experts for each input (N,)
            - distances: Distances to assigned experts (N,)
            - routing_weights: Soft routing weights (N x K) or None
        """
        # Compute distances to all centroids
        distances = self._compute_distances(input_vectors, self.centroids)
        
        if self.use_soft_routing:
            # Soft routing for all inputs
            routing_weights = np.array([
                self._compute_soft_weights(dist) for dist in distances
            ])
            expert_ids = np.argmax(routing_weights, axis=1)
            selected_distances = np.array([
                distances[i, expert_ids[i]] for i in range(len(expert_ids))
            ])
            stats = self.get_routing_statistics(input_vectors, expert_ids, selected_distances, routing_weights)
            return expert_ids, selected_distances, routing_weights, stats
        else:
            # Hard routing to nearest centroids
            expert_ids = np.argmin(distances, axis=1)
            selected_distances = np.array([
                distances[i, expert_ids[i]] for i in range(len(expert_ids))
            ])
            stats = self.get_routing_statistics(input_vectors, expert_ids, selected_distances, None)
            return expert_ids, selected_distances, None, stats
    
    def _compute_distances(self, 
                          input_vectors: np.ndarray, 
                          centroids: np.ndarray) -> np.ndarray:
        """
        Compute distances between input vectors and centroids.
        
        Args:
            input_vectors: Input vectors (N x D)
            centroids: Centroid vectors (K x D)
            
        Returns:
            Distance matrix (N x K)
        """
        if self.distance_metric == "cosine":
            return cosine_distances(input_vectors, centroids)
        elif self.distance_metric == "euclidean":
            return euclidean_distances(input_vectors, centroids)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _compute_soft_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute soft routing weights using temperature-scaled softmax.
        
        Args:
            distances: Distances to all centroids (K,)
            
        Returns:
            Soft routing weights (K,)
        """
        # Convert distances to similarities (negative distances)
        similarities = -distances
        
        # Apply temperature scaling
        scaled_similarities = similarities / self.temperature
        
        # Compute softmax
        exp_similarities = np.exp(scaled_similarities - np.max(scaled_similarities))
        weights = exp_similarities / np.sum(exp_similarities)
        
        return weights
    
    def compute_distance_based_distinguishability(self,
                                                   input_vectors: np.ndarray,
                                                   method: Literal["margin", "ratio", "gap_normalized", "std"] = "margin") -> Dict[str, np.ndarray]:
        """
        Compute sample distinguishability based on distance differences.
        
        Concept:
        - Large difference between min distance and others → sample clearly belongs to one expert (high distinguishability)
        - Small difference → sample is ambiguous between multiple experts (low distinguishability)
        
        Args:
            input_vectors: Input vectors (N x D)
            method: Distance difference computation method
                - "margin": second_min_distance - min_distance (larger margin = more distinguishable)
                - "ratio": second_min_distance / min_distance (larger ratio = more distinguishable)
                - "gap_normalized": (second_min - min) / min (normalized gap)
                - "std": standard deviation of distances (larger std = more distinguishable)
        
        Returns:
            Dictionary containing:
            - "scores": Distinguishability scores (N,), higher score = more distinguishable
            - "min_distances": Minimum distances (N,)
            - "second_min_distances": Second minimum distances (N,)
            - "best_expert_ids": Best expert IDs (N,)
            - "second_best_expert_ids": Second best expert IDs (N,)
            - "all_distances": All distances matrix (N x K)
        """
        # Compute distance matrix (N x K)
        distances = self._compute_distances(input_vectors, self.centroids)
        
        # Sort each row to find minimum and second minimum distances
        sorted_distances = np.sort(distances, axis=1)
        sorted_indices = np.argsort(distances, axis=1)
        
        min_distances = sorted_distances[:, 0]
        second_min_distances = sorted_distances[:, 1] if distances.shape[1] > 1 else min_distances
        
        best_expert_ids = sorted_indices[:, 0]
        second_best_expert_ids = sorted_indices[:, 1] if distances.shape[1] > 1 else sorted_indices[:, 0]
        
        # Compute distinguishability scores based on method
        if method == "margin":
            # Margin: second_min_distance - min_distance
            # Larger = more distinguishable (large distance gap)
            scores = second_min_distances - min_distances
            
        elif method == "ratio":
            # Ratio: second_min_distance / min_distance
            # Larger = more distinguishable (ratio indicates min distance is relatively small)
            eps = 1e-10
            scores = second_min_distances / (min_distances + eps)
            
        elif method == "gap_normalized":
            # Normalized gap: (second_min - min) / min
            # Larger = more distinguishable
            eps = 1e-10
            scores = (second_min_distances - min_distances) / (min_distances + eps)
            
        elif method == "std":
            # Standard deviation: std of distances
            # Larger = more distinguishable (some distances large, some small, clear distinction)
            scores = np.std(distances, axis=1)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return {
            "scores": scores,
            "min_distances": min_distances,
            "second_min_distances": second_min_distances,
            "best_expert_ids": best_expert_ids,
            "second_best_expert_ids": second_best_expert_ids,
            "all_distances": distances,
            "method": method
        }
    
    def get_distance_based_ranking(self,
                                   input_vectors: np.ndarray,
                                   method: Literal["margin", "ratio", "gap_normalized", "std"] = "margin",
                                   return_top_k: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get sample ranking results based on distance differences.
        
        Args:
            input_vectors: Input vectors (N x D)
            method: Distance difference computation method
            return_top_k: Only return top-k sample indices (None returns all)
        
        Returns:
            Dictionary containing:
            - "most_distinguishable_indices": Indices of most distinguishable samples (highest scores)
            - "least_distinguishable_indices": Indices of least distinguishable samples (lowest scores)
            - "scores": Distinguishability scores for all samples
            - "sorted_indices": Indices sorted by score from high to low
        """
        result = self.compute_distance_based_distinguishability(input_vectors, method)
        
        scores = result["scores"]
        
        # Sort by score (high to low: distinguishable -> ambiguous)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Most distinguishable samples (highest scores)
        most_distinguishable = sorted_indices[:return_top_k] if return_top_k else sorted_indices
        
        # Least distinguishable samples (lowest scores)
        least_distinguishable = sorted_indices[-return_top_k:][::-1] if return_top_k else sorted_indices[::-1]
        
        return {
            "most_distinguishable_indices": most_distinguishable,
            "least_distinguishable_indices": least_distinguishable,
            "scores": scores,
            "sorted_indices": sorted_indices,
            "min_distances": result["min_distances"],
            "second_min_distances": result["second_min_distances"],
            "best_expert_ids": result["best_expert_ids"],
            "second_best_expert_ids": result["second_best_expert_ids"],
            "method": method
        }
    
    def get_routing_statistics(self, 
                              input_vectors: np.ndarray,
                              expert_ids: np.ndarray,
                              distances: np.ndarray,
                              routing_weights: np.ndarray = None,
                              include_distinguishability: bool = True,
                              distinguishability_method: Literal["margin", "ratio", "gap_normalized", "std"] = "margin") -> dict:
        """
        Get statistics about routing behavior.
        
        Args:
            input_vectors: Input vectors to analyze (N x D)
            include_distinguishability: Whether to include distinguishability statistics (based on distance differences)
            distinguishability_method: Distinguishability computation method
            
        Returns:
            Dictionary containing routing statistics
        """
        # Count assignments per expert
        expert_counts = np.bincount(expert_ids, minlength=self.num_experts)
        
        # Compute distance statistics
        distance_stats = {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances))
        }
        
        # Compute expert load balancing
        load_balance = {
            "expert_counts": expert_counts.tolist(),
            "min_expert_load": int(np.min(expert_counts)),
            "max_expert_load": int(np.max(expert_counts)),
            "load_std": float(np.std(expert_counts)),
            "load_imbalance": int(np.max(expert_counts) - np.min(expert_counts))
        }
        
        stats = {
            "num_inputs": len(input_vectors),
            "num_experts": self.num_experts,
            "distance_stats": distance_stats,
            "load_balance": load_balance,
            "routing_type": "soft" if self.use_soft_routing else "hard"
        }
        
        if routing_weights is not None:
            # Compute entropy of routing weights (measure of routing diversity)
            entropies = []
            for weights in routing_weights:
                # Avoid log(0) by adding small epsilon
                eps = 1e-10
                weights_safe = weights + eps
                entropy = -np.sum(weights_safe * np.log(weights_safe))
                entropies.append(entropy)
            
            stats["routing_entropy"] = {
                "mean": float(np.mean(entropies)),
                "std": float(np.std(entropies)),
                "min": float(np.min(entropies)),
                "max": float(np.max(entropies))
            }
        
        # Add distinguishability statistics based on distance differences
        if include_distinguishability:
            dist_result = self.compute_distance_based_distinguishability(
                input_vectors, method=distinguishability_method
            )
            
            scores = dist_result["scores"]
            margins = dist_result["second_min_distances"] - dist_result["min_distances"]
            
            stats["distinguishability"] = {
                "method": distinguishability_method,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "median_score": float(np.median(scores)),
                "percentile_25": float(np.percentile(scores, 25)),
                "percentile_75": float(np.percentile(scores, 75)),

                # Distance margin statistics (always included for intuition)
                "mean_margin": float(np.mean(margins)),
                "std_margin": float(np.std(margins)),
                "min_margin": float(np.min(margins)),
                "max_margin": float(np.max(margins)),

                # Number of high/low confidence samples (based on median split)
                "num_high_confidence": int(np.sum(scores >= np.median(scores))),
                "num_low_confidence": int(np.sum(scores < np.median(scores)))
            }
        
        return stats
    
    def update_centroids(self, new_centroids: np.ndarray) -> None:
        """
        Update the centroids for routing.
        
        Args:
            new_centroids: New cluster centroids (K x D)
        """
        if new_centroids.shape[0] != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} centroids, "
                           f"got {new_centroids.shape[0]}")
        
        self.centroids = new_centroids
        logger.info(f"Updated centroids to shape {new_centroids.shape}")
    
    def set_temperature(self, temperature: float) -> None:
        """
        Update the temperature for soft routing.
        
        Args:
            temperature: New temperature value
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        self.temperature = temperature
        logger.info(f"Updated temperature to {temperature}")
    
    def enable_soft_routing(self, temperature: float = 1.0) -> None:
        """
        Enable soft routing with specified temperature.
        
        Args:
            temperature: Temperature for soft routing
        """
        self.use_soft_routing = True
        self.temperature = temperature
        logger.info(f"Enabled soft routing with temperature {temperature}")
    
    def disable_soft_routing(self) -> None:
        """Disable soft routing (use hard routing)."""
        self.use_soft_routing = False
        logger.info("Disabled soft routing, using hard routing")
