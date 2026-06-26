"""
Common loss functions for VectorMerge mapping strategies.

This module implements various loss functions used in neural network-based
mapping strategies, including cosine similarity, triplet loss, ranking loss, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from loguru import logger


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss.
    
    Args:
        predictions: Predicted embeddings
        targets: Target embeddings
        
    Returns:
        MSE loss
    """
    return F.mse_loss(predictions, targets)


def cosine_similarity_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss.
    
    Args:
        predictions: Predicted embeddings
        targets: Target embeddings
        
    Returns:
        Cosine similarity loss (1 - cosine_similarity)
    """
    # Normalize vectors
    predictions_norm = F.normalize(predictions, p=2, dim=1)
    targets_norm = F.normalize(targets, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(predictions_norm * targets_norm, dim=1)
    
    # Convert to loss (1 - similarity)
    loss = 1 - cosine_sim
    
    return loss.mean()


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                margin: float = 1.0) -> torch.Tensor:
    """Triplet loss for embedding learning.
    
    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings (similar to anchor)
        negative: Negative embeddings (dissimilar to anchor)
        margin: Margin for triplet loss
        
    Returns:
        Triplet loss
    """
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)
    return triplet_loss_fn(anchor, positive, negative)


def ranking_loss(predictions: torch.Tensor, targets: torch.Tensor,
                queries: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Ranking loss for information retrieval.
    
    This loss encourages predicted embeddings to have the same ranking
    as target embeddings when compared with query embeddings.
    
    Args:
        predictions: Predicted embeddings
        targets: Target embeddings
        queries: Query embeddings
        margin: Margin for ranking loss
        
    Returns:
        Ranking loss
    """
    # Compute similarities
    pred_similarities = torch.matmul(predictions, queries.T)
    target_similarities = torch.matmul(targets, queries.T)
    
    # Compute ranking loss
    diff = pred_similarities - target_similarities
    loss = torch.clamp(margin - diff, min=0.0)
    
    return loss.mean()


def contrastive_loss(predictions: torch.Tensor, targets: torch.Tensor,
                    labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Contrastive loss for similarity learning.
    
    Args:
        predictions: Predicted embeddings
        targets: Target embeddings  
        labels: Binary labels (1 for similar, 0 for dissimilar)
        margin: Margin for contrastive loss
        
    Returns:
        Contrastive loss
    """
    # Compute euclidean distance
    distances = F.pairwise_distance(predictions, targets)
    
    # Contrastive loss
    loss_pos = labels * torch.pow(distances, 2)
    loss_neg = (1 - labels) * torch.pow(torch.clamp(margin - distances, min=0.0), 2)
    
    loss = loss_pos + loss_neg
    return loss.mean()


def hierarchy_loss(predictions: torch.Tensor, targets: torch.Tensor,
                  query_embeddings: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Hierarchy loss combining multiple objectives.
    
    This loss combines cosine similarity loss with ranking loss,
    as used in some embedding mapping papers.
    
    Args:
        predictions: Predicted embeddings
        targets: Target embeddings
        query_embeddings: Query embeddings for ranking
        lambda_: Weight for ranking loss component
        
    Returns:
        Combined hierarchy loss
    """
    # Cosine similarity loss
    cosine_loss = cosine_similarity_loss(predictions, targets)
    
    # Ranking loss
    rank_loss = ranking_loss(predictions, targets, query_embeddings)
    
    # Combined loss
    total_loss = cosine_loss + lambda_ * rank_loss
    
    return total_loss


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor,
                loss_type: str, query_embeddings: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
    """Compute loss based on specified type.
    
    Args:
        predictions: Predicted embeddings
        targets: Target embeddings
        loss_type: Type of loss to compute
        query_embeddings: Query embeddings (for some loss types)
        **kwargs: Additional arguments for specific loss functions
        
    Returns:
        Computed loss
    """
    loss_type = loss_type.lower()
    
    if loss_type == "mse":
        return mse_loss(predictions, targets)
    
    elif loss_type == "cosine":
        return cosine_similarity_loss(predictions, targets)
    
    elif loss_type == "triplet":
        # For triplet loss, we need to create triplets
        if query_embeddings is None:
            # Use random negative sampling if no queries provided
            batch_size = predictions.size(0)
            neg_indices = torch.randperm(batch_size, device=predictions.device)
            negatives = targets[neg_indices]
        else:
            # Use query embeddings as anchors
            anchor_size = min(predictions.size(0), query_embeddings.size(0))
            anchors = query_embeddings[:anchor_size]
            positives = predictions[:anchor_size]
            negatives = targets[torch.randperm(anchor_size, device=targets.device)]
            
            margin = kwargs.get("margin", 1.0)
            return triplet_loss(anchors, positives, negatives, margin)
        
        margin = kwargs.get("margin", 1.0)
        return triplet_loss(predictions, targets, negatives, margin)
    
    elif loss_type == "ranking":
        if query_embeddings is None:
            logger.warning("Query embeddings required for ranking loss, using cosine loss")
            return cosine_similarity_loss(predictions, targets)
        
        margin = kwargs.get("margin", 0.1)
        return ranking_loss(predictions, targets, query_embeddings, margin)
    
    elif loss_type == "contrastive":
        # For contrastive loss, we need labels
        labels = kwargs.get("labels")
        if labels is None:
            # Create random labels if not provided
            batch_size = predictions.size(0)
            labels = torch.randint(0, 2, (batch_size,), device=predictions.device).float()
        
        margin = kwargs.get("margin", 1.0)
        return contrastive_loss(predictions, targets, labels, margin)
    
    elif loss_type == "hierarchy":
        if query_embeddings is None:
            logger.warning("Query embeddings required for hierarchy loss, using cosine loss")
            return cosine_similarity_loss(predictions, targets)
        
        lambda_ = kwargs.get("lambda_", 1.0)
        return hierarchy_loss(predictions, targets, query_embeddings, lambda_)
    
    else:
        logger.warning(f"Unknown loss type: {loss_type}, using MSE loss")
        return mse_loss(predictions, targets)


def get_loss_info(loss_type: str) -> Dict[str, Any]:
    """Get information about a specific loss type.
    
    Args:
        loss_type: Type of loss
        
    Returns:
        Dictionary with loss information
    """
    loss_info = {
        "mse": {
            "name": "Mean Squared Error",
            "description": "L2 distance between predictions and targets",
            "requires_queries": False,
            "requires_labels": False
        },
        "cosine": {
            "name": "Cosine Similarity Loss",
            "description": "1 - cosine similarity between normalized vectors",
            "requires_queries": False,
            "requires_labels": False
        },
        "triplet": {
            "name": "Triplet Loss",
            "description": "Triplet margin loss for embedding learning",
            "requires_queries": True,
            "requires_labels": False
        },
        "ranking": {
            "name": "Ranking Loss",
            "description": "Preserves ranking order with respect to queries",
            "requires_queries": True,
            "requires_labels": False
        },
        "contrastive": {
            "name": "Contrastive Loss",
            "description": "Pulls similar pairs together, pushes dissimilar apart",
            "requires_queries": False,
            "requires_labels": True
        },
        "hierarchy": {
            "name": "Hierarchy Loss",
            "description": "Combines cosine similarity and ranking loss",
            "requires_queries": True,
            "requires_labels": False
        }
    }
    
    return loss_info.get(loss_type.lower(), {
        "name": "Unknown Loss",
        "description": "Unknown loss type",
        "requires_queries": False,
        "requires_labels": False
    })


def validate_loss_inputs(loss_type: str, predictions: torch.Tensor, targets: torch.Tensor,
                        query_embeddings: Optional[torch.Tensor] = None,
                        labels: Optional[torch.Tensor] = None) -> bool:
    """Validate inputs for loss computation.
    
    Args:
        loss_type: Type of loss
        predictions: Predicted embeddings
        targets: Target embeddings
        query_embeddings: Query embeddings (optional)
        labels: Labels (optional)
        
    Returns:
        True if inputs are valid
    """
    # Basic validation
    if predictions.shape != targets.shape:
        logger.error(f"Predictions and targets must have same shape: "
                    f"{predictions.shape} vs {targets.shape}")
        return False
    
    # Loss-specific validation
    loss_info = get_loss_info(loss_type)
    
    if loss_info["requires_queries"] and query_embeddings is None:
        logger.error(f"Loss type '{loss_type}' requires query embeddings")
        return False
    
    if loss_info["requires_labels"] and labels is None:
        logger.error(f"Loss type '{loss_type}' requires labels")
        return False
    
    return True 