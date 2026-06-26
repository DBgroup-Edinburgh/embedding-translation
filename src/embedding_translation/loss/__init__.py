"""Loss functions for mapper training.

Two families coexist here:

- VM-era functional losses (`common.py`): cosine, MSE, ranking, contrastive,
  triplet, plus a `compute_loss(name, ...)` dispatcher.
- VT-era class-based losses (the other files): TripletLoss,
  VectorizedTripletRankingLoss, DistanceLoss, ranking losses, RCSLS, Spearman.
  These take Tensors and return scalars; used inside training loops.
"""

# VM-era functional losses
from .common import (
    compute_loss,
    contrastive_loss,
    cosine_similarity_loss,
    mse_loss,
    ranking_loss,
    triplet_loss,
)

# VT-era class-based losses
from .contrastive_loss import TripletLoss
from .distance import DistanceLoss
from .hierarchy_loss import VectorizedTripletRankingLoss
from .lambda_rank_loss import LambdaRankLoss
from .ranking_losses import get_ranking_loss
from .rcsls import rcsls_torch
from .triplet import ListWiseTripletLoss
from .util import get_knn_faiss

__all__ = [
    # Functional (VM)
    "cosine_similarity_loss",
    "triplet_loss",
    "ranking_loss",
    "contrastive_loss",
    "mse_loss",
    "compute_loss",
    # Class-based (VT)
    "TripletLoss",
    "VectorizedTripletRankingLoss",
    "DistanceLoss",
    "ListWiseTripletLoss",
    "LambdaRankLoss",
    "get_ranking_loss",
    "rcsls_torch",
    "get_knn_faiss",
]
