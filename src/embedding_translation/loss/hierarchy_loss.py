"""
Hierarchical ranking loss functions for embedding learning.

This module implements various hierarchical ranking losses that consider
the relative importance of different ranking positions.
"""

from typing import Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedTripletRankingLoss(nn.Module):
    """
    Vectorized triplet ranking loss with hierarchical weighting.
    
    This loss function creates triplets from ranked documents and applies
    different weights based on the ranking position, emphasizing higher-ranked
    documents more than lower-ranked ones.
    
    Args:
        margin: Margin for triplet loss. Defaults to 0.1.
        reduction: Specifies the reduction to apply to the output.
                  Can be 'none', 'mean', or 'sum'. Defaults to 'mean'.
        weight_mode: Weighting strategy for different ranking positions.
                    Can be 'linear' or 'softmax'. Defaults to 'linear'.
    
    Example:
        >>> loss_fn = VectorizedTripletRankingLoss(margin=0.2, weight_mode='softmax')
        >>> query_emb = torch.randn(32, 768)  # batch_size=32, dim=768
        >>> doc_emb = torch.randn(1000, 768)  # 1000 documents
        >>> index = torch.randint(0, 1000, (32, 10))  # top-10 docs per query
        >>> score = torch.randn(32, 10)  # relevance scores
        >>> loss = loss_fn(query_emb, doc_emb, index, score)
    """
    
    def __init__(
        self, 
        margin: float = 0.1, 
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
        weight_mode: Literal['linear', 'softmax'] = 'linear'
    ) -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.weight_mode = weight_mode
        
        # Validate parameters
        if margin <= 0:
            raise ValueError(f"Margin must be positive, got {margin}")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Reduction must be one of ['none', 'mean', 'sum'], got {reduction}")
        if weight_mode not in ['linear', 'softmax']:
            raise ValueError(f"Weight mode must be one of ['linear', 'softmax'], got {weight_mode}")

    def _compute_weights(self, k: int, device: torch.device) -> torch.Tensor:
        """
        Compute weights for different ranking positions.
        
        Args:
            k: Number of ranking positions (excluding the last one)
            device: Device to create weights on
            
        Returns:
            Tensor of weights with shape [k-1]
        """
        if self.weight_mode == 'linear':
            # Linear decay from 1.0 to 0.1
            weights = torch.linspace(1.0, 0.1, steps=k - 1, device=device)
        elif self.weight_mode == 'softmax':
            # Softmax weights with linear decay
            raw_weights = torch.linspace(1.0, 0.0, steps=k - 1, device=device)
            weights = F.softmax(raw_weights, dim=0)
        else:
            raise ValueError(f"Unsupported weight mode: {self.weight_mode}")
        
        return weights

    def forward(
        self,
        query_emb: torch.Tensor,  # [B, D]
        doc_emb: torch.Tensor,    # [N, D]
        index: torch.Tensor,      # [B, K]
        score: torch.Tensor       # [B, K]
    ) -> torch.Tensor:
        """
        Forward pass of the hierarchical triplet ranking loss.
        
        Args:
            query_emb: Query embeddings with shape [batch_size, embedding_dim]
            doc_emb: Document embeddings with shape [num_docs, embedding_dim]
            index: Document indices for each query with shape [batch_size, top_k]
            score: Relevance scores for each query-document pair with shape [batch_size, top_k]
            
        Returns:
            Computed loss value
        """
        B, D = query_emb.size()
        K = index.size(1)
        
        # Validate input dimensions
        if doc_emb.size(1) != D:
            raise ValueError(f"Document embedding dimension {doc_emb.size(1)} doesn't match query dimension {D}")
        if index.size(0) != B:
            raise ValueError(f"Index batch size {index.size(0)} doesn't match query batch size {B}")
        if score.shape != index.shape:
            raise ValueError(f"Score shape {score.shape} doesn't match index shape {index.shape}")
        if K < 2:
            raise ValueError(f"Need at least 2 documents per query for triplet loss, got {K}")

        # Normalize embeddings for cosine similarity
        query_emb = F.normalize(query_emb, dim=-1)           # [B, D]
        doc_emb = F.normalize(doc_emb, dim=-1)               # [N, D]
        
        # Gather relevant documents for each query
        doc_batch = doc_emb[index]                           # [B, K, D]

        # Sort documents by relevance scores (descending order)
        sorted_score_idx = torch.argsort(score, dim=1, descending=True)  # [B, K]
        sorted_doc = torch.gather(
            doc_batch, 
            1, 
            sorted_score_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]

        # Compute hierarchical weights
        weights = self._compute_weights(K, query_emb.device)  # [K-1]
        
        total_loss = 0.0
        total_count = 0

        # Create triplets with hierarchical weighting
        for i in range(K - 1):
            # Anchor: query embedding
            anchor = query_emb                              # [B, D]
            # Positive: higher-ranked document
            positive = sorted_doc[:, i]                     # [B, D]
            # Negatives: all lower-ranked documents
            negative = sorted_doc[:, (i+1):]                # [B, K-i-1, D]

            # Expand anchor and positive to match negative dimensions
            num_negatives = negative.size(1)
            a = anchor.unsqueeze(1).expand(-1, num_negatives, -1)      # [B, K-i-1, D]
            p = positive.unsqueeze(1).expand(-1, num_negatives, -1)    # [B, K-i-1, D]

            # Reshape for batched triplet loss computation
            a_flat = a.contiguous().view(-1, D)             # [B*(K-i-1), D]
            p_flat = p.contiguous().view(-1, D)             # [B*(K-i-1), D]
            n_flat = negative.contiguous().view(-1, D)      # [B*(K-i-1), D]

            # Compute triplet loss
            triplet_loss = F.triplet_margin_loss(
                a_flat, p_flat, n_flat, 
                margin=self.margin, 
                reduction='none'
            )  # [B*(K-i-1)]

            # Apply hierarchical weighting
            weighted_loss = weights[i] * triplet_loss
            
            if self.reduction == 'mean':
                total_loss += weighted_loss.mean()
            elif self.reduction == 'sum':
                total_loss += weighted_loss.sum()
            else:  # 'none'
                if i == 0:
                    total_loss = weighted_loss.view(B, -1)
                else:
                    # Concatenate losses for 'none' reduction
                    total_loss = torch.cat([total_loss, weighted_loss.view(B, -1)], dim=1)
            
            total_count += num_negatives

        # For 'none' reduction, return per-sample losses
        if self.reduction == 'none':
            return total_loss  # [B, total_triplets]
        
        return total_loss

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"margin={self.margin}, "
                f"reduction='{self.reduction}', "
                f"weight_mode='{self.weight_mode}')")


class AdaptiveHierarchicalLoss(VectorizedTripletRankingLoss):
    """
    Adaptive hierarchical loss that adjusts weights based on score differences.
    
    This variant of the hierarchical triplet loss adapts the weighting based on
    the actual score differences between documents, giving more weight to pairs
    with smaller score differences (harder negatives).
    
    Args:
        margin: Margin for triplet loss. Defaults to 0.1.
        reduction: Specifies the reduction to apply to the output. Defaults to 'mean'.
        weight_mode: Base weighting strategy. Defaults to 'linear'.
        adaptive_factor: Factor controlling adaptive weighting strength. Defaults to 1.0.
    """
    
    def __init__(
        self,
        margin: float = 0.1,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
        weight_mode: Literal['linear', 'softmax'] = 'linear',
        adaptive_factor: float = 1.0
    ) -> None:
        super().__init__(margin, reduction, weight_mode)
        self.adaptive_factor = adaptive_factor
        
        if adaptive_factor <= 0:
            raise ValueError(f"Adaptive factor must be positive, got {adaptive_factor}")

    def forward(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        index: torch.Tensor,
        score: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with adaptive weighting based on score differences.
        """
        B, D = query_emb.size()
        K = index.size(1)
        
        # Validate inputs (same as parent class)
        if K < 2:
            raise ValueError(f"Need at least 2 documents per query for triplet loss, got {K}")

        # Normalize embeddings
        query_emb = F.normalize(query_emb, dim=-1)
        doc_emb = F.normalize(doc_emb, dim=-1)
        doc_batch = doc_emb[index]

        # Sort by relevance scores
        sorted_score_idx = torch.argsort(score, dim=1, descending=True)
        sorted_doc = torch.gather(doc_batch, 1, sorted_score_idx.unsqueeze(-1).expand(-1, -1, D))
        sorted_scores = torch.gather(score, 1, sorted_score_idx)

        # Base hierarchical weights
        base_weights = self._compute_weights(K, query_emb.device)
        
        total_loss = 0.0

        for i in range(K - 1):
            anchor = query_emb
            positive = sorted_doc[:, i]
            negative = sorted_doc[:, (i+1):]
            
            # Compute adaptive weights based on score differences
            pos_scores = sorted_scores[:, i:i+1]  # [B, 1]
            neg_scores = sorted_scores[:, (i+1):]  # [B, K-i-1]
            score_diff = pos_scores - neg_scores   # [B, K-i-1]
            
            # Adaptive weighting: smaller differences get higher weights
            adaptive_weights = torch.exp(-self.adaptive_factor * score_diff.abs())  # [B, K-i-1]
            
            # Expand dimensions for triplet computation
            num_negatives = negative.size(1)
            a = anchor.unsqueeze(1).expand(-1, num_negatives, -1)
            p = positive.unsqueeze(1).expand(-1, num_negatives, -1)

            # Compute triplet loss
            a_flat = a.contiguous().view(-1, D)
            p_flat = p.contiguous().view(-1, D)
            n_flat = negative.contiguous().view(-1, D)

            triplet_loss = F.triplet_margin_loss(
                a_flat, p_flat, n_flat,
                margin=self.margin,
                reduction='none'
            ).view(B, -1)  # [B, K-i-1]

            # Apply both base and adaptive weights
            combined_weights = base_weights[i] * adaptive_weights
            weighted_loss = combined_weights * triplet_loss

            if self.reduction == 'mean':
                total_loss += weighted_loss.mean()
            elif self.reduction == 'sum':
                total_loss += weighted_loss.sum()
            else:  # 'none'
                if i == 0:
                    total_loss = weighted_loss
                else:
                    total_loss = torch.cat([total_loss, weighted_loss], dim=1)

        return total_loss

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"margin={self.margin}, "
                f"reduction='{self.reduction}', "
                f"weight_mode='{self.weight_mode}', "
                f"adaptive_factor={self.adaptive_factor})") 