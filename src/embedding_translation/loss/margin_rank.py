import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MarginRankingLoss(nn.Module):
    """
    Margin Ranking Loss for pairwise ranking.
    
    For a query and two documents doc_i and doc_j, if doc_i is more relevant than doc_j,
    we want: score_i > score_j + margin
    
    Loss = max(0, margin - (score_i - score_j))
    """
    
    def __init__(self, margin: float = 0.1, reduction: str = 'mean'):
        """
        Initialize Margin Ranking Loss.
        
        Args:
            margin (float): Margin value for ranking constraint. Default: 0.1
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, 
                query_emb: torch.Tensor,    # Shape: (B, D)
                doc_emb: torch.Tensor,      # Shape: (N, D)
                index: torch.Tensor,        # Shape: (B, K)
                score: torch.Tensor         # Shape: (B, K)
               ) -> torch.Tensor:
        """
        Compute Margin Ranking Loss.
        
        Args:
            query_emb (torch.Tensor): Query embeddings [B, D]
            doc_emb (torch.Tensor): Document embeddings [N, D]
            index (torch.Tensor): Document indices for each query [B, K]
            score (torch.Tensor): Relevance scores for documents [B, K]
            
        Returns:
            torch.Tensor: Margin ranking loss
        """
        B, D = query_emb.shape
        _, K = index.shape
        
        # Normalize embeddings
        query_emb = F.normalize(query_emb, dim=-1)
        doc_emb = F.normalize(doc_emb, dim=-1)
        
        # Get document embeddings for each query
        doc_batch_emb = doc_emb[index]  # [B, K, D]
        
        # Compute similarity scores
        # Using cosine similarity after normalization
        scores = torch.sum(query_emb.unsqueeze(1) * doc_batch_emb, dim=-1)  # [B, K]
        
        # Create all pairs within each query's K documents
        losses = []
        
        for b in range(B):
            query_scores = scores[b]      # [K]
            query_relevance = score[b]    # [K]
            
            # Get all pairs (i, j) where i != j
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    
                    # If doc_i is more relevant than doc_j
                    if query_relevance[i] > query_relevance[j]:
                        # We want score_i > score_j + margin
                        loss_ij = F.relu(self.margin - (query_scores[i] - query_scores[j]))
                        losses.append(loss_ij)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=query_emb.device, requires_grad=True)
        
        # Stack all losses
        losses = torch.stack(losses)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses


class VectorizedMarginRankingLoss(nn.Module):
    """
    Vectorized version of Margin Ranking Loss for better efficiency.
    """
    
    def __init__(self, margin: float = 0.1, reduction: str = 'mean'):
        """
        Initialize Vectorized Margin Ranking Loss.
        
        Args:
            margin (float): Margin value for ranking constraint. Default: 0.1
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(VectorizedMarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, 
                query_emb: torch.Tensor,    # Shape: (B, D)
                doc_emb: torch.Tensor,      # Shape: (N, D)
                index: torch.Tensor,        # Shape: (B, K)
                score: torch.Tensor         # Shape: (B, K)
               ) -> torch.Tensor:
        """
        Compute Margin Ranking Loss using vectorized operations.
        
        Args:
            query_emb (torch.Tensor): Query embeddings [B, D]
            doc_emb (torch.Tensor): Document embeddings [N, D]
            index (torch.Tensor): Document indices for each query [B, K]
            score (torch.Tensor): Relevance scores for documents [B, K]
            
        Returns:
            torch.Tensor: Margin ranking loss
        """
        B, D = query_emb.shape
        _, K = index.shape
        
        # Normalize embeddings
        query_emb = F.normalize(query_emb, dim=-1)
        doc_emb = F.normalize(doc_emb, dim=-1)
        
        # Get document embeddings for each query
        doc_batch_emb = doc_emb[index]  # [B, K, D]
        
        # Compute similarity scores
        # scores = torch.sum(query_emb.unsqueeze(1) * doc_batch_emb, dim=-1)  # [B, K]
        scores = F.cosine_similarity(query_emb, doc_batch_emb, dim=-1)  # [B, K]
        
        # Create pairwise score differences
        scores_i = scores.unsqueeze(2)  # [B, K, 1]
        scores_j = scores.unsqueeze(1)  # [B, 1, K]
        score_diff = scores_i - scores_j  # [B, K, K]
        
        # Create pairwise relevance comparisons
        relevance_i = score.unsqueeze(2)  # [B, K, 1]
        relevance_j = score.unsqueeze(1)  # [B, 1, K]
        
        # Mask for valid pairs: where relevance_i > relevance_j
        valid_pairs = (relevance_i > relevance_j).float()  # [B, K, K]
        
        # Compute losses for all pairs
        losses = F.relu(self.margin - score_diff) * valid_pairs  # [B, K, K]
        
        # Count valid pairs for proper averaging
        num_valid_pairs = valid_pairs.sum()
        
        if num_valid_pairs == 0:
            return torch.tensor(0.0, device=query_emb.device, requires_grad=True)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.sum() / num_valid_pairs
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses 