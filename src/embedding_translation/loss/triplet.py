import torch
from typing import Tuple, List, Dict
from .util import get_knn_faiss
from torch.nn import functional as F
import numpy as np


class ListWiseTripletLoss(torch.nn.Module):
    def __init__(self, args, Y_tgt: torch.Tensor, knn: int = 10, topk_indices: List = None, query_vectors: torch.Tensor = None):
        super().__init__()
        self.args = args
        self.knn = knn

        if topk_indices is None:
            self.query_vectors, self.topk_indices = self.target_distance_matrix(Y_tgt)
        else:
            self.query_vectors = query_vectors
            self.topk_indices = topk_indices

        self.triplet_groups = self.create_triplet_groups(self.query_vectors, self.topk_indices)
        self.triplet_margin = getattr(args, "triplet_margin", 0.01)

    def target_distance_matrix(self, Y_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_vectors = Y_tgt[:10]
        _, topk_indices = get_knn_faiss(query_vectors, Y_tgt, self.knn)
        # topk_vectors = Y_tgt[topk_indices]  # Shape: [X, K, D]
        # Y_tgt_reshaped = Y_tgt.unsqueeze(1)  # Shape: [X, 1, D]
        # dot_product = torch.bmm(Y_tgt_reshaped, topk_vectors.transpose(1, 2))  # Shape: [X, 1, K]
        return query_vectors, topk_indices
    
    def create_triplet_groups_by_negtive(self, query_vectors: torch.Tensor, trans_idx: torch.Tensor, query_topk: int = 10):
        topk_indices = self.topk_indices    
        triplet_groups = []
        for query_vector, topk, trans_k in zip(query_vectors, topk_indices, trans_idx):
            gt_results = topk[:query_topk]
            current_results = trans_k[:query_topk]
            
            negative_items = [idx for idx in current_results if idx not in gt_results]
            if len(negative_items) == 0:
                # select from the query_k: items
                if query_topk+10 < len(topk):
                    end = query_topk + 10
                else:
                    end = -1
                neg_indices = topk[query_topk:end]
            
            for pos in gt_results:
                for neg in negative_items:
                    triplet_groups.append((query_vector, pos, neg))
        return triplet_groups

    def create_triplet_groups(self, query_vectors: torch.Tensor, topk_indices: torch.Tensor, query_topk: int = 10) -> List[Tuple[int, int, int]]:
        """
        For each query, create triplets (anchor, positive, negative) from its topk_indices.
        Anchor is always the first (index 0), positive is the second (index 1), negatives are the rest.
        Returns a list of (anchor_idx, positive_idx, negative_idx).
        """
        triplet_groups: List[Tuple[int, int, int]] = []
        for query_vector, topk in zip(query_vectors, topk_indices):
            # query_group: [K]
            # anchor = query_group[0].item()
            # positive = query_vector
            anchor = query_vector
            # positive = query_group[1].item() if len(query_group) > 1 else None
            for i in range(0, query_topk):
                # negative = query_group[i].item()
                # positive = topk[i].item()
                # anchor = topk[i].item()
                positive = topk[i].item()
                # Generate random index for negative sample
                neg_indices = topk[query_topk:]
                if len(neg_indices) > 0:
                    ranks = torch.arange(len(neg_indices), device=neg_indices.device)
                    # Use softmax with temperature to create smoother weights
                    temperature = 0.5  # Lower temperature makes the distribution more uniform
                    weights = torch.softmax(-ranks.float() * temperature, dim=0)
                    rand_idx = torch.multinomial(weights, 1)[0]
                    negative = neg_indices[rand_idx].item()

                    triplet_groups.append((anchor, positive, negative))
        return triplet_groups

    def loss(self, X_trans: torch.Tensor, Y_tgt: torch.Tensor, triplet_groups = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        direct_term = F.mse_loss(X_trans, Y_tgt)
        if triplet_groups is None:
            triplet_groups = self.triplet_groups
        if not triplet_groups:
            triplet_term = torch.tensor(0.0, device=X_trans.device)
        else:
            # anchors = torch.stack([X_trans[a] for a, _, _ in triplet_groups])
            anchors = torch.stack([a for a, _, _ in triplet_groups])
            positives = torch.stack([X_trans[p] for _, p, _ in triplet_groups])
            # positives = torch.stack([p for _, p, _ in triplet_groups])
            negatives = torch.stack([X_trans[n] for _, _, n in triplet_groups])
            triplet_term = F.triplet_margin_loss(
                anchors, positives, negatives, margin=self.triplet_margin, reduction='mean')
        total_loss = direct_term + self.args.lambda_ * triplet_term
        return total_loss, {"triplet_term": float(triplet_term.item()), "direct_term": float(direct_term.item())}
