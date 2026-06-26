import torch
from typing import Tuple
from .util import get_knn_faiss
from torch.nn import functional as F


class DistanceLoss(torch.nn.Module):
    def __init__(self, args, Y_tgt: torch.Tensor, knn: int = 10):
        super(DistanceLoss, self).__init__()
        self.args = args
        self.knn = knn
        self.topk_indices, self.topk_distance_matrix = self.target_distance_matrix(Y_tgt)
    
    def target_distance_matrix(self, Y_tgt: torch.Tensor):
        _, topk_indices = get_knn_faiss(Y_tgt, Y_tgt, self.knn)
        topk_vectors = Y_tgt[topk_indices]  # Shape: [X, K, 300]
        Y_tgt_reshaped = Y_tgt.unsqueeze(1)  # Shape: [X, 1, 300]
        dot_product = torch.bmm(Y_tgt_reshaped, topk_vectors.transpose(1, 2))  # Shape: [X, 1, K]
        return topk_indices, dot_product.squeeze(1)  # Shape: [X, K]

    def loss(self, X_trans: torch.Tensor, Y_tgt: torch.Tensor) -> torch.Tensor:
        X_topk_vectors = X_trans[self.topk_indices]
        X_topk_distance_matrix = torch.bmm(X_trans.unsqueeze(1), X_topk_vectors.transpose(1, 2))

        mse_term = F.mse_loss(X_topk_distance_matrix, self.topk_distance_matrix)
        # mse_term = torch.tensor(0.0)
        direct_term = F.mse_loss(X_trans, Y_tgt)

        return direct_term + self.args.lambda_ * mse_term, {"mse_term": mse_term.item(), "direct_term": direct_term.item()}
