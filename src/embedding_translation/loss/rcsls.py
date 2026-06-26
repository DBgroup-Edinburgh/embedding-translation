import torch
from typing import Tuple
from .util import get_knn_faiss

def rcsls_torch(
    args,
    X_src: torch.Tensor,
    Y_tgt: torch.Tensor,
    Z_src: torch.Tensor,
    Z_tgt: torch.Tensor,
    R: torch.Tensor,
    knn: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float, float]]:
    """
    Compute RCSLS loss using FAISS for KNN
    
    Args:
        X_src: Source embeddings
        Y_tgt: Target embeddings
        Z_src: Source negative embeddings
        Z_tgt: Target negative embeddings
        R: Transformation matrix
        knn: Number of nearest neighbors
        
    Returns:
        Tuple of (loss, gradient, (f_log_value, fk0_log_value, fk1_log_value))
    """
    # Transform source embeddings
    X_trans = torch.mm(X_src, R.t())

    X_trans = torch.nn.functional.normalize(X_trans, p=2, dim=1)
    X_src = torch.nn.functional.normalize(X_src, p=2, dim=1)
    Y_tgt = torch.nn.functional.normalize(Y_tgt, p=2, dim=1)
    Z_tgt = torch.nn.functional.normalize(Z_tgt, p=2, dim=1)
    Z_src = torch.nn.functional.normalize(Z_src, p=2, dim=1)
    
    # Compute main similarity term
    f = 2 * torch.sum(X_trans * Y_tgt)
    df = 2 * torch.mm(Y_tgt.t(), X_src)
    
    # Get KNN terms using FAISS
    fk0_score, fk0_indices = get_knn_faiss(X_trans, X_src, Z_tgt, knn)
    fk1_score, fk1_indices = get_knn_faiss(torch.mm(Z_src, R.t()), Y_tgt, Z_src, knn)

    fk0_vectors = Z_tgt[fk0_indices]
    fk1_vectors = Z_src[fk1_indices]

    fk0 = (X_trans.unsqueeze(1) * fk0_vectors).sum(dim=2).mean(dim=1)  # [2000]
    fk1 = (fk1_vectors * Y_tgt.unsqueeze(1)).sum(dim=2).mean(dim=1)  # [2000]

    f_log_value = (f / X_src.shape[0]).item()
    
    # Combine terms
    f = f - args.lambda_ * (fk0.sum() + fk1.sum())

    # log value
    fk0_log_value = (fk0.sum() / X_src.shape[0]).item()
    fk1_log_value = (fk1.sum() / X_src.shape[0]).item()
    
    return -f / X_src.shape[0], -df / X_src.shape[0], (f_log_value, fk0_log_value, fk1_log_value) 