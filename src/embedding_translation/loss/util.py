import torch
import faiss
import numpy as np
from typing import Tuple

def get_knn_faiss(
    query: torch.Tensor,
    target: torch.Tensor,
    k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute k-nearest neighbors using FAISS
    
    Args:
        query: Query embeddings (N x D)
        target: Target embeddings (M x D)
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (scores, indices) for top-k nearest neighbors
    """
    # Convert to numpy and normalize
    query_np = query.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    faiss.normalize_L2(query_np)
    faiss.normalize_L2(target_np)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(target_np.shape[1])
    index.add(target_np)
    
    # Search
    scores, indices = index.search(query_np, k)
    
    return torch.from_numpy(scores).to(query.device), torch.from_numpy(indices).to(query.device) 