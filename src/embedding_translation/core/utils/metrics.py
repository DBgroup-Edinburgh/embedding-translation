"""Metrics utilities."""

import numpy as np
from typing import Dict

def compute_alignment_metrics(embeddings1: np.ndarray, embeddings2: np.ndarray) -> Dict[str, float]:
    """Compute alignment metrics between two embedding sets."""
    # MSE
    mse = float(np.mean((embeddings1 - embeddings2) ** 2))
    
    # Cosine similarity
    norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
    norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
    cos_sim = float(np.mean(np.sum(norm1 * norm2, axis=1)))
    
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'cosine_similarity': cos_sim
    } 