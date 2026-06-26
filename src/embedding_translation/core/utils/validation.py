"""Validation utilities."""

import numpy as np

def validate_embeddings(embeddings: np.ndarray) -> bool:
    """Validate embedding array."""
    if not isinstance(embeddings, np.ndarray):
        return False
    if len(embeddings.shape) != 2:
        return False
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        return False
    return True 