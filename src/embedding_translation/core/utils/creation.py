"""Creation utilities."""

import numpy as np
from typing import Union

def create_reference_indices(n_total: int, n_ref: Union[int, float], 
                           random_state: int = 42) -> np.ndarray:
    """Create random reference indices."""
    np.random.seed(random_state)
    
    if isinstance(n_ref, float):
        n_ref = int(n_total * n_ref)
    
    n_ref = min(n_ref, n_total)
    return np.random.choice(n_total, size=n_ref, replace=False) 