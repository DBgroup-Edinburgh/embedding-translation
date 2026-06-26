"""I/O utilities for embeddings."""

import numpy as np
from pathlib import Path
from typing import Union

def load_embeddings(path: Union[str, Path]) -> np.ndarray:
    """Load embeddings from file."""
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.npz':
        data = np.load(path)
        return data[data.files[0]]
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

def save_embeddings(embeddings: np.ndarray, path: Union[str, Path]) -> None:
    """Save embeddings to file.""" 
    path = Path(path)
    if path.suffix == '.npy':
        np.save(path, embeddings)
    elif path.suffix == '.npz':
        np.savez(path, embeddings=embeddings)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}") 