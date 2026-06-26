"""Embedding adapter — delegates to vectorbench.

Imports are lazy so callers who never touch embeddings can use the rest of the
library without vectorbench installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Typed request for fetching pre-computed embeddings."""

    dataset_name: str
    model_name: str
    embedding_path: Path = Field(default_factory=lambda: Path("./.cache/embeddings"))
    type_: Literal["corpus", "query"] = "corpus"
    align: bool = True
    download: bool = True
    target_model_name: str | None = None

    model_config = {"protected_namespaces": ()}


def get_embedding(req: EmbeddingRequest) -> np.ndarray:
    """Fetch pre-computed embeddings via vectorbench."""
    from vectorbench.embeddings.wrapper import get_embedding as _vb_get_embedding

    return _vb_get_embedding(
        dataset_name=req.dataset_name,
        model_name=req.model_name,
        embedding_path=str(req.embedding_path),
        target_model_name=req.target_model_name,
        type_=req.type_,
        align=req.align,
        download=req.download,
    )


def generate_embeddings(*args: Any, **kwargs: Any) -> Any:
    """Generate embeddings via vectorbench."""
    from vectorbench.embeddings.wrapper import generate_embeddings as _vb_gen

    return _vb_gen(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "SUPPORTED_MODELS":
        from vectorbench.embeddings.generator.factory import SUPPORTED_MODELS as _S

        return _S
    if name == "get_embedding_generator":
        from vectorbench.embeddings.generator.factory import (
            get_embedding_generator as _g,
        )

        return _g
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EmbeddingRequest",
    "get_embedding",
    "generate_embeddings",
    "SUPPORTED_MODELS",
    "get_embedding_generator",
]
