"""High-level reference-split orchestration.

The original VectorMerge code took an `EmbeddingDataset` wrapper that bundled
a text dataset together with cached embeddings. We removed that wrapper —
callers now pass a `vectorbench.dataset.Dataset` directly, which exposes
the same `.qrels`, `get_dataset_index()`, and `get_internal_index()` surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base_split import BaseSplit as RandomSplit
from .base_split import LA2MSplitConfig, SplitConfig
from .la2m_split import LA2MSplit

if TYPE_CHECKING:
    from ..dataset import Dataset


def get_split_config(split_method_name: str, **kwargs: Any) -> SplitConfig:
    if split_method_name == "la2m":
        return LA2MSplitConfig().update(**kwargs)
    elif split_method_name == "random":
        return SplitConfig().update(**kwargs)
    else:
        raise ValueError(f"Invalid split method: {split_method_name}")


def split_dataset(
    dataset: "Dataset",
    reference_method: str,
    update_split_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a Dataset into D0 (reference), D1, D2 index sets."""
    update_split_config = update_split_config or {}
    split_config = get_split_config(reference_method, **update_split_config)

    if reference_method == "la2m":
        splitter = LA2MSplit.from_qrels(
            dataset_name=dataset.name,
            dataset_index=dataset.get_dataset_index(),
            internal_index=dataset.get_internal_index(),
            dataset_obj=dataset,
            qrels=dataset.qrels,
            reference_ratio=split_config.reference_ratio,
            reference_path=split_config.reference_path,
            remove_dup_answer=split_config.remove_dup_answer,
            select_top_1=split_config.select_top_1,
        )
        return splitter.split(save=split_config.save, verbose=split_config.verbose)
    elif reference_method == "random":
        splitter = RandomSplit(
            dataset_name=dataset.name,
            internal_index=dataset.get_internal_index(),
            reference_ratio=split_config.reference_ratio,
            reference_path=split_config.reference_path,
        )
        return splitter.split(save=split_config.save, verbose=split_config.verbose)
    else:
        raise ValueError(f"Invalid split method: {reference_method}")