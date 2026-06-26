"""Dataset adapter — delegates to vectorbench.

`Dataset`, `load_dataset`, and the supported-dataset registry come from
`vectorbench.dataset`. We import lazily so that callers who never touch the
dataset surface can use the rest of the library without vectorbench installed.

Do NOT import `vectorbench.embedding_dataset` from here — it transitively
pulls in vectormerge, which this repo eliminates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vectorbench.dataset.base import Dataset as Dataset  # noqa: F401


def _require_vectorbench() -> Any:
    try:
        import vectorbench  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "embedding_translation.dataset requires `vectorbench`. "
            "It is declared as a git dependency in pyproject.toml."
        ) from e
    import vectorbench as vb

    return vb


def __getattr__(name: str) -> Any:
    if name == "Dataset":
        from vectorbench.dataset.base import Dataset as _Dataset

        return _Dataset
    if name == "load_dataset":
        from vectorbench.dataset.loader import load_dataset as _load_dataset

        return _load_dataset
    if name in {"list_datasets", "SUPPORTED_DATASETS"}:
        vb = _require_vectorbench()
        return getattr(vb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Dataset", "load_dataset", "list_datasets", "SUPPORTED_DATASETS"]
