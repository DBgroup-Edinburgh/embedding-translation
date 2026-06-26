"""Numpy-array adapter that quacks like VectorTranslation's MultiMemmapDatasetLoader.

The hmoe inner mappers were written to consume VT's streaming
MultiMemmapDatasetLoader. Rather than port the loader (which depends on disk
memmaps), we wrap in-memory numpy arrays in a shim with the same interface:
``source_embedding_dim``, ``target_embedding_dim``, ``total_samples``,
``load_batch(start, end, return_target=True)``, and iteration yielding
``(src_tensor, tgt_tensor)`` batches. The hmoe mappers don't need the disk
memmap machinery for the small-scale fit paths we exercise.
"""

from __future__ import annotations

import os

import numpy as np
import torch


class _InnerDataset:
    """Minimal dataset object exposing .source_embeddings / .target_embeddings."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source_embeddings = source
        self.target_embeddings = target


def _as_fp32_lazy(arr: np.ndarray) -> np.ndarray:
    """Keep the array lazy; DO NOT force-copy a memmap-backed fp32/fp16 array.

    Calling ``np.ascontiguousarray(memmap, dtype=fp32)`` on a Fever-scale
    (5.4M × dim) memmap reads the entire file into RAM up front — defeating
    mmap loading. fp32 *and* fp16 contiguous arrays are kept as the original
    view; ``__iter__`` fancy-indexes one batch at a time and ``_make_batch``
    casts that batch to fp32. fp16-on-disk halves the footprint so the full
    pair (~80 GB vs 161 GB) fits in the page cache → no disk after warmup.
    """
    if arr.dtype in (np.float32, np.float16) and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


class ArrayDatasetLoader:
    """In-memory shim with the MultiMemmapDatasetLoader interface."""

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        pin_memory: bool = False,
        num_workers: int | None = None,
        prefetch: int | None = None,
        indices: np.ndarray | None = None,
    ):
        if source.shape[0] != target.shape[0]:
            raise ValueError(
                f"source and target must have equal first-dim; "
                f"got {source.shape[0]} vs {target.shape[0]}"
            )
        # Avoid copying a memmap-backed fp32 array (see _as_fp32_lazy).
        self.source = _as_fp32_lazy(source)
        self.target = _as_fp32_lazy(target)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self._rng = np.random.default_rng(seed)
        # Threaded prefetch: worker threads gather+copy+pin upcoming batches
        # while the GPU computes the current one, so the GPU isn't starved by
        # this loader (the base-stage bottleneck). numpy fancy-index/memcpy
        # and pin_memory release the GIL, so threads parallelise the I/O+copy.
        # Tune via HMOE_ARRAY_WORKERS (0 = old synchronous path).
        self.num_workers = (
            num_workers if num_workers is not None
            else int(os.environ.get("HMOE_ARRAY_WORKERS", "8"))
        )
        self.prefetch = prefetch if prefetch is not None else max(2 * self.num_workers, 4)

        # Optional row subset: iterate only these global indices (e.g. a LoRA
        # leaf cluster) while keeping source/target as the full (memmap) arrays.
        # Batches fancy-index the subset rows in one numpy gather per batch
        # instead of one Python __getitem__ per row.
        self._pool = None if indices is None else np.asarray(indices)

        # MultiMemmapDatasetLoader-compatible properties
        self.source_embedding_dim: int = int(source.shape[1])
        self.target_embedding_dim: int = int(target.shape[1])
        self.total_samples: int = (
            int(source.shape[0]) if self._pool is None else int(len(self._pool))
        )

        # Multi-dataset concatenation interface: we expose a single inner
        # dataset, so all global indices map to ds_idx=0, local_idx=global_idx.
        self.datasets = [_InnerDataset(self.source, self.target)]

    def _find_dataset_idx(self, global_idx: int) -> tuple[int, int]:
        """Return (dataset_idx, local_idx). Single-dataset shim → always (0, global_idx)."""
        return 0, int(global_idx)

    def load_batch(
        self, start_idx: int, end_idx: int | None = None, return_target: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return numpy slices to match what hmoe inner code expects.

        The inner code calls torch.from_numpy(src[0]) and similar; VT's loader
        returns numpy, so we match that contract here.
        """
        if end_idx is None:
            end_idx = min(start_idx + self.batch_size, self.total_samples)
        end_idx = min(end_idx, self.total_samples)
        src = self.source[start_idx:end_idx]
        if return_target:
            return src, self.target[start_idx:end_idx]
        return src

    def _make_batch(self, idx: np.ndarray):
        # Fancy-indexing into a memmap pulls only these rows from disk, then
        # returns a fresh in-memory ndarray (writable, contiguous). ``.float()``
        # casts fp16-on-disk batches up to fp32 (no-op for fp32 input), so the
        # backing file can be half-size and cache-resident.
        src_b = torch.from_numpy(np.ascontiguousarray(self.source[idx])).float()
        tgt_b = torch.from_numpy(np.ascontiguousarray(self.target[idx])).float()
        if self.pin_memory and torch.cuda.is_available():
            src_b = src_b.pin_memory()
            tgt_b = tgt_b.pin_memory()
        return src_b, tgt_b

    def __iter__(self):
        # `order` holds GLOBAL row indices to visit: the subset pool when given,
        # else all rows. Batches fancy-index source/target at these indices.
        order = self._pool.copy() if self._pool is not None else np.arange(self.total_samples)
        if self.shuffle:
            self._rng.shuffle(order)
        index_batches = [
            order[i : i + self.batch_size]
            for i in range(0, len(order), self.batch_size)
        ]

        if self.num_workers <= 0:
            # Synchronous fallback (old behaviour).
            for idx in index_batches:
                yield self._make_batch(idx)
            return

        # ORDER-PRESERVING threaded prefetch: a sliding window of `prefetch`
        # batches is submitted to a thread pool in order; we yield results in
        # submission order (popleft). This overlaps each batch's gather/copy/
        # pin with GPU compute on the previous batch — keeping the GPU fed —
        # while producing the *exact same batch sequence* as the synchronous
        # loader, so order-dependent consumers (e.g. the clustering pass) are
        # unaffected.
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        ex = ThreadPoolExecutor(max_workers=self.num_workers)
        try:
            it = iter(index_batches)
            inflight: deque = deque()
            for _ in range(max(self.prefetch, 1)):
                try:
                    inflight.append(ex.submit(self._make_batch, next(it)))
                except StopIteration:
                    break
            while inflight:
                fut = inflight.popleft()
                try:
                    inflight.append(ex.submit(self._make_batch, next(it)))
                except StopIteration:
                    pass
                yield fut.result()
        finally:
            ex.shutdown(wait=False)

    def __len__(self) -> int:
        return (self.total_samples + self.batch_size - 1) // self.batch_size
