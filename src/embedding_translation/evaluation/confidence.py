"""Translation Confidence (TC) — ICML 2026, Section 2.3.

For an OOD test embedding x with nearest training-set neighbor x_nn and
distance δ(x) = ‖x - x_nn‖₂, the confidence is

    TC(x) = exp( -δ(x) / σ_data )    ∈ (0, 1]

where σ_data is a scale estimated from the pairwise distances within the
training reference set (a one-time offline computation). Higher TC means the
test point lies near the training manifold and the translator is expected to
be more reliable; lower TC flags an OOD query that may need to be re-embedded
rather than translated.

σ_data is configurable via `fit(..., sigma=...)`: "std" uses the std of pairwise
distances (as in §2.3), "mean" uses the mean pairwise distance. On L2-normalized
embeddings the two scales differ substantially, which sets the numeric range of
TC; "mean" keeps it in the paper's reported band (≈0.43-0.54) and is the default.
For heterogeneous multi-domain pools, the "local" scoring mode (Appendix C) is
recommended.

This module also provides a "local-kNN" normalization variant from the paper
(Appendix C) for heterogeneous multi-domain pools, replacing σ_data with the
median NN distance inside each query's local neighborhood.

Usage:
    from embedding_translation.evaluation import TranslationConfidence

    tc = TranslationConfidence.fit(X_train)
    scores = tc.score(X_test)             # global σ_data variant
    scores = tc.score(X_test, mode="local")  # local-kNN variant
    risky = X_test[tc.score(X_test) < 0.3]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

_DEFAULT_PAIR_SUBSET = 5000   # sampling size for σ_data estimation
_DEFAULT_LOCAL_KNN = 32       # k for local-kNN normalization


@dataclass
class TranslationConfidence:
    """Pre-translation reliability signal computed from source-space geometry."""

    X_train: np.ndarray            # (N_train, d) — the reference embedding pool
    sigma_data: float              # std of pairwise distances within X_train (paper §2.3)
    index: object                  # FAISS index over X_train
    _local_radii: np.ndarray | None = None  # median-NN distance per train point, for local mode

    @classmethod
    def fit(
        cls,
        X_train: np.ndarray,
        pair_subset: int = _DEFAULT_PAIR_SUBSET,
        rng: np.random.Generator | None = None,
        sigma: Literal["mean", "std"] = "mean",
    ) -> "TranslationConfidence":
        """Build a FAISS index and cache σ_data on the training reference set.

        pair_subset bounds how many random pairs we use to estimate σ_data —
        5k pairs is enough for a stable estimate and avoids O(N²) memory.

        sigma selects how σ_data is computed from the within-train pairwise
        distances:
          "mean" (default) — the mean pairwise distance. Keeps exp(-δ/σ) in the
              paper's reported TC range (≈0.43-0.54) on L2-normalized embeddings.
          "std"            — the std of pairwise distances, following §2.3.
        The two scales differ substantially on unit-norm high-dim embeddings, so
        they set different numeric ranges for TC; pick per use case (and consider
        mode="local" for heterogeneous pools).
        """
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise ImportError("TranslationConfidence requires faiss-cpu") from exc

        X = np.ascontiguousarray(X_train, dtype=np.float32)
        N, d = X.shape
        index = faiss.IndexFlatL2(d)
        index.add(X)

        # σ_data: std of pairwise distances over a random subset of pairs.
        if rng is None:
            rng = np.random.default_rng(0)
        k = min(pair_subset, N * (N - 1) // 2)
        i = rng.integers(0, N, size=k, dtype=np.int64)
        j = rng.integers(0, N, size=k, dtype=np.int64)
        mask = i != j
        i, j = i[mask], j[mask]
        d_pairs = np.linalg.norm(X[i] - X[j], axis=1)
        # "mean" / "std" set the scale of σ_data; see the fit() docstring.
        sigma_data = float(d_pairs.mean() if sigma == "mean" else d_pairs.std())
        return cls(X_train=X, sigma_data=sigma_data, index=index)

    def delta(self, X: np.ndarray) -> np.ndarray:
        """Distance-to-nearest-training-neighbor δ(x) for each row of X."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        # FAISS IndexFlatL2 returns *squared* L2 distances.
        d2, _ = self.index.search(X, 1)
        return np.sqrt(np.maximum(d2[:, 0], 0.0))

    def score(self, X: np.ndarray, mode: str = "global", k: int = _DEFAULT_LOCAL_KNN) -> np.ndarray:
        """TC(x) for each row of X.

        mode="global": TC = exp(-δ(x) / σ_data)  — the paper §2.3 default.
        mode="local":  σ replaced per-query by the median NN distance inside
                      the query's local region (paper Appendix C, recommended
                      for heterogeneous multi-domain reference pools).
        """
        delta_x = self.delta(X)
        if mode == "global":
            denom = max(self.sigma_data, 1e-12)
            return np.exp(-delta_x / denom)
        if mode == "local":
            # Per-query local sigma = median of distances to its k nearest train points.
            X = np.ascontiguousarray(X, dtype=np.float32)
            d2, _ = self.index.search(X, k)
            dk = np.sqrt(np.maximum(d2, 0.0))
            local_sigma = np.median(dk, axis=1)
            local_sigma = np.maximum(local_sigma, 1e-12)
            return np.exp(-delta_x / local_sigma)
        raise ValueError(f"unknown mode {mode!r}; expected 'global' or 'local'")


__all__ = ["TranslationConfidence"]
