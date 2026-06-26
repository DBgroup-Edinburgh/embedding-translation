"""Hierarchical Mixture-of-Experts (hmoe) mapping.

Public surface: :class:`HMoEMapper` — a wrapper that conforms to
:class:`embedding_translation.core.mapping.MappingStrategy` and dispatches to
one of three inner implementations based on ``MappingConfig.hmoe_config.moe_type``:

    - ``flat``              → :class:`FlatMoEMapper`              (single-level experts)
    - ``hierarchical``      → :class:`HierarchicalMoEMapper`      (tree experts)
    - ``hierarchical_lora`` → :class:`HierarchicalLoRAMoEMapper`  (tree + shared base + LoRA, default)

The internal implementations are ported from
``VectorTranslation/src/mapper/strategy/gating_moe/`` and retain VT's internal
``VectorMapper`` ABC (kept private to this package) so we didn't have to rewrite
the whole training pipeline. The :class:`HMoEMapper` wrapper translates between
VT's constructor style and our pydantic-config style.
"""

from __future__ import annotations

import numpy as np

from ...config import MappingConfig
from ...core.mapping import MappingStrategy

# Internal building blocks (kept exported for advanced users)
from .base_mapper import BaseMoEMapper
from .expert_clusterer import ExpertClusterer
from .gating_mechanism import GatingMechanism
from .router import CascadeRouter, FlatRouter
from .tree_structure import (
    BottomUpHierarchyTree,
    HierarchyNode,
    HierarchyTree,
    TreeNode,
)

# Three inner mappers
from .flat.mapper import FlatMoEMapper
from .hierarchical.mapper import HierarchicalMoEMapper
from .hierarchical_lora.mapper import HierarchicalLoRAMoEMapper


class HMoEMapper(MappingStrategy):
    """Public MappingStrategy wrapper around hmoe's three inner variants.

    Reads ``config.hmoe_config`` and constructs the matching inner mapper, then
    delegates ``fit`` and ``transform``. Use this class everywhere outside
    ``embedding_translation.mapper.hmoe``; the inner classes are exposed for
    debugging / experimentation only.
    """

    def __init__(self, config: MappingConfig, dir_channel: str = "u1"):
        """Construct an H-MoE mapper.

        dir_channel selects which PCA channel L_dir aligns to:
            "u1" (default) — mixing channel, used for pairwise and the
                             upstream translator in chaining.
            "u2"           — chaining channel, used for the downstream
                             (second-hop) translator in s → Hub → t.
        """
        super().__init__(config)
        cfg = config.hmoe_config
        self._moe_type = cfg.moe_type
        self._dir_channel = dir_channel

        # The inner mappers take the SimpleLinearMapper config separately —
        # pass our pydantic SimpleLinearMapperConfig directly; they call
        # `.model_dump()` and splat the kwargs into the inner SimpleLinearMapper.
        expert_cfg = cfg.mapper_config

        if cfg.moe_type == "flat":
            self._inner: BaseMoEMapper = FlatMoEMapper(
                num_experts=cfg.num_experts,
                mapper_config=expert_cfg,
                distance_metric=cfg.distance_metric,
                clustering_method=cfg.clustering_method,
                random_state=cfg.random_state,
                clustering_sample_size=cfg.clustering_sample_size,
                use_soft_routing=cfg.use_soft_routing,
                gating_temperature=cfg.gating_temperature,
            )
        elif cfg.moe_type == "hierarchical":
            self._inner = HierarchicalMoEMapper(
                num_levels=cfg.num_levels,
                branch_factor=cfg.branch_factor,
                mapper_config=expert_cfg,
                distance_metric=cfg.distance_metric,
            )
        elif cfg.moe_type == "hierarchical_lora":
            self._inner = HierarchicalLoRAMoEMapper(
                num_levels=cfg.num_levels,
                branch_factor=cfg.branch_factor,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                share_base_model=cfg.share_base_model,
                base_model_epochs=cfg.base_model_epochs,
                lora_epochs=cfg.lora_epochs,
                mapper_config=expert_cfg,
                distance_metric=cfg.distance_metric,
                # ICML 2026 knobs
                alpha=cfg.alpha,
                beta=cfg.beta,
                beta_base=cfg.beta_base,
                tau=cfg.tau,
                local_nn_m=cfg.local_nn_m,
                base_loss=cfg.base_loss,
                dir_channel=dir_channel,
                dir_norm=cfg.dir_norm,
                local_anchors=cfg.local_anchors,
                retr_weight=cfg.retr_weight,
                retr_tau=cfg.retr_tau,
                retr_pool_size=cfg.retr_pool_size,
            )
        else:
            raise ValueError(f"Unknown moe_type {cfg.moe_type!r}")

    def fit(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        reference_indices: np.ndarray,
        **kwargs: object,
    ) -> None:
        self._inner.fit(source_embeddings, target_embeddings, reference_indices)
        self.metadata["moe_type"] = self._moe_type
        self.is_fitted = True

    def transform(self, embeddings: np.ndarray, **_: object) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HMoEMapper must be fit before transform")
        return self._inner.transform(embeddings)


__all__ = [
    "HMoEMapper",
    # Inner mappers (advanced users)
    "FlatMoEMapper",
    "HierarchicalMoEMapper",
    "HierarchicalLoRAMoEMapper",
    # Building blocks
    "BaseMoEMapper",
    "ExpertClusterer",
    "GatingMechanism",
    "FlatRouter",
    "CascadeRouter",
    "HierarchyTree",
    "HierarchyNode",
    "BottomUpHierarchyTree",
    "TreeNode",
]
