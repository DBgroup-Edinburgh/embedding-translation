"""Pydantic configuration models for embedding_translation.

Every config in this repo is a pydantic.BaseModel. No dataclasses, no dacite.
Sub-models per mapper strategy live alongside the strategy code (e.g.
`mapper/procrustes/__init__.py` exports `ProcrustesConfig`) and are aggregated
here under MapperConfig as a discriminated set of optional fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class KMeansConfig(BaseModel):
    n_clusters: int = Field(10, ge=1)
    max_iter: int = Field(300, ge=1)
    tol: float = 1e-4
    random_state: int = 42


class LA2MClusteringConfig(BaseModel):
    d_prime: int = Field(10, ge=1)


class ClusteringConfig(BaseModel):
    clustering_method: Literal["kmeans", "la2m-cluster"] = "la2m-cluster"
    device: str = "auto"
    verbose: bool = False
    compute_metrics: bool = True
    la2m_config: LA2MClusteringConfig = Field(default_factory=LA2MClusteringConfig)
    kmeans_config: KMeansConfig = Field(default_factory=KMeansConfig)

    def to_string(self) -> str:
        return str(self.model_dump())


# ---------------------------------------------------------------------------
# Per-mapper configs (the small, foundational ones from VectorMerge)
# Larger mappers (hmoe) ship their own sub-models alongside the strategy.
# ---------------------------------------------------------------------------


class LA2MConfig(BaseModel):
    num_clusters: int = Field(50, ge=1)
    cluster_method: str = "la2m-cluster"
    d_prime: int = Field(10, ge=1)
    pca_mapping: bool = True
    pca_dim: int = Field(14, ge=1)
    use_norm: bool = False


class NonLinearMappingConfig(BaseModel):
    num_layers: int = Field(3, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(0.001, gt=0)
    num_epochs: int = Field(100, ge=1)
    hidden_dim: int = Field(512, ge=1)
    dropout_rate: float = Field(0.1, ge=0, le=1)
    loss_type: str = "mse"


class ProcrustesConfig(BaseModel):
    approximate: bool = False
    q: int = Field(1500, ge=1)
    with_rotation: bool = True
    with_scaling: bool = True
    use_pca: bool = False
    reduced_dim: int = 0
    procrustes_pca_type: Literal["none", "source", "target", "both"] = "none"
    use_norm: bool = True


class CCAConfig(BaseModel):
    n_components: int = Field(10, ge=1)
    scale: bool = True
    max_iter: int = Field(500, ge=1)
    tol: float = 1e-6
    # Avoid shadowing BaseModel.copy(). Serialised as "copy" so existing YAMLs
    # continue to work; access in code via cfg.copy_input.
    copy_input: bool = Field(True, alias="copy")

    model_config = {"populate_by_name": True}


class GromovWassersteinConfig(BaseModel):
    loss_fun: Literal["square_loss", "kl_loss"] = "square_loss"
    max_iter: int = Field(1000, ge=1)
    tol: float = 1e-9
    verbose: bool = False
    log: bool = False
    armijo: bool = False
    epsilon: float = 0.1
    symmetric: bool = True
    G0: Literal["uniform", "random"] = "uniform"


# ---------------------------------------------------------------------------
# Aggregate mapper config — holds general settings + per-strategy sub-models.
# Strategy is selected by name; the corresponding sub-model is consulted.
# ---------------------------------------------------------------------------


MapperStrategy = Literal[
    "procrustes",
    "cca",
    "simple_linear",
    "linear",
    "nonlinear",
    "gromov_wasserstein",
    "la2m",
    "hmoe",
]


# ---------------------------------------------------------------------------
# VT mapper configs (ported as-is from VectorTranslation/src/config/models.py).
# Already pydantic upstream — only field-validator decorators needed a touch-up
# for our pydantic v2 version.
# ---------------------------------------------------------------------------


class LinearMapperConfig(BaseModel):
    hidden_dim: int = Field(1024, ge=1)
    num_epochs: int = Field(1000, ge=1)
    batch_size: int = Field(320, ge=1)
    learning_rate: float = Field(1e-4, gt=0)
    triplet_margin: float = 0.5
    rank_margin: float = 0.1
    lambda_: float = 1.0
    hierarchy_k: int = Field(10, ge=1)
    hierarchy_weight_mode: Literal["linear", "softmax"] = "linear"
    loss_type: Literal["cos", "cos_triplet_hierarchy", "triplet", "hierarchy", "mse"] = "cos"


class SimpleLinearMapperConfig(BaseModel):
    learning_rate: float = Field(1e-4, gt=0)
    num_epochs: int = Field(50, ge=1)
    batch_size: int = Field(4028, ge=1)
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    scheduler_patience: int = Field(5, ge=1)
    scheduler_factor: float = 0.5
    early_stopping_patience: int = Field(10, ge=1)
    min_delta: float = 1e-6
    device: str | None = None
    layer_num: int = Field(4, ge=1)  # paper backbone is 4-layer
    activation: Literal["relu", "gelu", "tanh", "leaky_relu", "selu"] = "selu"  # paper uses SELU
    dropout: float = Field(0.0, ge=0, le=1)  # paper does not mention dropout; keep off
    hidden_dim: int = Field(512, ge=1)
    use_local_distill: bool = False
    local_k: int = Field(50, ge=1)
    local_tau: float = 0.1
    local_weight: float = Field(0.5, ge=0, le=1)
    faiss_use_float32: bool = True
    knn_recompute_epochs: int = Field(0, ge=0)
    global_weight: float = 0.5
    # Loss/output knobs (added for tuning the hmoe inner mapper)
    cosine_weight: float = 0.5
    mse_weight: float = 1.0
    normalize_output: bool = False
    # ICML 2026 Stage-1 base loss: "l1" uses ‖f(x)-y‖_1 (paper Algorithm 1),
    # "mse_cos" is the legacy hybrid, "cos" pure cosine.
    loss_kind: Literal["mse_cos", "l1", "cos", "mse", "cos_retr"] = "mse_cos"


class GatingMoEConfig(BaseModel):
    """Hierarchical MoE (hmoe) mapper config.

    Three variants selected by `moe_type`:
      - flat:              single-level expert ensemble (kept as backstop)
      - hierarchical:      tree-based experts, bottom-up clustering
      - hierarchical_lora: hierarchical experts with a shared base model
                           plus per-leaf LoRA adapters (the headline variant)

    The paper-canonical configuration (ICML 2026, Section 5.1) is:
        moe_type = "hierarchical_lora"
        num_leaves K = 4 (8 for Fever)            ← controlled via branch_factor / num_levels
        lora_rank r = 8
        alpha = 0.5     (local-structure loss weight)
        beta = 0.7      (directional residual loss weight)
        tau = 0.8       (routing ambiguity threshold)
        m = 100         (NN count for L_local)
        learning_rate = 1e-4
        base loss = L1, freeze, then LoRA stage with L_reg + α·L_local + β·L_dir
    """

    moe_type: Literal["flat", "hierarchical", "hierarchical_lora"] = "hierarchical_lora"

    # Common
    num_experts: int = Field(8, ge=1)
    clustering_method: Literal["kmeans", "minibatch_kmeans", "agglomerative"] = "agglomerative"
    distance_metric: Literal["cosine", "l2", "euclidean"] = "euclidean"
    random_state: int = 42
    clustering_sample_size: int = Field(100_000, ge=1)

    # Flat routing
    use_soft_routing: bool = False
    gating_temperature: float = 1.0

    # Hierarchical (defaults give K=4 leaves; for Fever use num_levels=4, branch_factor=2 → K=8)
    num_levels: int = Field(3, ge=2)      # 3 levels with branch_factor=2 → 4 leaves
    branch_factor: int = Field(2, ge=2)

    # LoRA
    lora_rank: int = Field(8, ge=1)
    lora_alpha: int = Field(16, ge=1)
    lora_dropout: float = Field(0.0, ge=0, le=1)
    share_base_model: bool = True
    base_model_epochs: int = Field(200, ge=1)
    lora_epochs: int = Field(100, ge=1)

    # Paper hyperparameters (Algorithm 1 + Algorithm 2)
    alpha: float = Field(0.5, ge=0)       # L_local weight (LoRA stage)
    beta: float = Field(0.7, ge=0)        # L_dir weight at LoRA stage
    beta_base: float = Field(0.0, ge=0)   # L_dir weight at Stage-1 base training
                                          # (0 = base trained without L_dir; ≥0
                                          # injects β_base · L_dir during L1 base
                                          # training so the base is already
                                          # directionally regularized before LoRA)
    tau: float = Field(0.8, ge=0, le=1)   # routing ambiguity threshold
    local_nn_m: int = Field(100, ge=1)    # NN count for L_local
    base_loss: Literal["l1", "mse", "cos", "mse_cos", "cos_retr"] = "l1"  # Stage-1 base translator
    # L_dir normalization. "fraction" = ‖orth‖²/‖e‖² per sample (scale-invariant).
    # "fixed" = ‖orth‖²/σ² with σ² a precomputed constant (target variance), i.e.
    # the paper's raw off-axis energy under a fixed scale.
    dir_norm: Literal["fraction", "fixed"] = "fraction"
    # L_local anchor subsampling per batch (0 = all rows, paper-faithful).
    local_anchors: int = Field(0, ge=0)
    # H-MoE-specific mixing-aware retr loss: each LoRA expert's InfoNCE uses a
    # GLOBAL pool of native target docs as negatives so it learns not to outrank
    # native docs from other clusters (the multi-model mixing fix). 0 disables.
    retr_weight: float = Field(0.0, ge=0)
    retr_tau: float = Field(0.05, gt=0)
    retr_pool_size: int = Field(2048, ge=1)

    # Expert mapper hyperparams (shared SimpleLinearModel-like architecture)
    mapper_config: SimpleLinearMapperConfig = Field(default_factory=SimpleLinearMapperConfig)


class MappingConfig(BaseModel):
    """Top-level mapping configuration.

    The aggregate config carries general training/device settings plus one
    sub-model per registered strategy. When a strategy is selected (by name),
    its corresponding sub-model is used. Unrelated sub-models are ignored.
    """

    # General settings
    device: str = "auto"
    batch_size: int = Field(32, ge=1)
    verbose: bool = False
    save_param: bool = True
    save_embedding: bool = True

    # Per-strategy sub-configs. One field per registered strategy; only the
    # field matching the active strategy is consulted.
    la2m_config: LA2MConfig = Field(default_factory=LA2MConfig)
    nonlinear_config: NonLinearMappingConfig = Field(default_factory=NonLinearMappingConfig)
    procrustes_config: ProcrustesConfig = Field(default_factory=ProcrustesConfig)
    cca_config: CCAConfig = Field(default_factory=CCAConfig)
    gromov_wasserstein_config: GromovWassersteinConfig = Field(
        default_factory=GromovWassersteinConfig
    )
    # Phase 2 additions (VT mappers)
    linear_config: LinearMapperConfig = Field(default_factory=LinearMapperConfig)
    simple_linear_config: SimpleLinearMapperConfig = Field(
        default_factory=SimpleLinearMapperConfig
    )
    hmoe_config: GatingMoEConfig = Field(default_factory=GatingMoEConfig)

    def to_string(self) -> str:
        return str(self.model_dump())


# ---------------------------------------------------------------------------
# Embedding model config (foundation for vectorbench adapter)
# ---------------------------------------------------------------------------


class EmbeddingModelConfig(BaseModel):
    model_name: str
    batch_size: int = Field(32, ge=1)
    device: str = "cpu"
    max_tokens: int = Field(4096, ge=1)

    model_config = {"protected_namespaces": ()}  # allow `model_name` as a field


# ---------------------------------------------------------------------------
# Backwards-compatibility helpers — drop after callers migrate
# ---------------------------------------------------------------------------


def _to_dict(cfg: BaseModel) -> dict[str, Any]:
    """Bridge for code still calling `.to_dict()` on configs."""
    return cfg.model_dump()


def _from_dict(cls: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    """Bridge for code still calling `Cls.from_dict(d)`."""
    return cls.model_validate(data)
