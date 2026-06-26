"""embedding_translation — unified library for translating embeddings between vector spaces.

Supersedes VectorMerge + VectorTranslation. Datasets and embedding generation
delegate to vectorbench; everything else (mappers, losses, clustering,
reference splits, evaluation, analysis, pipelines) lives here.
"""

__version__ = "0.1.0"
__author__ = "Beining Yang"
__email__ = "suchunsv@outlook.com"

# Configuration (pydantic-only)
from .config import (
    CCAConfig,
    ClusteringConfig,
    EmbeddingModelConfig,
    GatingMoEConfig,
    GromovWassersteinConfig,
    KMeansConfig,
    LA2MClusteringConfig,
    LA2MConfig,
    LinearMapperConfig,
    MapperStrategy,
    MappingConfig,
    NonLinearMappingConfig,
    ProcrustesConfig,
    Settings,
    SimpleLinearMapperConfig,
    get_settings,
)

# Mapping
from .mapper import (
    SUPPORTED_MAPPING_METHODS,
    CCAMappingStrategy,
    GromovWassersteinMappingStrategy,
    HMoEMapper,
    LA2MMapper,
    LA2MStrategy,
    LinearMapper,
    NonLinearMappingStrategy,
    ProcrustesMapper,
    ProcrustesMappingStrategy,
    SimpleLinearMapper,
    VectorSpaceMapper,
)

# Clustering
from .clustering import (
    SUPPORTED_CLUSTERING_METHODS,
    ClusterData,
    ClusterManager,
    ClusteringResult,
    ClusteringStrategy,
    KMeansClusteringStrategy,
    LA2MClusteringStrategy,
)

__all__ = [
    # Config
    "MappingConfig",
    "MapperStrategy",
    "ClusteringConfig",
    "KMeansConfig",
    "LA2MClusteringConfig",
    "EmbeddingModelConfig",
    "ProcrustesConfig",
    "CCAConfig",
    "LA2MConfig",
    "NonLinearMappingConfig",
    "GromovWassersteinConfig",
    "LinearMapperConfig",
    "SimpleLinearMapperConfig",
    "GatingMoEConfig",
    "Settings",
    "get_settings",
    # Mapping
    "VectorSpaceMapper",
    "LA2MMapper",
    "ProcrustesMapper",
    "ProcrustesMappingStrategy",
    "CCAMappingStrategy",
    "NonLinearMappingStrategy",
    "LA2MStrategy",
    "GromovWassersteinMappingStrategy",
    "LinearMapper",
    "SimpleLinearMapper",
    "HMoEMapper",
    "SUPPORTED_MAPPING_METHODS",
    # Clustering
    "ClusterManager",
    "ClusteringStrategy",
    "ClusteringResult",
    "ClusterData",
    "KMeansClusteringStrategy",
    "LA2MClusteringStrategy",
    "SUPPORTED_CLUSTERING_METHODS",
    # Metadata
    "__version__",
]
