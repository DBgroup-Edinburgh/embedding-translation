"""Configuration package — pydantic only.

Public surface:
    from embedding_translation.config import (
        MappingConfig, ClusteringConfig, EmbeddingModelConfig,
        ProcrustesConfig, CCAConfig, LA2MConfig,
        NonLinearMappingConfig, GromovWassersteinConfig,
        Settings, get_settings,
        load, save,
    )
"""

from .loader import load, save
from .models import (
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
    SimpleLinearMapperConfig,
)
from .settings import Settings, get_settings

__all__ = [
    # Domain configs
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
    # Settings
    "Settings",
    "get_settings",
    # Loader
    "load",
    "save",
]
