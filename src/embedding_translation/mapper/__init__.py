"""Mapper package — one subfolder per mapping strategy.

Ported:
    procrustes, cca, nonlinear, la2m, gromov_wasserstein  (from VectorMerge)
    linear, simple_linear, hmoe                            (from VectorTranslation)
"""

from .manager import VectorSpaceMapper
from .wrapper import LA2MMapper, ProcrustesMapper

# VM strategies
from .cca import CCAMappingStrategy
from .gromov_wasserstein import GromovWassersteinMappingStrategy
from .la2m import LA2MStrategy
from .nonlinear import NonLinearMappingStrategy
from .procrustes import ProcrustesMappingStrategy

# VT strategies (Phase 2)
from .linear import LinearMapper
from .simple_linear import SimpleLinearMapper
from .hmoe import HMoEMapper

SUPPORTED_MAPPING_METHODS = [
    "procrustes",
    "cca",
    "nonlinear",
    "la2m",
    "gromov_wasserstein",
    "linear",
    "simple_linear",
    "hmoe",
]

__all__ = [
    # Manager + convenience wrappers
    "VectorSpaceMapper",
    "LA2MMapper",
    "ProcrustesMapper",
    # Strategy classes
    "ProcrustesMappingStrategy",
    "CCAMappingStrategy",
    "NonLinearMappingStrategy",
    "LA2MStrategy",
    "GromovWassersteinMappingStrategy",
    "LinearMapper",
    "SimpleLinearMapper",
    "HMoEMapper",
    # Registry
    "SUPPORTED_MAPPING_METHODS",
]
