"""Procrustes mapping — orthogonal transform learned by SVD on paired anchors."""

from .strategy import ProcrustesMappingStrategy, procrustes_mapping_torch

__all__ = ["ProcrustesMappingStrategy", "procrustes_mapping_torch"]
