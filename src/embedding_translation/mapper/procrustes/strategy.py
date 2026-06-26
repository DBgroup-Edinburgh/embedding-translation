"""
Procrustes mapping strategy for VectorMerge.

This module implements Procrustes analysis for mapping between embedding spaces.
The implementation is based on the reference code and provides both standard
and approximated versions of the algorithm.
"""

import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger
from sklearn.decomposition import PCA

from ...core.mapping import MappingStrategy
from ...config import MappingConfig

def procrustes_mapping_torch(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    overlap_ids: np.ndarray,
    source_bound: np.ndarray,
    target_bound: np.ndarray,
    approximate: bool = False,
    q: int = 1500,
    with_rotation: bool = True,
    with_scaling: bool = True
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Procrustes mapping using PyTorch for GPU acceleration.
    
    Args:
        source_embeddings: Source embeddings (N x D1)
        target_embeddings: Target embeddings (N x D2)
        overlap_ids: Indices of overlapping points
        source_bound: Source embeddings to be transformed
        target_bound: Target embeddings (for reference)
        approximate: Whether to use approximated SVD
        q: Number of components for approximation
        with_rotation: Whether to include rotation in the transformation
        with_scaling: Whether to include scaling/normalization in the transformation
        
    Returns:
        Tuple of (transformed_embeddings, transformation_params)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to torch tensors
    X = torch.from_numpy(source_embeddings[overlap_ids]).float().to(device)
    Y = torch.from_numpy(target_embeddings[overlap_ids]).float().to(device)
    source_bound_tensor = torch.from_numpy(source_bound).float().to(device)
    
    # Center the data
    X_mean = torch.mean(X, dim=0)
    Y_mean = torch.mean(Y, dim=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    # Add scaling/normalization if requested
    if with_scaling:
        X_norm = torch.norm(X_centered)
        Y_norm = torch.norm(Y_centered)
        X_centered = X_centered / X_norm
        Y_centered = Y_centered / Y_norm
    else:
        X_norm = 1.0
        Y_norm = 1.0
    
    if approximate and X_centered.shape[0] > q:
        # Use randomized SVD for large matrices
        logger.info(f"Using approximated SVD with q={q}")
        H = torch.matmul(Y_centered.T, X_centered)
        
        # Randomized SVD approximation
        U, S, Vt = torch.linalg.svd(H, full_matrices=False)
        if U.shape[1] > q:
            U = U[:, :q]
            S = S[:q]
            Vt = Vt[:q, :]
    else:
        # Standard SVD
        H = torch.matmul(Y_centered.T, X_centered)
        U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    
    if with_rotation:
        # Compute rotation matrix
        R = torch.matmul(U, Vt)
    else:
        # Identity rotation
        R = torch.eye(min(U.shape[0], Vt.shape[0])).to(device)
    
    # Apply transformation: center, scale, rotate, then translate
    source_bound_centered = source_bound_tensor - X_mean
    if with_scaling:
        source_bound_centered = source_bound_centered / X_norm
    
    transformed = torch.matmul(source_bound_centered, R.T)
    
    if with_scaling:
        transformed = transformed * Y_norm
    
    transformed = transformed + Y_mean
    
    # Prepare transformation parameters
    params = {
        'rotation_matrix': R.cpu().numpy(),
        'source_mean': X_mean.cpu().numpy(),
        'target_mean': Y_mean.cpu().numpy(),
        'with_scaling': with_scaling
    }
    
    if with_scaling:
        params['source_norm'] = X_norm.cpu().item()
        params['target_norm'] = Y_norm.cpu().item()
    
    return transformed.cpu().numpy(), params


def procrustes_no_norm_scale_with_param(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    overlap_ids: np.ndarray,
    source_bound: np.ndarray,
    target_bound: np.ndarray,
    approximate: bool = False,
    q: int = 1500,
    with_rotation: bool = True,
    params: Optional[dict] = None
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Procrustes mapping without normalization and scaling, with parameter reuse.
    
    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings
        overlap_ids: Overlapping point indices
        source_bound: Source embeddings to transform
        target_bound: Target embeddings (reference)
        approximate: Use approximated SVD
        q: Approximation parameter
        with_rotation: Include rotation
        params: Pre-computed transformation parameters
        
    Returns:
        Tuple of (transformed_embeddings, transformation_params)
    """
    if params is not None:
        # Use pre-computed parameters
        R = params['rotation_matrix']
        source_mean = params['source_mean']
        target_mean = params['target_mean']
        
        # Apply transformation
        source_bound_centered = source_bound - source_mean
        transformed = np.dot(source_bound_centered, R.T) + target_mean
        
        return transformed, params
    else:
        # Compute new parameters
        return procrustes_mapping_torch(
            source_embeddings, target_embeddings, overlap_ids,
            source_bound, target_bound, approximate, q, with_rotation
        )


def procrustes_pca_mapping(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    overlap_ids: np.ndarray,
    source_bound: np.ndarray,
    target_bound: np.ndarray,
    approximate: bool = False,
    q: int = 1500,
    with_rotation: bool = True,
    reduced_dim: int = 300
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Procrustes mapping with PCA dimensionality reduction.
    
    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings
        overlap_ids: Overlapping point indices
        source_bound: Source embeddings to transform
        target_bound: Target embeddings (reference)
        approximate: Use approximated SVD
        q: Approximation parameter
        with_rotation: Include rotation
        reduced_dim: Target dimensionality for PCA
        
    Returns:
        Tuple of (transformed_embeddings, transformation_params)
    """
    # Apply PCA to reduce dimensionality
    pca_source = PCA(n_components=reduced_dim)
    pca_target = PCA(n_components=reduced_dim)
    
    # Fit PCA on overlap data
    source_overlap = source_embeddings[overlap_ids]
    target_overlap = target_embeddings[overlap_ids]
    
    pca_source.fit(source_overlap)
    pca_target.fit(target_overlap)
    
    # Transform data
    source_reduced = pca_source.transform(source_bound)
    target_reduced = pca_target.transform(target_bound)
    
    # Apply Procrustes in reduced space
    transformed_reduced, params = procrustes_mapping_torch(
        pca_source.transform(source_embeddings),
        pca_target.transform(target_embeddings),
        overlap_ids, source_reduced, target_reduced,
        approximate, q, with_rotation
    )
    
    # Transform back to original space
    transformed = pca_target.inverse_transform(transformed_reduced)
    
    # Store PCA parameters
    if params:
        params['pca_source'] = pca_source
        params['pca_target'] = pca_target
    
    return transformed, params


class ProcrustesMappingStrategy(MappingStrategy):
    """Procrustes analysis-based mapping strategy."""
    
    def __init__(self, config: MappingConfig):
        """Initialize Procrustes mapping strategy.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.rotation_matrix: Optional[np.ndarray] = None
        self.source_mean: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.scale: Optional[float] = None
        self.pca_source: Optional[PCA] = None
        self.pca_target: Optional[PCA] = None
        
        logger.info(f"Procrustes mapping initialized with approximate={config.procrustes_config.approximate}, "
                   f"with_rotation={config.procrustes_config.with_rotation}, with_scaling={config.procrustes_config.with_scaling}")
    
    def fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
            reference_indices: np.ndarray, **kwargs) -> None:
        """Fit the Procrustes mapping.
        
        Args:
            source_embeddings: Source embedding space
            target_embeddings: Target embedding space  
            reference_indices: Reference point indices for alignment
            **kwargs: Additional arguments
        """
        logger.info(f"Fitting Procrustes mapping with {len(reference_indices)} reference points")
        
        # Extract reference embeddings
        source_ref = source_embeddings[reference_indices]
        target_ref = target_embeddings[reference_indices]
        
        if self.config.procrustes_config.procrustes_pca_type == "inner" and self.config.procrustes_config.reduced_dim > 0:
            # Use PCA-based Procrustes
            _, params = procrustes_pca_mapping(
                source_embeddings, target_embeddings, reference_indices,
                source_ref, target_ref,
                approximate=self.config.procrustes_config.approximate,
                q=self.config.procrustes_config.q,
                with_rotation=self.config.procrustes_config.with_rotation,
                reduced_dim=self.config.procrustes_config.reduced_dim
            )
            
            if params:
                self.pca_source = params.get('pca_source')
                self.pca_target = params.get('pca_target')
        
        elif self.config.procrustes_config.use_norm:
            # Standard Procrustes with normalization
            _, params = procrustes_mapping_torch(
                source_embeddings, target_embeddings, reference_indices,
                source_ref, target_ref,
                approximate=self.config.procrustes_config.approximate,
                q=self.config.procrustes_config.q,
                with_rotation=self.config.procrustes_config.with_rotation
            )
        else:
            # Procrustes without normalization
            _, params = procrustes_no_norm_scale_with_param(
                source_embeddings, target_embeddings, reference_indices,
                source_ref, target_ref,
                approximate=self.config.procrustes_config.approximate,
                q=self.config.procrustes_config.q,
                with_rotation=self.config.procrustes_config.with_rotation
            )
        
        # Store transformation parameters
        if params:
            self.rotation_matrix = params['rotation_matrix']
            self.source_mean = params['source_mean']
            self.target_mean = params['target_mean']
            self.scale = params.get('scale', 1.0)
        
        self.is_fitted = True
        self.metadata = {
            'reference_size': len(reference_indices),
            'approximate': self.config.procrustes_config.approximate,
            'with_rotation': self.config.procrustes_config.with_rotation,
            'with_scaling': self.config.procrustes_config.with_scaling,
            'use_norm': self.config.procrustes_config.use_norm,
            'procrustes_pca_type': self.config.procrustes_config
        }
        
        logger.info("Procrustes mapping fitting completed")
    
    def transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using fitted Procrustes mapping.
        
        Args:
            embeddings: Embeddings to transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed embeddings
        """
        if not self.is_fitted:
            raise ValueError("Mapping must be fitted before transformation")
        
        # Ensure transformation parameters are available
        assert self.rotation_matrix is not None, "Rotation matrix not fitted"
        assert self.source_mean is not None, "Source mean not fitted"
        assert self.target_mean is not None, "Target mean not fitted"
        assert self.scale is not None, "Scale not fitted"
        
        if self.pca_source is not None and self.pca_target is not None:
            # PCA-based transformation
            embeddings_reduced = self.pca_source.transform(embeddings)
            embeddings_centered = embeddings_reduced - self.source_mean
            transformed_reduced = self.scale * np.dot(embeddings_centered, self.rotation_matrix.T) + self.target_mean
            transformed = self.pca_target.inverse_transform(transformed_reduced)
        else:
            # Standard transformation: center, scale, rotate, then translate
            embeddings_centered = embeddings - self.source_mean
            transformed = self.scale * np.dot(embeddings_centered, self.rotation_matrix.T) + self.target_mean
        
        return transformed
    
    def save(self, path) -> None:
        """Save Procrustes mapping parameters."""
        super().save(path)
        
        # Save additional Procrustes-specific parameters
        save_path = Path(path)
        
        if self.rotation_matrix is not None:
            np.save(save_path / "rotation_matrix.npy", self.rotation_matrix)
        if self.source_mean is not None:
            np.save(save_path / "source_mean.npy", self.source_mean)
        if self.target_mean is not None:
            np.save(save_path / "target_mean.npy", self.target_mean)
        if self.scale is not None:
            np.save(save_path / "scale.npy", self.scale)
        
        # Save PCA models if available
        if self.pca_source is not None:
            import joblib
            joblib.dump(self.pca_source, save_path / "pca_source.pkl")
        if self.pca_target is not None:
            import joblib
            joblib.dump(self.pca_target, save_path / "pca_target.pkl")
    
    @classmethod
    def load(cls, path) -> 'ProcrustesMappingStrategy':
        """Load Procrustes mapping parameters."""
        load_path = Path(path)
        
        # Load metadata
        import json
        with open(load_path / "mapping_info.json", "r") as f:
            metadata = json.load(f)
        
        # Create instance
        config = MappingConfig.from_dict(metadata["config"])
        instance = cls(config)
        
        # Load transformation matrix from base class if available
        transformation_path = load_path / "transformation_matrix.npy"
        if transformation_path.exists():
            instance.transformation_matrix = np.load(transformation_path)
        
        # Load Procrustes-specific parameters
        if (load_path / "rotation_matrix.npy").exists():
            instance.rotation_matrix = np.load(load_path / "rotation_matrix.npy")
        if (load_path / "source_mean.npy").exists():
            instance.source_mean = np.load(load_path / "source_mean.npy")
        if (load_path / "target_mean.npy").exists():
            instance.target_mean = np.load(load_path / "target_mean.npy")
        if (load_path / "scale.npy").exists():
            instance.scale = np.load(load_path / "scale.npy")
        
        # Load PCA models if available
        try:
            import joblib
            if (load_path / "pca_source.pkl").exists():
                instance.pca_source = joblib.load(load_path / "pca_source.pkl")
            if (load_path / "pca_target.pkl").exists():
                instance.pca_target = joblib.load(load_path / "pca_target.pkl")
        except ImportError:
            logger.warning("joblib not available, PCA models not loaded")
        
        # Restore state
        instance.is_fitted = metadata["is_fitted"]
        instance.metadata = metadata["metadata"]
        
        logger.info(f"Loaded Procrustes mapping from {load_path}")
        return instance 
    
    def check_fit(self, path) -> bool:
        """Check if the mapping is fitted."""
        load_path = Path(path)
        return (load_path/ "rotation_matrix.npy").exists() and (load_path/ "source_mean.npy").exists() and (load_path/ "target_mean.npy").exists() and (load_path/ "scale.npy").exists()