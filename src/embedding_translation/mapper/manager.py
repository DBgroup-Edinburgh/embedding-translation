"""
Main mapper class for VectorMerge.

This module provides the VectorSpaceMapper class, which serves as the main
interface for all embedding mapping strategies in VectorMerge.
"""

import numpy as np
from typing import Union, Dict, Any, Optional, List
import hashlib
import shutil
from pathlib import Path
import time
from loguru import logger

from ..core.mapping import MappingStrategy, MappingResult
from ..clustering import ClusterManager
from ..config import ClusteringConfig, MappingConfig
from .procrustes.strategy import ProcrustesMappingStrategy
from .nonlinear.strategy import NonLinearMappingStrategy
from .la2m.strategy import LA2MStrategy
from .cca.strategy import CCAMappingStrategy
from .gromov_wasserstein.strategy import GromovWassersteinMappingStrategy
from .linear import LinearMapper
from .simple_linear import SimpleLinearMapper
from .hmoe import HMoEMapper


def config_hash_path(dataset_name, source_model, target_model, strategy, config, reference_key):
    if reference_key is None:
        hash_target_str = config.to_string()
    else:
        hash_target_str = config.to_string() + reference_key
    config_hash = hashlib.md5(hash_target_str.encode()).hexdigest()
    return f"{strategy}_{dataset_name}_{source_model}_{target_model}_{config_hash}"

class VectorSpaceMapper:
    """
    Main interface for vector space mapping in VectorMerge.
    
    This class provides a unified interface for different mapping strategies
    and handles the complete mapping workflow from training to transformation.
    """
    
    # Available mapping strategies
    AVAILABLE_STRATEGIES = {
        "procrustes": ProcrustesMappingStrategy,
        "nonlinear": NonLinearMappingStrategy,
        "la2m": LA2MStrategy,
        "cca": CCAMappingStrategy,
        "gromov_wasserstein": GromovWassersteinMappingStrategy,
        "linear": LinearMapper,
        "simple_linear": SimpleLinearMapper,
        "hmoe": HMoEMapper,
    }
    
    def __init__(self, strategy_name: str, config: MappingConfig, dataset_name: str, source_model: str, target_model: str, reference_key: str, reference_path: str, cluster_path: str, embedding_path: str, mapping_param_path: Union[str, Path], mapping_embedding_path: Union[str, Path], force: bool = False, save_param: bool = False, save_embedding: bool = False, clustering_config: Optional[ClusteringConfig] = None):
        """Initialize the VectorSpaceMapper.
        
        Args:
            strategy: Name of the mapping strategy to use
            config: Configuration object (if None, uses default config)
            dataset_name: Name of the dataset
            source_model: Name of the source model
            target_model: Name of the target model
            mapping_param_path: Path to save mapping parameters
            mapping_embedding_path: Path to save mapping embeddings
            force: Force overwrite existing mappings
        """
        if strategy_name not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: "
                           f"{list(self.AVAILABLE_STRATEGIES.keys())}")
        
        self.strategy_name = strategy_name
        self.config = config if config is not None else MappingConfig()
        self.force = force
        self.dataset_name = dataset_name
        self.source_model = source_model
        self.target_model = target_model
        self.reference_key = reference_key
        self.reference_path = reference_path
        self.embedding_path = embedding_path
        self.cluster_path = cluster_path
        self.save_param = save_param
        self.save_embedding = save_embedding
        self.cluster_config = clustering_config

        self.hash_path = config_hash_path(dataset_name, source_model, target_model, strategy_name, config, reference_key)
        
        # Setup paths with strategy and config hash
        self.mapping_param_path, self.mapping_embedding_path = self._setup_paths(
            mapping_param_path, mapping_embedding_path, strategy_name, self.config, force,
        )
        
        # Initialize the mapping strategy
        strategy_class = self.AVAILABLE_STRATEGIES[strategy_name]
        if strategy_name == "la2m":
            if clustering_config is None:
                raise ValueError("clustering_config is required for la2m strategy")
            cluster_manager = self._init_cluster_manager(clustering_config)
            self.mapping_strategy = strategy_class(self.config, cluster_manager)
        else:
            self.mapping_strategy = strategy_class(self.config)
        
        # Training history and metadata
        self.training_history: Dict[str, Any] = {}
        self.is_fitted = False
        
        logger.info(f"VectorSpaceMapper initialized with strategy: {strategy_name}")
        logger.info(f"Parameter path: {self.mapping_param_path}")
        logger.info(f"Embedding path: {self.mapping_embedding_path}")
    
    def _init_cluster_manager(self, clustering_config: ClusteringConfig):
        return ClusterManager(
            dataset_name=self.dataset_name,
            model=self.source_model,
            reference_key=self.reference_key,
            reference_path=self.reference_path,
            cluster_path=self.cluster_path,
            embedding_path=self.embedding_path,
            strategy_name=self.config.la2m_config.cluster_method,
            strategy_config=self.cluster_config,
            auto_save_results=True,
            verbose=self.config.verbose
        )
    
    def _setup_paths(self, mapping_param_path: Union[str, Path], mapping_embedding_path: Union[str, Path], 
                     strategy: str, config: MappingConfig, force: bool) -> tuple[Path, Path]:
        """Setup and create mapping paths with strategy and config hash.
        
        Args:
            mapping_param_path: Base parameter path
            mapping_embedding_path: Base embedding path
            strategy: Mapping strategy name
            config: Configuration object
            force: Whether to force overwrite existing directories
            
        Returns:
            Tuple of (final_param_path, final_embedding_path)
        """
        # Expand paths with strategy name
        param_path = Path(mapping_param_path)
        embedding_path = Path(mapping_embedding_path)
        
        # Create base strategy directories
        param_path.mkdir(parents=True, exist_ok=True)
        embedding_path.mkdir(parents=True, exist_ok=True)
        
        # Add config hash for unique configurations
        final_param_path = param_path / self.hash_path
        final_embedding_path = embedding_path / self.hash_path
        
        return final_param_path, final_embedding_path
    
    def check_on_disk(self, path: Union[str, Path]) -> bool:
        """Check if the mapping is fitted and on disk."""
        return Path(path).exists() and self.mapping_strategy.check_on_disk(path)
    
    def fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
            reference_indices: np.ndarray, **kwargs) -> 'VectorSpaceMapper':
        """Fit the mapping strategy on reference data.
        
        Args:
            source_embeddings: Source embedding space (N x D1)
            target_embeddings: Target embedding space (N x D2)  
            reference_indices: Indices of reference points for training
            **kwargs: Additional arguments for the mapping strategy
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.strategy_name} mapping strategy...")
        
        # Validate inputs
        self._validate_inputs(source_embeddings, target_embeddings, reference_indices)
        
        # Record training start time
        start_time = time.time()

        if not self.force:
            if self.check_on_disk(self.mapping_param_path):
                logger.info(f"Loading existing mapping parameters from {self.mapping_param_path}")
                try:
                    self.mapping_strategy = self.mapping_strategy.load(self.mapping_param_path)
                    self.is_fitted = True
                    logger.info(f"Loaded existing mapping parameters from {self.mapping_param_path}")
                    return self
                except Exception as e:
                    logger.info(f"Fitting {self.strategy_name} mapping strategy from scratch...")

        # Fit the mapping strategy
        self.mapping_strategy.fit(
            source_embeddings, target_embeddings, reference_indices, **kwargs
        )
        
        # Record training time
        training_time = time.time() - start_time
        
        # Update training history
        self.training_history = {
            'strategy': self.strategy_name,
            'training_time': training_time,
            'reference_size': len(reference_indices),
            'source_shape': source_embeddings.shape,
            'target_shape': target_embeddings.shape,
            'config': self.config.model_dump(),
            'strategy_metadata': self.mapping_strategy.metadata.copy()
        }
        
        if self.save_param:
            self.mapping_strategy.save(self.mapping_param_path)
        
        logger.info(f"Mapping strategy fitted successfully in {training_time:.2f} seconds")
        return self
    
    
    def transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using the fitted mapping strategy.
        
        Args:
            embeddings: Embeddings to transform (N x D1)
            **kwargs: Additional arguments for the mapping strategy
            
        Returns:
            Transformed embeddings (N x D2)
        """
        if not self.mapping_strategy.is_fitted:
            raise ValueError("Mapper must be fitted before transformation")
        
        if not self.force:
            if self.check_embedding_on_disk():
                logger.info(f"Loading existing mapping embeddings from {self.mapping_embedding_path}")
                return self.load_embedding_from_disk()
        
        logger.info(f"Transforming {len(embeddings)} embeddings...")
        
        start_time = time.time()
        transformed = self.mapping_strategy.transform(embeddings, **kwargs)
        transform_time = time.time() - start_time
        
        logger.info(f"Transformation completed in {transform_time:.2f} seconds")

        if self.save_embedding:
            self.save_embedding_to_disk(transformed)
        
        return transformed
    
    def fit_and_transform(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                     reference_indices: np.ndarray, embeddings_to_transform: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        """Fit the mapping and transform embeddings in one step.
        
        Args:
            source_embeddings: Source embedding space
            target_embeddings: Target embedding space
            reference_indices: Reference indices for training
            embeddings_to_transform: Embeddings to transform (if None, uses source_embeddings)
            **kwargs: Additional arguments
            
        Returns:
            Transformed embeddings
        """
        # Fit the mapping
        self.fit(source_embeddings, target_embeddings, reference_indices, **kwargs)
        
        # Transform embeddings
        if embeddings_to_transform is None:
            embeddings_to_transform = source_embeddings
        
        return self.transform(embeddings_to_transform, **kwargs)
    
    def save_embedding_to_disk(self, embeddings: np.ndarray) -> None:
        """Save the transformed embeddings to disk."""
        mapping_embedding_path = self.mapping_embedding_path
        if not str(mapping_embedding_path).endswith(".npy"):
            mapping_embedding_path = mapping_embedding_path.with_suffix(".npy")
        
        logger.info(f"Saving transformed embeddings to {mapping_embedding_path}")
        np.save(mapping_embedding_path, embeddings)
    
    def load_embedding_from_disk(self) -> np.ndarray:
        """Load the transformed embeddings from disk."""
        mapping_embedding_path = self.mapping_embedding_path
        if not str(mapping_embedding_path).endswith(".npy"):
            mapping_embedding_path = mapping_embedding_path.with_suffix(".npy")
        
        logger.info(f"Loading transformed embeddings from {mapping_embedding_path}")
        return np.load(mapping_embedding_path)
    
    def check_embedding_on_disk(self) -> bool:
        """Check if the transformed embeddings exist."""
        mapping_embedding_path = self.mapping_embedding_path
        if not str(mapping_embedding_path).endswith(".npy"):
            mapping_embedding_path = mapping_embedding_path.with_suffix(".npy")
        
        return mapping_embedding_path.exists()
        
    
    def get_mapping_result(self, transformed_embeddings: np.ndarray) -> MappingResult:
        """Create a MappingResult object with metadata.
        
        Args:
            transformed_embeddings: The transformed embeddings
            
        Returns:
            MappingResult object
        """
        if not self.is_fitted:
            raise ValueError("Mapper must be fitted before creating result")
        
        metadata = self.training_history.copy()
        metadata['transform_shape'] = transformed_embeddings.shape
        
        return MappingResult(
            transformed_embeddings=transformed_embeddings,
            mapping_strategy=self.strategy_name,
            metadata=metadata
        )
    
    def evaluate_mapping(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                        test_indices: np.ndarray) -> Dict[str, float]:
        """Evaluate the mapping quality on test data.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings  
            test_indices: Indices of test points
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Mapper must be fitted before evaluation")
        
        logger.info(f"Evaluating mapping on {len(test_indices)} test points...")
        
        # Transform test source embeddings
        test_source = source_embeddings[test_indices]
        test_target = target_embeddings[test_indices]
        transformed_test = self.transform(test_source)
        
        # Compute evaluation metrics
        metrics = {}
        
        # Mean Squared Error
        mse = np.mean((transformed_test - test_target) ** 2)
        metrics['mse'] = float(mse)
        
        # Root Mean Squared Error
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Mean Absolute Error
        mae = np.mean(np.abs(transformed_test - test_target))
        metrics['mae'] = float(mae)
        
        # Cosine Similarity
        def cosine_similarity(a, b):
            a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
            return np.mean(np.sum(a_norm * b_norm, axis=1))
        
        cos_sim = cosine_similarity(transformed_test, test_target)
        metrics['cosine_similarity'] = float(cos_sim)
        
        # Pearson Correlation (if embeddings are 1D or we take mean)
        try:
            if transformed_test.shape[1] == 1:
                corr = np.corrcoef(transformed_test.flatten(), test_target.flatten())[0, 1]
            else:
                # Average correlation across dimensions
                corrs = []
                for i in range(transformed_test.shape[1]):
                    corr = np.corrcoef(transformed_test[:, i], test_target[:, i])[0, 1]
                    if not np.isnan(corr):
                        corrs.append(corr)
                corr = np.mean(corrs) if corrs else 0.0
            
            metrics['correlation'] = float(corr) if not np.isnan(corr) else 0.0
        except:
            metrics['correlation'] = 0.0
        
        logger.info(f"Evaluation completed. MSE: {metrics['mse']:.6f}, "
                   f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        
        return metrics
    
    def save(self, path: Union[str, Path, None] = None) -> None:
        """Save the fitted mapper to disk.
        
        Args:
            path: Path to save the mapper
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted mapper")

        if path is None:
            path = self.mapping_param_path
        
        if path is None:
            raise ValueError("Path is required")

        if isinstance(path, str):
            path = Path(path)
        
        if self.force:
            if path.exists():
                shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save mapper metadata
        mapper_info = {
            'strategy_name': self.strategy_name,
            'dataset_name': self.dataset_name,
            'source_model': self.source_model,
            'target_model': self.target_model,
            'reference_key': self.reference_key,
            'reference_path': self.reference_path,
            'cluster_path': self.cluster_path,
            'embedding_path': self.embedding_path,
            'config': self.config.model_dump(),
            'training_history': self.training_history,
            'is_fitted': self.is_fitted,
            'cluster_config': self.cluster_config.model_dump() if self.cluster_config is not None else None,
            'mapping_param_path': str(self.mapping_param_path),
            'mapping_embedding_path': str(self.mapping_embedding_path),
        }
        
        import json
        with open(path / "mapper_info.json", "w") as f:
            json.dump(mapper_info, f, indent=2)
        
        self.mapping_strategy.save(path / "strategy")
        
        logger.info(f"Saved VectorSpaceMapper to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], mapping_param_path: Union[str, Path] = "./output/mapping_models/", 
             mapping_embedding_path: Union[str, Path] = "./output/mapping_embeddings/",
             reference_path: str = "", cluster_path: str = "", embedding_path: str = "") -> 'VectorSpaceMapper':
        """Load a fitted mapper from disk.
        
        Args:
            path: Path to load the mapper from
            mapping_param_path: Path for mapping parameters (for future use)
            mapping_embedding_path: Path for mapping embeddings (for future use)
            
        Returns:
            Loaded VectorSpaceMapper instance
        """
        load_path = Path(path)
        
        # Load mapper metadata
        import json
        with open(load_path / "mapper_info.json", "r") as f:
            mapper_info = json.load(f)
        
        # Create mapper instance
        config = MappingConfig.model_validate(mapper_info['config'])
        mapper = cls(
            strategy_name=mapper_info['strategy_name'], 
            config=config,
            dataset_name=mapper_info['dataset_name'],
            source_model=mapper_info['source_model'],
            target_model=mapper_info['target_model'],
            reference_key=mapper_info['reference_key'],
            reference_path=mapper_info['reference_path'],
            cluster_path=mapper_info['cluster_path'],
            embedding_path=mapper_info['embedding_path'],
            mapping_param_path=mapper_info['mapping_param_path'],
            mapping_embedding_path=mapper_info['mapping_embedding_path'],
            clustering_config=ClusteringConfig.model_validate(mapper_info['cluster_config']) if mapper_info['cluster_config'] is not None else None
        )
        
        # Load the mapping strategy
        mapper.mapping_strategy = mapper.mapping_strategy.load(load_path / "strategy")
        
        # Restore state
        mapper.training_history = mapper_info['training_history']
        mapper.is_fitted = mapper_info['is_fitted']
        
        logger.info(f"Loaded VectorSpaceMapper from {load_path}")
        return mapper
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current mapping strategy.
        
        Returns:
            Dictionary with strategy information
        """
        info = {
            'strategy_name': self.strategy_name,
            'is_fitted': self.is_fitted,
            'config': self.config.model_dump()
        }
        
        if self.is_fitted:
            info['training_history'] = self.training_history
            info['strategy_metadata'] = self.mapping_strategy.metadata
        
        # Add strategy-specific information
        cluster_stats = getattr(self.mapping_strategy, 'get_cluster_statistics', None)
        if cluster_stats is not None:
            info['cluster_statistics'] = cluster_stats()
        
        training_history = getattr(self.mapping_strategy, 'get_training_history', None)
        if training_history is not None:
            info['training_losses'] = training_history()
        
        return info
    
    def _validate_inputs(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                        reference_indices: np.ndarray) -> None:
        """Validate input arrays for fitting.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings
            reference_indices: Reference indices
        """
        # Check array types
        if not isinstance(source_embeddings, np.ndarray):
            raise TypeError("source_embeddings must be a numpy array")
        if not isinstance(target_embeddings, np.ndarray):
            raise TypeError("target_embeddings must be a numpy array")
        if not isinstance(reference_indices, np.ndarray):
            raise TypeError("reference_indices must be a numpy array")
        
        # Check dimensions
        if len(source_embeddings.shape) != 2:
            raise ValueError("source_embeddings must be 2D array")
        if len(target_embeddings.shape) != 2:
            raise ValueError("target_embeddings must be 2D array")
        
        # Check matching number of samples
        if source_embeddings.shape[0] != target_embeddings.shape[0]:
            raise ValueError("source_embeddings and target_embeddings must have same number of samples")
        
        # Check reference indices
        if len(reference_indices) == 0:
            raise ValueError("reference_indices cannot be empty")
        
        if np.any(reference_indices < 0) or np.any(reference_indices >= source_embeddings.shape[0]):
            raise ValueError("reference_indices contain invalid indices")
        
        # Check for sufficient reference points
        min_points = 2 if self.strategy_name != "la2m" else 4  # LA2M strategy needs more points for clustering
        if len(reference_indices) < min_points:
            raise ValueError(f"At least {min_points} reference points required for {self.strategy_name} strategy")
    
    @classmethod
    def list_available_strategies(cls) -> List[str]:
        """List all available mapping strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls.AVAILABLE_STRATEGIES.keys())
    
    @classmethod
    def get_strategy_description(cls, strategy: str) -> str:
        """Get description of a mapping strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Strategy description
        """
        descriptions = {
            "procrustes": "Procrustes analysis - orthogonal transformation using SVD",
            "nonlinear": "Non-linear neural network - deep network with non-linear activations",
            "la2m": "Clustering-based mapping - learns local mappings for each cluster",
            "cca": "Canonical Correlation Analysis - finds linear transformations that maximize correlation",
            "gromov_wasserstein": "Gromov-Wasserstein alignment - compares internal structure of embedding spaces"
        }
        
        return descriptions.get(strategy, "Unknown strategy")
    
    def __repr__(self) -> str:
        """String representation of the mapper."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"VectorSpaceMapper(strategy='{self.strategy_name}', status='{status}')" 