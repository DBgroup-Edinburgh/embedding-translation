import numpy as np
from typing import Optional, Dict, Any, Union
import json
from pathlib import Path
import hashlib
import joblib
from loguru import logger

from .base import ClusteringStrategy, ClusteringResult
from .strategies import KMeansClusteringStrategy, LA2MClusteringStrategy
from ..config import ClusteringConfig
from ..embedding import EmbeddingRequest, get_embedding
from ..reference import get_reference

method_map = {
    "kmeans": KMeansClusteringStrategy,
    "la2m-cluster": LA2MClusteringStrategy
}

SUPPORTED_CLUSTERING_METHODS = list(method_map.keys())

def config_hash_path(dataset_name: str, model: str, reference_key: str, strategy_name: str, strategy_config: ClusteringConfig) -> str:
    if reference_key is not None:
        hash_target_str = strategy_config.to_string() + reference_key
    else:
        hash_target_str = strategy_config.to_string()
    return hashlib.md5(hash_target_str.encode()).hexdigest()

def get_cluster_save_path(
    cluster_path: Union[str, Path],
    dataset_name: str,
    model: str,
    reference_key: str,
    strategy_name: str,
    strategy_config: ClusteringConfig
) -> Path:
    base_path = Path(cluster_path)
    cluster_key = config_hash_path(dataset_name, model, reference_key, strategy_name, strategy_config)
    return base_path / f"{strategy_name}_{dataset_name}_{model}_{cluster_key}"

class ClusterManager:
    def __init__(
        self,
        dataset_name: str,
        model: str,
        reference_key: str,
        reference_path: str,
        cluster_path: Union[str, Path],
        embedding_path: str,
        strategy_name: str,
        strategy_config: Optional[ClusteringConfig] = None,
        force: bool = False,
        auto_save_results: bool = False,
        verbose: bool = False
    ) -> None:
        self.dataset_name = dataset_name
        self.model = model
        self.reference_key = reference_key
        self.reference_path = reference_path
        self.embedding_path = embedding_path
        self.strategy_name = strategy_name
        self.strategy_config = strategy_config or ClusteringConfig()
        self.cluster_path = Path(cluster_path)
        self.final_save_path = get_cluster_save_path(
            self.cluster_path,
            self.dataset_name,
            self.model,
            self.reference_key,
            self.strategy_name,
            self.strategy_config
        )
        self.force = force
        self.auto_save_results = auto_save_results
        self.verbose = verbose
        self.strategy = self._create_strategy()
        self.last_result: Optional[ClusteringResult] = None

    def _create_strategy(self) -> ClusteringStrategy:
        if self.strategy_name not in SUPPORTED_CLUSTERING_METHODS:
            raise ValueError(f"Unsupported clustering method: {self.strategy_name}")
        return method_map[self.strategy_name](self.strategy_config)

    def _resolve_save_path(self, save_path: Optional[Union[str, Path]] = None) -> Path:
        path = Path(save_path) if save_path else self.final_save_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fit(self, embeddings: Optional[np.ndarray] = None, save_path: Optional[Union[str, Path]] = None, save_embeddings: bool = True, reference_indices: Optional[np.ndarray] = None) -> ClusteringResult:
        if not self.force:
            try:
                return self.load()
            except Exception as e:
                logger.error(f"Error trying to load clustering result: {e}")

        if embeddings is None:
            loaded_embeddings = get_embedding(
                EmbeddingRequest(
                    dataset_name=self.dataset_name,
                    model_name=self.model,
                    embedding_path=Path(self.embedding_path),
                )
            )
            if isinstance(loaded_embeddings, tuple):
                embeddings = loaded_embeddings[0]
            else:
                embeddings = loaded_embeddings

        if reference_indices is None:
            reference = get_reference(self.reference_path, self.reference_key)
            reference_indices = reference["d0_index"]

        result = self.strategy.fit(embeddings, reference_indices)
        self.last_result = result

        if self.auto_save_results:
            self.save(result, save_embeddings=save_embeddings, save_path=save_path)

        return result

    def save(self, clustering_result: Optional[ClusteringResult] = None, save_embeddings: bool = False, save_path: Optional[Union[str, Path]] = None) -> None:
        path = self._resolve_save_path(save_path)
        logger.info(f"Saving clustering result to {path}")
        result = clustering_result or self.last_result
        if result is None:
            raise ValueError("No result to save.")

        result.save(path, save_embeddings)
        with open(path / "config.json", "w") as f:
            json.dump(self._serialize_config(), f)
        logger.info(f"Saved clustering result to {path}")

    def load(self, save_path: Optional[Union[str, Path]] = None) -> ClusteringResult:
        path = self._resolve_save_path(save_path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        self.last_result = ClusteringResult.load(path)
        if self.last_result is None:
            raise ValueError(f"No clustering result found in {path}")
        logger.info(f"Loaded clustering result from {path}")
        return self.last_result

    def _serialize_config(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model": self.model,
            "reference_key": self.reference_key,
            "reference_path": self.reference_path,
            "cluster_path": str(self.cluster_path),
            "embedding_path": self.embedding_path,
            "strategy_name": self.strategy_name,
            "strategy_config": self.strategy_config.model_dump(),
            "force": self.force,
            "auto_save_results": self.auto_save_results,
            "verbose": self.verbose
        }
    
    def predict(self, clustering_result: ClusteringResult, embeddings: np.ndarray, target_indices: Optional[np.ndarray] = None) -> np.ndarray:
        if target_indices is not None:
            embeddings = embeddings[target_indices]

        return self.strategy.predict(clustering_result, embeddings)

    @classmethod
    def from_saved(cls, save_path: Union[str, Path]) -> 'ClusterManager':
        with open(Path(save_path) / "config.json", "r") as f:
            config = json.load(f)
        return cls(
            dataset_name=config["dataset_name"],
            model=config["model"],
            reference_key=config["reference_key"],
            reference_path=config["reference_path"],
            cluster_path=config["cluster_path"],
            embedding_path=config["embedding_path"],
            strategy_name=config["strategy_name"],
            strategy_config=ClusteringConfig.model_validate(config["strategy_config"]),
            force=config["force"],
            auto_save_results=config["auto_save_results"],
            verbose=config["verbose"]
        )