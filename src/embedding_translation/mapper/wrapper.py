from .la2m.strategy import LA2MStrategy
from .procrustes.strategy import ProcrustesMappingStrategy
from ..clustering import ClusterManager
from ..config import (
    ClusteringConfig,
    KMeansConfig,
    LA2MClusteringConfig,
    LA2MConfig,
    MappingConfig,
    ProcrustesConfig,
)

class LA2MMapper(LA2MStrategy):
    def __init__(self, 
            cluster_method: str="la2m-cluster", d_prime: int=10, pca_mapping: bool=True, pca_dim: int=14, use_norm: bool=False, device: str="auto", batch_size: int=32, verbose: bool=False, save_param: bool=True, save_embedding: bool=False, dataset_name: str="", model: str="", reference_key: str="", reference_path: str="./data/processed/references", cluster_path:str ="./data/processed/clusters", embedding_path: str="./data/processed/embeddings", force: bool=True, auto_save_results: bool=False, n_clusters: int=10, max_iter: int=300, tol: float=1e-4, random_state: int=42):
        """
        Args:
            cluster_method: str,
            d_prime: int,
            pca_mapping: bool,
            pca_dim: int,
            use_norm: bool,
            device: str,
            batch_size: int,
            verbose: bool,
            save_param: bool,
            save_embedding: bool,
            dataset_name: str,
            model: str,
            reference_key: str,
            reference_path: str,
            cluster_path: str,
            embedding_path: str,
            force: bool,
            auto_save_results: bool,
            n_clusters: int=10,
            max_iter: int=300,
            tol: float=1e-4,
            random_state: int=42,
        """

        config = MappingConfig(
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            save_param=save_param,
            save_embedding=save_embedding,
            la2m_config=LA2MConfig(
                cluster_method=cluster_method,
                d_prime=d_prime,
                pca_mapping=pca_mapping,
                pca_dim=pca_dim,
                use_norm=use_norm,
            )
        )


        clustering_manager = ClusterManager(
            dataset_name=dataset_name,
            model=model,
            reference_key=reference_key,
            reference_path=reference_path,
            cluster_path=cluster_path,
            embedding_path=embedding_path,
            strategy_name=cluster_method,
            strategy_config=ClusteringConfig(
                clustering_method=cluster_method,
                la2m_config=LA2MClusteringConfig(d_prime=d_prime),
                kmeans_config=KMeansConfig(n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state),
            ),
            force=force,
            auto_save_results=auto_save_results,
            verbose=verbose,
        )
        super().__init__(config=config, clustering_manager=clustering_manager)


class ProcrustesMapper(ProcrustesMappingStrategy):
    def __init__(self, 
            approximate: bool = False,
            q: int = 1500,
            with_rotation: bool = True,
            with_scaling: bool = True,
            use_pca: bool = False,
            reduced_dim: int = 0,
            procrustes_pca_type: str = "none",
            use_norm: bool = True,
            device: str = "auto",
            batch_size: int = 32,
            verbose: bool = False,
            save_param: bool = False, 
            save_embedding: bool = False):
        """
        Procrustes mapping with configurable parameters.
        
        Args:
            approximate: bool, whether to use approximated SVD
            q: int, number of components for approximation
            with_rotation: bool, whether to include rotation in the transformation
            with_scaling: bool, whether to include scaling/normalization
            use_pca: bool, whether to use PCA for dimensionality reduction
            reduced_dim: int, target dimensionality for PCA (if use_pca=True)
            procrustes_pca_type: str, type of PCA to use ("none", "inner", "outer")
            use_norm: bool, whether to use normalization
            device: str, device to use ("auto", "cpu", "cuda")
            batch_size: int, batch size for processing
            verbose: bool, whether to print verbose logs
            save_param: bool, whether to save parameters
            save_embedding: bool, whether to save embeddings
        """
        
        config = MappingConfig(
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            save_param=save_param,
            save_embedding=save_embedding,
            procrustes_config=ProcrustesConfig(
                approximate=approximate,
                q=q,
                with_rotation=with_rotation,
                with_scaling=with_scaling,
                use_pca=use_pca,
                reduced_dim=reduced_dim,
                procrustes_pca_type=procrustes_pca_type,
                use_norm=use_norm,
            )
        )
        
        super().__init__(config=config)