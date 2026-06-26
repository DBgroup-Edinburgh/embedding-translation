import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import faiss


def get_retrieval_list(query_emb_2: np.ndarray, embeddings: np.ndarray, top_k: int, metric: str = "l2") -> np.ndarray:
    """
    Get the retrieval list using FAISS for efficient similarity search.
    
    Args:
        query_emb_2: Query embeddings with shape (num_queries, embedding_dim)
        embeddings: Corpus embeddings with shape (num_docs, embedding_dim)
        top_k: Number of top documents to retrieve
        metric: Distance metric to use ("l2" or "cosine")
        
    Returns:
        np.ndarray: Array of retrieved document indices with shape (num_queries, top_k)
    """
    dim = embeddings.shape[1]
    
    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32)) # type: ignore
        distances, indices = index.search(query_emb_2.astype(np.float32), top_k) # type: ignore
        return indices
    elif metric == "cosine":
        # For cosine similarity, normalize embeddings manually
        embeddings_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = (embeddings / (embeddings_norms + 1e-8)).astype(np.float32)
        
        query_norms = np.linalg.norm(query_emb_2, axis=1, keepdims=True)  
        query_normalized = (query_emb_2 / (query_norms + 1e-8)).astype(np.float32)
        
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_normalized) # type: ignore
        distances, indices = index.search(query_normalized, top_k) # type: ignore
        return indices
    else:
        raise ValueError(f"Unsupported metric: {metric}")





class Evaluator:
    def __init__(self, corpus_emb_1: np.ndarray, corpus_emb_2: np.ndarray, query_emb_1: np.ndarray, query_emb_2: np.ndarray, query_index2answer_index: Dict[int, List[int]], d0: np.ndarray, d1: np.ndarray, d2: np.ndarray, corpus_emb_1_transformed: np.ndarray, k_list: List[int] = [10, 50, 100, 500, 1000]):
        self.corpus_emb_1 = corpus_emb_1
        self.corpus_emb_2 = corpus_emb_2
        self.query_emb_1 = query_emb_1
        self.query_emb_2 = query_emb_2
        self.query_index2answer_index = query_index2answer_index
        for key, value in query_index2answer_index.items():
                self.query_index2answer_index[key] = [value[0]] # only use the first answer
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.corpus_emb_1_transformed = corpus_emb_1_transformed
        self.k_list = k_list
        self._align_dimensions()
    
    @staticmethod
    def get_rank_of_answer(query_emb: np.ndarray, answer_indices: List[int], 
                          corpus_embeddings: np.ndarray, metric: str = "l2") -> int:
        """
        Get the rank of the first answer document for a given query.
        
        Args:
            query_emb: Query embedding (1, embedding_dim)
            answer_indices: List of answer document indices
            corpus_embeddings: Corpus embeddings (num_docs, embedding_dim)
            metric: Distance metric to use ("l2" or "cosine")
            
        Returns:
            int: Rank of the first answer document (1-indexed)
        """
        if not answer_indices:
            return len(corpus_embeddings) + 1  # Return a large rank if no answers
        
        # Compute distances/similarities between query and all documents
        if metric == "l2":
            distances = np.linalg.norm(corpus_embeddings - query_emb, axis=1)
            # Sort by distance (ascending)
            sorted_indices = np.argsort(distances)
        elif metric == "cosine":
            # Normalize embeddings for cosine similarity
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            corpus_normalized = corpus_embeddings / (corpus_norms + 1e-8)
            
            similarities = np.dot(corpus_normalized, query_norm.T).flatten()
            # Sort by similarity (descending)
            sorted_indices = np.argsort(-similarities)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Find the rank of the first answer document
        for rank, doc_idx in enumerate(sorted_indices, 1):
            if doc_idx in answer_indices:
                return rank
        
        return len(corpus_embeddings) + 1  # Return a large rank if no answers found
    
    def _align_dimensions(self):
        max_dim = max(self.corpus_emb_1.shape[1], self.corpus_emb_2.shape[1], self.corpus_emb_1_transformed.shape[1], self.query_emb_2.shape[1])
        self.corpus_emb_1 = np.pad(self.corpus_emb_1, ((0, 0), (0, max_dim - self.corpus_emb_1.shape[1])), mode="constant", constant_values=0)
        self.corpus_emb_2 = np.pad(self.corpus_emb_2, ((0, 0), (0, max_dim - self.corpus_emb_2.shape[1])), mode="constant", constant_values=0)
        self.corpus_emb_1_transformed = np.pad(self.corpus_emb_1_transformed, ((0, 0), (0, max_dim - self.corpus_emb_1_transformed.shape[1])), mode="constant", constant_values=0)
        self.query_emb_2 = np.pad(self.query_emb_2, ((0, 0), (0, max_dim - self.query_emb_2.shape[1])), mode="constant", constant_values=0)

    def evaluate(self) -> Dict:
        """Main evaluation method that computes all metrics."""
        results = {}
        
        # Get basic recall metrics
        recall_results = self.merge_and_evaluate_embeddings(
            self.corpus_emb_1, self.corpus_emb_2, self.corpus_emb_1_transformed, 
            self.query_emb_2, self.query_index2answer_index, self.d0, self.d1, self.d2
        )
        results.update(recall_results)
        
        # Get standard Recall@k metrics
        standard_recall_results = self._evaluate_recall_at_k()
        results.update(standard_recall_results)
        
        # Get additional evaluation metrics
        d1_recall_results = self._evaluate_retrieval_D1()
        results.update(d1_recall_results)
        
        relative_ndcg_results = self._evaluate_relative_ndcg()
        results.update(relative_ndcg_results)
        
        beta_results = self._evaluate_beta()
        results.update(beta_results)
        
        return results

    def _evaluate_retrieval_D1(self) -> Dict[str, float]:
        """
        Evaluate Recall@k[D1] - recall considering only queries where the top answer is in D1.
        
        Returns:
            Dict containing Recall@k[D1] scores for different k values
        """
        logger.info("Computing Recall@k[D1] metrics...")
        
        # Create integrated database using transformed embeddings
        integrated_db = self.corpus_emb_2.copy()
        integrated_db[self.d1] = self.corpus_emb_1_transformed[self.d1]
        
        # Filter queries to only include those where the top answer is in D1
        d1_queries = {}
        for query_idx, answer_indices in self.query_index2answer_index.items():
            # Check if any answer is in D1
            if any(ans_idx in self.d1 for ans_idx in answer_indices):
                d1_queries[query_idx] = answer_indices
        
        if not d1_queries:
            logger.warning("No queries found with answers in D1")
            return {f"recall_d1@{k}": 0.0 for k in self.k_list}
        
        results = {}
        for k in self.k_list:
            hit_count = 0
            for query_idx, answer_indices in d1_queries.items():
                query_emb = self.query_emb_2[query_idx:query_idx+1]
                rank = self.get_rank_of_answer(query_emb, answer_indices, integrated_db)
                if rank <= k:
                    hit_count += 1
            
            recall_d1 = hit_count / len(d1_queries)
            results[f"recall_d1@{k}"] = recall_d1
        
        return results

    def _evaluate_relative_ndcg(self) -> Dict[str, float]:
        """
        Evaluate Relative NDCG - measures relative rank changes compared to ground truth.
        
        Returns:
            Dict containing relative NDCG scores
        """
        logger.info("Computing Relative NDCG metrics...")
        
        # Create integrated database using transformed embeddings
        integrated_db = self.corpus_emb_2.copy()
        integrated_db[self.d1] = self.corpus_emb_1_transformed[self.d1]
        
        # Create ground truth database (D* = emb2(O1 ∪ O∩ ∪ O2))
        # This includes all documents embedded with emb2
        ground_truth_db = self.corpus_emb_2.copy()
        
        scores = []
        for query_idx, answer_indices in self.query_index2answer_index.items():
            if not answer_indices:
                continue
                
            query_emb = self.query_emb_2[query_idx:query_idx+1]
            
            # Get rank in integrated database
            r_merge = self.get_rank_of_answer(query_emb, answer_indices, integrated_db)
            
            # Get rank in ground truth database
            r_star = self.get_rank_of_answer(query_emb, answer_indices, ground_truth_db)
            
            # Compute relative NDCG score: 1/log2(|r_merge - r_star| + 1)
            if r_merge == r_star:
                score = 1.0  # Perfect alignment
            else:
                score = 1.0 / np.log2(abs(r_merge - r_star) + 1)
            
            scores.append(score)
        
        if not scores:
            logger.warning("No valid queries for relative NDCG computation")
            return {"relative_ndcg": 0.0}
        
        relative_ndcg = np.mean(scores)
        return {"relative_ndcg": float(relative_ndcg)}

    def _evaluate_beta(self) -> Dict[str, float]:
        """
        Evaluate integration error β - measures distance between transformed and target embeddings.
        
        Returns:
            Dict containing beta (integration error) scores
        """
        logger.info("Computing integration error β...")
        
        # For each o ∈ O1, compute distance ||A(emb1(o)) - emb2(o)||
        # where A(emb1(o)) is the transformed representation in D2 space
        
        # Get embeddings for D1 documents
        d1_emb_1_transformed = self.corpus_emb_1_transformed[self.d1]  # A(emb1(o))
        d1_emb_2 = self.corpus_emb_2[self.d1]  # emb2(o)
        
        # Compute L2 distances
        distances = np.linalg.norm(d1_emb_1_transformed - d1_emb_2, axis=1)
        
        # Compute average β
        beta = np.mean(distances)
        beta_std = np.std(distances)
        
        return {
            "integration_error_beta": float(beta),
            "integration_error_beta_std": float(beta_std),
            "integration_error_beta_min": float(np.min(distances)),
            "integration_error_beta_max": float(np.max(distances))
        }

    def _evaluate_recall_at_k(self) -> Dict[str, float]:
        """
        Evaluate standard Recall@k - measures if the top answer is retrieved within top-k.
        
        Formula: (1/|Q|) * Σ_{q∈Q} I(rank(emb2(q), A(D1, D2)) ≤ k)
        where I(p) is the indicator function (1 if p is true, 0 otherwise)
        
        Returns:
            Dict containing Recall@k scores for different k values
        """
        logger.info("Computing standard Recall@k metrics...")
        
        # Create integrated database using transformed embeddings
        integrated_db = self.corpus_emb_2.copy()
        integrated_db[self.d1] = self.corpus_emb_1_transformed[self.d1]
        
        results = {}
        total_queries = len(self.query_index2answer_index)
        
        for k in self.k_list:
            hit_count = 0
            for query_idx, answer_indices in self.query_index2answer_index.items():
                if not answer_indices:
                    continue
                    
                query_emb = self.query_emb_2[query_idx:query_idx+1]
                rank = self.get_rank_of_answer(query_emb, answer_indices, integrated_db)
                
                # Indicator function: 1 if rank ≤ k, 0 otherwise
                if rank <= k:
                    hit_count += 1
            
            recall_at_k = hit_count / total_queries
            results[f"recall_at_{k}"] = recall_at_k
        
        return results
    
    def get_retrieval_list(self, query_emb_2: np.ndarray, embeddings: np.ndarray, top_k: int, metric: str = "l2") -> np.ndarray:
        return get_retrieval_list(query_emb_2, embeddings, top_k=top_k, metric=metric)

    def merge_and_evaluate_embeddings(
        self,
        corpus_emb_1: np.ndarray,
        corpus_emb_2: np.ndarray,
        corpus_emb_1_transformed: np.ndarray,
        query_emb_2: np.ndarray,
        query_index2answer_index: Dict[int, List[int]],
        d0: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
    ) -> Dict:
        """
        Merge embedding spaces and evaluate retrieval performance.

        Args:
            corpus_emb_1: Embeddings from first model (source space)
            corpus_emb_2: Embeddings from second model (target space)
            corpus_emb_1_transformed: Transformed embeddings from first model
            query_emb_2: Query embeddings in target space
            query_index2answer_index: Dict mapping query index to list of answer indices
            d0: Reference set indices
            d1: Non-reference set indices for first model
            d2: Non-reference set indices for second model

        Returns:
            Dictionary containing recall metrics for different methods
        """
        non_ref_indices = d1

        merged_embeddings = {
            "target_only": corpus_emb_2.copy(),  # Only use target space embeddings
            "direct_concat": corpus_emb_2.copy(),  # Directly concatenate source and target embeddings
            "our_method": corpus_emb_2.copy()  # Our translation method
        }
        
        # For direct concatenation, use original source embeddings for non-reference indices
        merged_embeddings["direct_concat"][non_ref_indices] = corpus_emb_1[non_ref_indices]
        
        # For our method, use transformed source embeddings for non-reference indices
        merged_embeddings["our_method"][non_ref_indices] = corpus_emb_1_transformed[non_ref_indices]

        recalls = {}
        for top_k in [10, 50, 100, 500, 1000]:
            retrieval_results = {
                method: get_retrieval_list(query_emb_2, embeddings, top_k=top_k)
                for method, embeddings in merged_embeddings.items()
            }

            D1_hit_counts = {method: 0 for method in merged_embeddings.keys()}
            D2_hit_counts = {method: 0 for method in merged_embeddings.keys()}
            hit_counts = {method: 0 for method in merged_embeddings.keys()}
            
            # Iterate through each query
            for query_idx, answer_indices in query_index2answer_index.items():
                answer_indices = [answer_indices[0]] # only use the first answer
                for method, results in retrieval_results.items():
                    # Check if any answer is in the retrieved results for this query
                    retrieved_for_query = results[query_idx]
                    for answer_idx in answer_indices:
                        if answer_idx in retrieved_for_query:
                            hit_counts[method] += 1
                            if answer_idx in d1:
                                D1_hit_counts[method] += 1
                            if answer_idx in d2:
                                D2_hit_counts[method] += 1
                            break  # Count only one hit per query
                            
            total_queries = len(query_index2answer_index)
            recalls.update({method+f"@{top_k}": count / total_queries for method, count in hit_counts.items()})

        return recalls
