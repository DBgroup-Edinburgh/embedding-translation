"""
We implement the split method used in the LA2M paper.

The intuition is to considering the distribution of the answers (to the quires), the answer cannot be in the D0(reference), and should be distributed evenly in D1 and D2.

The steps are:
1. Get the distribution of the answers (to the quires)
2. Split the dataset into three parts:
    - D0: reference (no answers)
    - D1: unique for Embedding Model 1 (answers evenly distributed)
    - D2: unique for Embedding Model 2 (answers evenly distributed)

**Notice**: in the paper, we only consider the top 1 relevant answer to make the most balance split.
"""
from .base_split import BaseSplit
import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, Set, List
from pathlib import Path
from loguru import logger

if TYPE_CHECKING:
    from ..dataset import Dataset

class LA2MSplit(BaseSplit):
    def __init__(self, dataset_name: str, dataset_index: List[str], internal_index: np.ndarray, answer_index: Optional[np.ndarray], reference_ratio: float, reference_path: str, remove_dup_answer: bool = True, qrels: Optional[Dict] = None, select_top_1: bool = True, dataset_obj: "Optional[Dataset]" = None):
        super().__init__(dataset_name, internal_index, reference_ratio, reference_path)
        if answer_index is None and qrels is not None and dataset_obj is not None and hasattr(dataset_obj, "batch_original_ids_to_internal_indices"):
            logger.info(f"Extracting answer indices from qrels, select_top_1={select_top_1}")
            original_answer_index = self._extract_answer_indices_from_qrels(qrels, dataset_index, select_top_1)
            answer_index = dataset_obj.batch_original_ids_to_internal_indices(original_answer_index)
        elif answer_index is None:
            raise ValueError("Either answer_index or qrels must be provided")
        
        self.answer_index = answer_index
        self.dataset_obj = dataset_obj
        self.dataset_index = dataset_index
        self.remove_dup_answer = remove_dup_answer
        self.reference_key = Path(f"la2m_split_{dataset_name}_{reference_ratio:.2f}_remove_dup_answer_{remove_dup_answer}_select_top_1_{select_top_1}")
        logger.info(f"LA2MSplit initialized with reference key: {self.reference_key}")

    def _split(self, internal_index: Optional[np.ndarray] = None, answer_index: Optional[np.ndarray] = None, reference_ratio: Optional[float] = None, save: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into three parts using LA2M strategy:
        - D0: reference (no answers)
        - D1: unique for Embedding Model 1 (answers evenly distributed)
        - D2: unique for Embedding Model 2 (answers evenly distributed)
        
        Args:
            dataset_index: indices of all documents in the dataset
            answer_index: indices of documents that are answers to queries
            reference_ratio: ratio of documents to use as reference
            save: whether to save the split results
            
        Returns:
            tuple of (d0_index, d1_index, d2_index)
        """
        if internal_index is None:
            internal_index = self.internal_index
        if reference_ratio is None:
            reference_ratio = self.reference_ratio
        if answer_index is None:
            answer_index = self.answer_index
        
        if answer_index is None:
            raise ValueError("Answer index is required for LA2M split")
        
        logger.info("Starting LA2M split process")
        logger.info(f"Dataset size: {len(internal_index)}, Answer docs: {len(answer_index)}, Reference ratio: {reference_ratio}")
        
        # Step 1: Handle duplicate answers if needed
        unique_answer_index = self._process_answer_duplicates(answer_index)
        
        # Step 2: Categorize documents into answer and non-answer
        answer_docs, non_answer_docs = self._categorize_documents(internal_index, unique_answer_index)
        
        # Step 3: Allocate D0 (reference) from non-answer documents only
        d0_docs = self._allocate_reference_set(non_answer_docs, reference_ratio, len(internal_index))
        
        # Step 4: Evenly distribute remaining documents between D1 and D2
        d1_docs, d2_docs = self._distribute_remaining_documents(answer_docs, non_answer_docs - d0_docs)
        
        # Step 5: Convert back to original indices
        d0_index, d1_index, d2_index = self._convert_to_indices(internal_index, d0_docs, d1_docs, d2_docs)
        
        # Step 6: Validate LA2M constraints
        self._validate_la2m_split(d0_index, d1_index, d2_index, unique_answer_index)
        
        # Step 7: Log results and save if requested
        self._log_split_results(d0_index, d1_index, d2_index, d0_docs, d1_docs, d2_docs, unique_answer_index)
        
        if save:
            logger.info(f"Saving LA2M split to {self.final_reference_path}")
            self.save_reference(d0_index, d1_index, d2_index)
        
        return d0_index, d1_index, d2_index
    
    def _process_answer_duplicates(self, answer_index: np.ndarray) -> np.ndarray:
        """Process answer duplicates based on remove_dup_answer setting."""
        if self.remove_dup_answer:
            unique_answer_index = np.unique(answer_index)
            removed_count = len(answer_index) - len(unique_answer_index)
            if removed_count > 0:
                logger.info(f"Removing duplicate answers: {len(answer_index)} → {len(unique_answer_index)} (removed {removed_count} duplicates)")
            return unique_answer_index
        else:
            logger.info(f"Keeping all answers including duplicates: {len(answer_index)} answer documents")
            return answer_index
    
    def _categorize_documents(self, dataset_index: np.ndarray, answer_index: np.ndarray) -> tuple[Set[int], Set[int]]:
        """Categorize documents into answer and non-answer sets."""
        all_docs = set(dataset_index)
        answer_docs = set(answer_index)
        
        # Ensure answer docs are actually in the dataset
        answer_docs_in_dataset = answer_docs.intersection(all_docs)
        non_answer_docs = all_docs - answer_docs_in_dataset
        
        logger.info(f"Document categorization:")
        logger.info(f"  • Total docs: {len(all_docs)}")
        logger.info(f"  • Answer docs in dataset: {len(answer_docs_in_dataset)}")
        logger.info(f"  • Non-answer docs: {len(non_answer_docs)}")
        
        if len(answer_docs_in_dataset) != len(answer_docs):
            excluded_count = len(answer_docs) - len(answer_docs_in_dataset)
            logger.warning(f"  • {excluded_count} answer docs not found in dataset (excluded)")
        
        return answer_docs_in_dataset, non_answer_docs
    
    def _allocate_reference_set(self, non_answer_docs: Set[int], reference_ratio: float, total_docs: int) -> Set[int]:
        """Allocate D0 (reference set) from non-answer documents only."""
        d0_target_size = int(total_docs * reference_ratio)
        available_non_answer = len(non_answer_docs)
        
        if available_non_answer < d0_target_size:
            logger.warning(f"Insufficient non-answer docs for reference set:")
            logger.warning(f"  • Target D0 size: {d0_target_size}")
            logger.warning(f"  • Available non-answer docs: {available_non_answer}")
            logger.warning(f"  • Adjusting D0 size to {available_non_answer}")
            d0_size = available_non_answer
        else:
            d0_size = d0_target_size
            logger.info(f"Allocating D0 (reference): {d0_size} docs from {available_non_answer} non-answer docs")
        
        # Randomly select D0 from non-answer documents
        if d0_size > 0:
            d0_docs = set(np.random.choice(list(non_answer_docs), size=d0_size, replace=False))
        else:
            d0_docs = set()
            logger.warning("D0 is empty - no non-answer documents available")
        
        return d0_docs
    
    def _distribute_remaining_documents(self, answer_docs: Set[int], remaining_non_answer_docs: Set[int]) -> tuple[Set[int], Set[int]]:
        """Evenly distribute remaining documents between D1 and D2."""
        logger.info("Distributing remaining documents between D1 and D2")
        
        # Step 1: Evenly split answer documents
        answer_docs_list = list(answer_docs)
        np.random.shuffle(answer_docs_list)
        
        mid_answers = len(answer_docs_list) // 2
        d1_answer_docs = set(answer_docs_list[:mid_answers])
        d2_answer_docs = set(answer_docs_list[mid_answers:])
        
        logger.info(f"Answer distribution: D1={len(d1_answer_docs)}, D2={len(d2_answer_docs)}")
        
        # Step 2: Evenly split remaining non-answer documents
        remaining_non_answer_list = list(remaining_non_answer_docs)
        np.random.shuffle(remaining_non_answer_list)
        
        mid_non_answer = len(remaining_non_answer_list) // 2
        d1_non_answer_docs = set(remaining_non_answer_list[:mid_non_answer])
        d2_non_answer_docs = set(remaining_non_answer_list[mid_non_answer:])
        
        logger.info(f"Non-answer distribution: D1={len(d1_non_answer_docs)}, D2={len(d2_non_answer_docs)}")
        
        # Step 3: Combine answer and non-answer for each set
        d1_docs = d1_answer_docs | d1_non_answer_docs
        d2_docs = d2_answer_docs | d2_non_answer_docs
        
        return d1_docs, d2_docs
    
    def _convert_to_indices(self, dataset_index: np.ndarray, d0_docs: Set[int], d1_docs: Set[int], d2_docs: Set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert document sets back to numpy arrays with original indices."""
        d0_index = np.array([doc for doc in dataset_index if doc in d0_docs])
        d1_index = np.array([doc for doc in dataset_index if doc in d1_docs])
        d2_index = np.array([doc for doc in dataset_index if doc in d2_docs])
        
        return d0_index, d1_index, d2_index
    
    def _validate_la2m_split(self, d0_index: np.ndarray, d1_index: np.ndarray, d2_index: np.ndarray, answer_index: np.ndarray):
        """Validate that LA2M constraints are satisfied."""
        logger.info("Validating LA2M constraints...")
        
        # Convert to sets for easier operations
        d0_set = set(d0_index)
        d1_set = set(d1_index)
        d2_set = set(d2_index)
        answer_set = set(answer_index)
        
        # Constraint 1: No overlap between sets
        self.check_overlap(d0_index, d1_index, d2_index)
        
        # Constraint 2: No answer documents in D0
        answers_in_d0 = d0_set.intersection(answer_set)
        if answers_in_d0:
            raise ValueError(f"LA2M constraint violated: {len(answers_in_d0)} answer documents found in D0 (reference)")
        
        # Constraint 3: All answer documents should be in D1 or D2
        answers_in_d1 = d1_set.intersection(answer_set)
        answers_in_d2 = d2_set.intersection(answer_set)
        total_answers_placed = len(answers_in_d1) + len(answers_in_d2)
        
        # Find answers that are actually in the dataset
        all_dataset_docs = d0_set | d1_set | d2_set
        answers_in_dataset = answer_set.intersection(all_dataset_docs)
        
        if total_answers_placed != len(answers_in_dataset):
            missing_answers = len(answers_in_dataset) - total_answers_placed
            raise ValueError(f"LA2M constraint violated: {missing_answers} answer documents not placed in D1 or D2")
        
        logger.info("✅ All LA2M constraints validated successfully")
        logger.info(f"  • No answers in D0: ✓")
        logger.info(f"  • All answers distributed in D1/D2: ✓ ({len(answers_in_d1)}/{len(answers_in_d2)})")
        logger.info(f"  • No overlaps between sets: ✓")
    
    def _log_split_results(self, d0_index: np.ndarray, d1_index: np.ndarray, d2_index: np.ndarray, 
                          d0_docs: Set[int], d1_docs: Set[int], d2_docs: Set[int], answer_index: np.ndarray):
        """Log detailed results of the split."""
        answer_set = set(answer_index)
        
        d1_answers = len(set(d1_index).intersection(answer_set))
        d2_answers = len(set(d2_index).intersection(answer_set))
        
        logger.info("=" * 50)
        logger.info("LA2M Split Results:")
        logger.info(f"  D0 (Reference): {len(d0_index):5d} docs (0 answers)")
        logger.info(f"  D1 (Model 1):   {len(d1_index):5d} docs ({d1_answers} answers)")
        logger.info(f"  D2 (Model 2):   {len(d2_index):5d} docs ({d2_answers} answers)")
        logger.info(f"  Total:          {len(d0_index) + len(d1_index) + len(d2_index):5d} docs")
        logger.info(f"  Answer balance: D1/D2 = {d1_answers}/{d2_answers}")
        logger.info("=" * 50)
    
    @classmethod
    def from_qrels(cls, dataset_name: str, dataset_index: List[str], internal_index: np.ndarray, dataset_obj: "Dataset", qrels: Dict, reference_ratio: float, reference_path: str, remove_dup_answer: bool = True, select_top_1: bool = True):
        """
        Create LA2MSplit instance from qrels data.
        
        Args:
            dataset_name: name of the dataset
            dataset_index: indices of all documents
            qrels: query relevance judgments {query_id: {doc_id: relevance_score}}
            reference_ratio: ratio for reference set
            reference_path: path to save reference data
            remove_dup_answer: whether to remove duplicate answer documents
            select_top_1: whether to select only the top 1 relevant answer for each query
            
        Returns:
            LA2MSplit instance
        """
        answer_index = cls._extract_answer_indices_from_qrels(qrels, dataset_index, select_top_1)
        answer_internal_index = dataset_obj.batch_original_ids_to_internal_indices(answer_index)
        return cls(dataset_name, dataset_index, internal_index, answer_internal_index, reference_ratio, reference_path, remove_dup_answer)
    
    @staticmethod
    def _extract_answer_indices_from_qrels(qrels: Dict, dataset_index: List[str], select_top_1: bool = True) -> List[str]:
        """
        Extract answer document indices from qrels.

        Args:
            qrels: query relevance judgments {query_id: {doc_id: relevance_score}}
            dataset_index: indices of all documents
            select_top_1: whether to select the top 1 relevant answer for each query

        Returns:
            answer_index: indices of answer documents
        """
        answer_doc_ids = set()
        
        logger.info(f"Extracting answer documents (select_top_1={select_top_1})")
        
        for query_id, doc_relevances in qrels.items():
            # Filter to only relevant documents (relevance > 0)
            relevant_docs = {doc_id: score for doc_id, score in doc_relevances.items() if score > 0}
            
            if not relevant_docs:
                logger.warning(f"No relevant documents found for query {query_id}")
                continue
            
            if select_top_1:
                # Select only the document with highest relevance score
                top_doc_id = max(relevant_docs.keys(), key=lambda x: relevant_docs[x])
                selected_docs = [top_doc_id]
                # logger.debug(f"Query {query_id}: selected top doc {top_doc_id} (score: {relevant_docs[top_doc_id]}) from {len(relevant_docs)} relevant docs")
            else:
                # Select all relevant documents
                selected_docs = list(relevant_docs.keys())
                # logger.debug(f"Query {query_id}: selected all {len(selected_docs)} relevant docs")
            
            # Convert doc_ids to integers and add to answer set
            for doc_id in selected_docs:
                try:
                    # Convert doc_id to int if it's a string
                    # doc_idx = int(doc_id) if isinstance(doc_id, str) else doc_id
                    answer_doc_ids.add(doc_id)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert doc_id {doc_id} to integer, skipping")
        
        # Filter to only include docs that are in dataset_index
        dataset_set = set(dataset_index)
        valid_answer_indices = answer_doc_ids.intersection(dataset_set)
        
        selection_mode = "top-1" if select_top_1 else "all relevant"
        logger.info(f"Extracted {len(valid_answer_indices)} valid answer documents using {selection_mode} strategy")
        logger.info(f"  • Total queries processed: {len(qrels)}")
        logger.info(f"  • Raw answer docs found: {len(answer_doc_ids)}")
        logger.info(f"  • Valid answer docs (in dataset): {len(valid_answer_indices)}")
        
        if len(answer_doc_ids) != len(valid_answer_indices):
            excluded = len(answer_doc_ids) - len(valid_answer_indices)
            logger.warning(f"Excluded {excluded} answer docs not found in dataset")

        return list(valid_answer_indices)
        