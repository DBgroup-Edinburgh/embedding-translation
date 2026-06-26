import numpy as np
from typing import Any, Optional
from pathlib import Path
from loguru import logger
from pydantic import BaseModel


class SplitConfig(BaseModel):
    reference_method: str = "la2m"
    reference_ratio: float = 0.5
    reference_path: str = "./data/processed/reference/"
    reference_key: str = "la2m_split"
    save: bool = False
    verbose: bool = False
    remove_dup_answer: bool = True
    select_top_1: bool = True

    def update(self, **kwargs: Any) -> "SplitConfig":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


class LA2MSplitConfig(SplitConfig):
    remove_dup_answer: bool = False
    select_top_1: bool = True


def get_reference(reference_path: str, reference_key: str) -> dict:
    """
    Load the reference from the reference_path, the reference is a dictionary with keys "d0_index", "d1_index", and "d2_index"

    Args:
        reference_path: the path to the reference
        reference_key: the key of the reference

    Returns:
        reference: a dictionary with keys "d0_index", "d1_index", and "d2_index"
    """
    if not reference_key.endswith(".npz"):
        reference_key = f"{reference_key}.npz"
    if not (Path(reference_path) / reference_key).exists():
        raise FileNotFoundError(f"Reference key {reference_key} not found in {reference_path}")
    
    return np.load(Path(reference_path) / reference_key)

class BaseSplit:
    def __init__(self, dataset_name: str, internal_index: np.ndarray, reference_ratio: float, reference_path: str):
        self.internal_index = internal_index
        self.reference_ratio = reference_ratio
        self.reference_path = reference_path
        self.reference_path = Path(reference_path)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        self.reference_key = Path(f"random_split_{dataset_name}_{reference_ratio:.2f}")

    @property
    def final_reference_path(self):
        return self.reference_path / self.reference_key
    
    def split(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into three parts:
        - D0: reference
        - D1: query
        - D2: test
        """
        d0_index, d1_index, d2_index = self._split(**kwargs)
        self.check_overlap(d0_index, d1_index, d2_index)
        if "save" in kwargs and not kwargs["save"]:
            logger.info(f"Not saving reference to {self.final_reference_path} due to save=False")
        else:
            self.save_reference(d0_index, d1_index, d2_index)
        return d0_index, d1_index, d2_index

    def _split(self, internal_index: Optional[np.ndarray] = None, reference_ratio: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into three parts, basic random split.
        - D0: reference
        - D1: query
        - D2: test

        Args:
            dataset_index: the index of the dataset
            reference_ratio: the ratio of the reference
            save: whether to save the reference

        Returns:
            d0_index: the index of the reference
            d1_index: the index of the D1
            d2_index: the index of the D2
        """
        if internal_index is None:
            internal_index = self.internal_index
        if reference_ratio is None:
            reference_ratio = self.reference_ratio
        
        # random split reference_ratio of dataset_index to reference (D0), then evenly split the rest to d1 and d2 (D1 and D2)
        d0_index = np.random.choice(internal_index, size=int(len(internal_index) * reference_ratio), replace=False)
        remaining_index = np.setdiff1d(internal_index, d0_index)
        
        # Evenly split the remaining indices between D1 and D2
        np.random.shuffle(remaining_index)
        split_point = len(remaining_index) // 2
        d1_index = remaining_index[:split_point]
        d2_index = remaining_index[split_point:]

        return d0_index, d1_index, d2_index
    
    def check_overlap(self, d0_index: np.ndarray, d1_index: np.ndarray, d2_index: np.ndarray) -> bool:
        """
        Check if there is any overlap between D0, D1, and D2

        Args:
            d0_index: the index of the reference
            d1_index: the index of the D1
            d2_index: the index of the D2
        """
        if len(np.intersect1d(d0_index, d1_index)) > 0 or len(np.intersect1d(d0_index, d2_index)) > 0 or len(np.intersect1d(d1_index, d2_index)) > 0:
            raise ValueError("Overlap detected between D0, D1, and D2")
        return True
    
    def save_reference(self, d0_index: np.ndarray, d1_index: np.ndarray, d2_index: np.ndarray) -> None:
        """
        Save the reference to the reference_path, the reference is a dictionary with keys "d0_index", "d1_index", and "d2_index"

        Args:
            d0_index: the index of the reference
            d1_index: the index of the D1
            d2_index: the index of the D2
        """
        logger.info(f"Saving reference to {self.final_reference_path}")
        np.savez(
            self.final_reference_path,
            d0_index=d0_index,
            d1_index=d1_index,
            d2_index=d2_index
        )

