from .base_split import BaseSplit as RandomSplit, get_reference, SplitConfig, LA2MSplitConfig
from .la2m_split import LA2MSplit
from .manage import split_dataset, get_split_config
SUPPORTED_REFERENCE_STRATEGIES = ["random", "la2m"]

__all__ = ["RandomSplit", "get_reference", "LA2MSplit", "SUPPORTED_REFERENCE_STRATEGIES", "SplitConfig", "LA2MSplitConfig", "split_dataset", "get_split_config"]