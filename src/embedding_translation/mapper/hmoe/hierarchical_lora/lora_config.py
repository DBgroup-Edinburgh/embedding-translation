"""
LoRA configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA layers."""
    
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    scaling: Optional[float] = None
    
    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.alpha / self.rank

