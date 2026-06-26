"""
LoRA layer implementation.
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Basic LoRA layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: scaling * B @ A @ x
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Apply dropout to input
        x_dropped = self.dropout(x)
        
        # LoRA computation: x @ A @ B with scaling
        # x: (batch, in_features)
        # A: (in_features, rank)
        # B: (rank, out_features)
        lora_out = x_dropped @ self.lora_A @ self.lora_B
        
        return self.scaling * lora_out

