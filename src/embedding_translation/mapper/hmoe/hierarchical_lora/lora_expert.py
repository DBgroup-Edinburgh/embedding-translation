"""
LoRA expert implementation.

This module implements a parameter-efficient expert using LoRA (Low-Rank Adaptation).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from loguru import logger

from .lora_config import LoRAConfig
from .lora_layer import LoRALayer


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Combines a frozen base linear layer with trainable LoRA parameters.
    Output = base_linear(x) + lora_layer(x)
    """
    
    def __init__(
        self,
        base_linear: nn.Linear,
        lora_config: LoRAConfig,
        freeze_base: bool = True
    ):
        super().__init__()
        self.base_linear = base_linear
        self.freeze_base = freeze_base
        
        # Freeze base model parameters if requested
        if freeze_base:
            for param in self.base_linear.parameters():
                param.requires_grad = False
        
        # Create LoRA layer
        self.lora_layer = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base linear and LoRA.
        
        Args:
            x: Input tensor
            
        Returns:
            base_output + lora_output
        """
        # Base model output
        base_out = self.base_linear(x)
        
        # LoRA adaptation
        lora_out = self.lora_layer(x)
        
        return base_out + lora_out
    
    def merge_weights(self):
        """
        Merge LoRA weights into base linear layer.
        
        This is useful for inference - merges the LoRA adaptation
        into the base weights to avoid the additional computation.
        """
        if self.freeze_base:
            logger.warning("Cannot merge weights when base is frozen")
            return
        
        with torch.no_grad():
            # Compute LoRA delta: scaling * B @ A
            lora_delta = self.lora_layer.scaling * (
                self.lora_layer.lora_B @ self.lora_layer.lora_A.T
            )
            
            # Add to base weight: W' = W + BA
            self.base_linear.weight.data += lora_delta.T


class LoRAExpert(nn.Module):
    """
    Expert using LoRA for parameter-efficient fine-tuning.
    
    This class wraps a base model (e.g., SimpleLinearMapper) and injects
    LoRA layers into all Linear layers, allowing efficient adaptation
    while keeping the base model frozen.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lora_config: LoRAConfig,
        freeze_base: bool = True,
        target_modules: Optional[List[str]] = None
    ):
        """
        Initialize LoRA expert.
        
        Args:
            base_model: Base model to adapt (e.g., SimpleLinearMapper)
            lora_config: LoRA configuration
            freeze_base: Whether to freeze base model parameters
            target_modules: List of module names to apply LoRA to.
                          If None, applies to all nn.Linear modules.
        """
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config
        self.freeze_base = freeze_base
        self.target_modules = target_modules
        
        # Store original modules for reference
        self.lora_modules: Dict[str, LoRALinear] = {}
        
        # Inject LoRA layers into base model
        self._inject_lora_layers()
        
        # Log parameter counts
        self._log_parameter_info()
    
    def _inject_lora_layers(self):
        """
        Inject LoRA layers into the base model.
        
        Replaces all nn.Linear layers (or specified target modules)
        with LoRALinear layers.
        """
        modules_replaced = 0
        
        for name, module in self.base_model.named_modules():
            # Skip the root module
            if name == '':
                continue
            
            # Check if this is a Linear layer
            if isinstance(module, nn.Linear):
                # Check if we should apply LoRA to this module
                if self.target_modules is None or any(target in name for target in self.target_modules):
                    # Create LoRALinear wrapper
                    lora_linear = LoRALinear(
                        base_linear=module,
                        lora_config=self.lora_config,
                        freeze_base=self.freeze_base
                    )
                    
                    # Replace the module in base_model
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = self.base_model.get_submodule(parent_name)
                    else:
                        parent = self.base_model
                    
                    setattr(parent, child_name, lora_linear)
                    
                    # Store reference
                    self.lora_modules[name] = lora_linear
                    modules_replaced += 1
        
        logger.info(f"Injected LoRA into {modules_replaced} Linear layers")
    
    def _log_parameter_info(self):
        """Log information about trainable and frozen parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(f"LoRA Expert Parameters:")
        logger.info(f"  Total: {total_params:,}")
        logger.info(f"  Trainable (LoRA): {trainable_params:,} ({trainable_pct:.2f}%)")
        logger.info(f"  Frozen (Base): {frozen_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA-adapted model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output from base_model with LoRA adaptations
        """
        return self.base_model(x)
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """
        Get only the LoRA parameters (for optimization).
        
        Returns:
            List of LoRA parameters
        """
        lora_params = []
        for module in self.lora_modules.values():
            lora_params.extend([
                module.lora_layer.lora_A,
                module.lora_layer.lora_B
            ])
        return lora_params
    
    def merge_and_unload(self):
        """
        Merge LoRA weights into base model and return the base model.
        
        This is useful for deployment - creates a single model with
        the LoRA adaptations merged in.
        
        Returns:
            Base model with merged weights
        """
        if self.freeze_base:
            logger.warning("Cannot merge when base is frozen. Returning model as-is.")
            return self.base_model
        
        # Merge all LoRA layers
        for name, lora_module in self.lora_modules.items():
            lora_module.merge_weights()
        
        logger.info("Merged LoRA weights into base model")
        return self.base_model
    
    def save_lora_weights(self, path: str):
        """
        Save only the LoRA parameters (not the base model).
        
        Args:
            path: Path to save LoRA weights
        """
        lora_state = {
            name: {
                'lora_A': module.lora_layer.lora_A,
                'lora_B': module.lora_layer.lora_B,
                'scaling': module.lora_layer.scaling
            }
            for name, module in self.lora_modules.items()
        }
        
        torch.save({
            'lora_modules': lora_state,
            'lora_config': self.lora_config,
        }, path)
        
        logger.info(f"Saved LoRA weights to {path}")
    
    def load_lora_weights(self, path: str):
        """
        Load LoRA parameters from a saved file.
        
        Args:
            path: Path to load LoRA weights from
        """
        checkpoint = torch.load(path)
        lora_state = checkpoint['lora_modules']
        
        for name, module in self.lora_modules.items():
            if name in lora_state:
                module.lora_layer.lora_A.data = lora_state[name]['lora_A']
                module.lora_layer.lora_B.data = lora_state[name]['lora_B']
        
        logger.info(f"Loaded LoRA weights from {path}")


def count_lora_parameters(model: LoRAExpert) -> Dict[str, int]:
    """
    Count LoRA parameters in a model.
    
    Args:
        model: LoRAExpert instance
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_only = sum(p.numel() for p in model.get_lora_parameters())
    
    return {
        'total': total,
        'trainable': trainable,
        'lora': lora_only,
        'frozen': total - trainable,
        'trainable_percentage': 100 * trainable / total if total > 0 else 0
    }

