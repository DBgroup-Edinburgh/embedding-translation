"""
LoRA training utilities.

Helper functions for training and managing LoRA adapters.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from loguru import logger

from .lora_expert import LoRAExpert


def train_lora_adapter(
    lora_expert: LoRAExpert,
    train_loader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a single LoRA adapter.
    
    Args:
        lora_expert: LoRAExpert instance with frozen base model
        train_loader: DataLoader with training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        verbose: Whether to log training progress
        
    Returns:
        Dictionary with training statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lora_expert.to(device)
    
    # Setup optimizer (only for LoRA parameters!)
    optimizer = torch.optim.Adam(
        lora_expert.get_lora_parameters(),
        lr=learning_rate
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training statistics
    epoch_losses = []
    
    # Training loop
    lora_expert.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for src_batch, tgt_batch in train_loader:
            # Move to device
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (base + LoRA)
            outputs = lora_expert(src_batch)
            
            # Compute loss
            loss = criterion(outputs, tgt_batch)
            
            # Backward pass (only LoRA parameters get gradients)
            loss.backward()
            
            # Update LoRA parameters
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(avg_loss)
        
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            logger.debug(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    lora_expert.eval()
    
    return {
        'epoch_losses': epoch_losses,
        'final_loss': epoch_losses[-1] if epoch_losses else None,
        'num_epochs': num_epochs
    }


def estimate_lora_memory(
    base_model: nn.Module,
    lora_rank: int,
    num_adapters: int,
    bytes_per_param: int = 4
) -> Dict[str, float]:
    """
    Estimate memory usage for LoRA adapters.
    
    Args:
        base_model: Base model to wrap with LoRA
        lora_rank: LoRA rank
        num_adapters: Number of LoRA adapters
        bytes_per_param: Bytes per parameter (4 for float32, 2 for float16)
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Count Linear layers
    num_linear_layers = sum(
        1 for module in base_model.modules()
        if isinstance(module, nn.Linear)
    )
    
    # Estimate LoRA parameters per adapter
    # For each Linear(in_features, out_features):
    # LoRA adds: A (in_features, rank) + B (rank, out_features)
    total_lora_params_per_adapter = 0
    for module in base_model.modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            lora_params = in_features * lora_rank + lora_rank * out_features
            total_lora_params_per_adapter += lora_params
    
    # Base model parameters
    base_params = sum(p.numel() for p in base_model.parameters())
    
    # Memory calculations (in MB)
    base_memory_mb = (base_params * bytes_per_param) / (1024 ** 2)
    lora_per_adapter_mb = (total_lora_params_per_adapter * bytes_per_param) / (1024 ** 2)
    total_lora_memory_mb = lora_per_adapter_mb * num_adapters
    total_memory_mb = base_memory_mb + total_lora_memory_mb
    
    # Comparison with full experts
    full_experts_memory_mb = base_memory_mb * num_adapters
    compression_ratio = full_experts_memory_mb / total_memory_mb
    
    return {
        'base_model_mb': base_memory_mb,
        'lora_per_adapter_mb': lora_per_adapter_mb,
        'total_lora_mb': total_lora_memory_mb,
        'total_memory_mb': total_memory_mb,
        'full_experts_mb': full_experts_memory_mb,
        'compression_ratio': compression_ratio,
        'num_linear_layers': num_linear_layers,
        'lora_params_per_adapter': total_lora_params_per_adapter,
        'base_params': base_params
    }


def log_lora_statistics(
    base_model: nn.Module,
    lora_adapters: Dict[int, LoRAExpert],
    lora_rank: int
):
    """
    Log comprehensive statistics about LoRA adapters.
    
    Args:
        base_model: Base model
        lora_adapters: Dictionary of LoRA adapters
        lora_rank: LoRA rank used
    """
    if len(lora_adapters) == 0:
        logger.warning("No LoRA adapters to analyze")
        return
    
    # Get first adapter for analysis
    first_adapter = next(iter(lora_adapters.values()))
    
    # Parameter counts
    base_params = sum(p.numel() for p in base_model.parameters())
    lora_params_per_adapter = sum(
        p.numel() for p in first_adapter.get_lora_parameters()
    )
    total_lora_params = lora_params_per_adapter * len(lora_adapters)
    total_params = base_params + total_lora_params
    
    # Calculate what full experts would cost
    full_experts_params = base_params * len(lora_adapters)
    compression_ratio = full_experts_params / total_params
    
    # Memory estimation
    mem_stats = estimate_lora_memory(base_model, lora_rank, len(lora_adapters))
    
    # Log everything
    logger.info("=" * 70)
    logger.info("LoRA Statistics Summary")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Number of adapters: {len(lora_adapters)}")
    logger.info(f"  LoRA rank: {lora_rank}")
    logger.info(f"  Number of Linear layers: {mem_stats['num_linear_layers']}")
    logger.info("")
    logger.info(f"Parameter Counts:")
    logger.info(f"  Base model: {base_params:,}")
    logger.info(f"  LoRA per adapter: {lora_params_per_adapter:,}")
    logger.info(f"  Total LoRA: {total_lora_params:,}")
    logger.info(f"  Grand total: {total_params:,}")
    logger.info(f"  LoRA overhead: {(lora_params_per_adapter / base_params * 100):.2f}% per adapter")
    logger.info("")
    logger.info(f"Comparison with Full Experts:")
    logger.info(f"  Full experts would need: {full_experts_params:,} parameters")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}x")
    logger.info(f"  Parameter savings: {(1 - 1/compression_ratio) * 100:.1f}%")
    logger.info("")
    logger.info(f"Memory Estimates (float32):")
    logger.info(f"  Base model: {mem_stats['base_model_mb']:.1f} MB")
    logger.info(f"  LoRA per adapter: {mem_stats['lora_per_adapter_mb']:.1f} MB")
    logger.info(f"  Total LoRA: {mem_stats['total_lora_mb']:.1f} MB")
    logger.info(f"  Total: {mem_stats['total_memory_mb']:.1f} MB")
    logger.info(f"  Full experts: {mem_stats['full_experts_mb']:.1f} MB")
    logger.info(f"  Memory savings: {(1 - mem_stats['total_memory_mb']/mem_stats['full_experts_mb']) * 100:.1f}%")
    logger.info("=" * 70)


def compute_lora_efficiency_score(
    base_model: nn.Module,
    lora_rank: int,
    num_adapters: int
) -> float:
    """
    Compute an efficiency score for LoRA configuration.
    
    Higher score means better parameter efficiency.
    
    Args:
        base_model: Base model
        lora_rank: LoRA rank
        num_adapters: Number of adapters
        
    Returns:
        Efficiency score (compression ratio)
    """
    mem_stats = estimate_lora_memory(base_model, lora_rank, num_adapters)
    return mem_stats['compression_ratio']

