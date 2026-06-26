"""
Non-linear neural network mapping strategy for VectorMerge.

This module implements a deeper neural network with non-linear activations
for mapping between embedding spaces. It extends the linear mapping with
additional layers and activation functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Any
from pathlib import Path
from loguru import logger
from tqdm import trange

from ...core.mapping import MappingStrategy
from ...config import MappingConfig

class NonLinearMappingModel(nn.Module):
    """
    Deep non-linear neural network for embedding mapping.
    Uses multiple layers with non-linear activations.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout_rate: float = 0.1):
        """Initialize the non-linear mapping model.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout_rate: Dropout rate for regularization
        """
        super(NonLinearMappingModel, self).__init__()
        self.input_dim: int = input_dim
        self.hidden_dims: List[int] = hidden_dims
        self.output_dim: int = output_dim
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Non-linear activation
            layers.append(nn.ReLU())
            
            # Dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Final activation (tanh for bounded output)
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class NonLinearMappingStrategy(MappingStrategy):
    """
    Non-linear neural network mapping strategy.
    
    This strategy uses a deeper neural network with non-linear activations
    to learn complex mappings between embedding spaces.
    """
    
    def __init__(self, config: MappingConfig):
        """Initialize non-linear mapping strategy.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.model: NonLinearMappingModel
        self.optimizer: torch.optim.adam.Adam
        self.scheduler: torch.optim.lr_scheduler.StepLR
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        
        # Define hidden layer architecture
        self.hidden_dims = [
            config.nonlinear_config.hidden_dim, 
            config.nonlinear_config.hidden_dim * 2, 
            config.nonlinear_config.hidden_dim * 2, 
            config.nonlinear_config.hidden_dim
        ]
        self.dropout_rate = getattr(config.nonlinear_config, 'dropout_rate', 0.1)
        
        logger.info(f"Non-linear mapping initialized with hidden_dims={self.hidden_dims}")
    
    def _create_model(self, input_dim: int, output_dim: int) -> None:
        """Create and initialize the neural network model.
        
        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create model
        self.model = NonLinearMappingModel(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Create optimizer with weight decay
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=self.config.nonlinear_config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config.nonlinear_config.num_epochs // 3, 
            gamma=0.5
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Created non-linear model: {input_dim} -> {self.hidden_dims} -> {output_dim}")
        logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    def _create_dataloader(self, source_emb: torch.Tensor, 
                          target_emb: torch.Tensor, 
                          validation_split: float = 0.1) -> tuple:
        """Create training and validation DataLoaders.
        
        Args:
            source_emb: Source embeddings tensor
            target_emb: Target embeddings tensor
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split into train and validation
        n_samples = len(source_emb)
        n_val = int(n_samples * validation_split)
        
        # Random split
        indices = torch.randperm(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        # Create datasets
        train_dataset = TensorDataset(source_emb[train_indices], target_emb[train_indices])
        val_dataset = TensorDataset(source_emb[val_indices], target_emb[val_indices])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.nonlinear_config.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.nonlinear_config.batch_size, 
            shuffle=False
        ) if n_val > 0 else None
        
        return train_loader, val_loader
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for non-linear mapping.
        
        Args:
            predictions: Model predictions
            targets: Target embeddings
            
        Returns:
            Loss value
        """
        if self.config.nonlinear_config.loss_type == "mse":
            return nn.MSELoss()(predictions, targets)
        
        elif self.config.nonlinear_config.loss_type == "cosine":
            # Cosine embedding loss
            cos_sim = nn.CosineSimilarity(dim=1)(predictions, targets)
            return (1 - cos_sim).mean()
        
        elif self.config.nonlinear_config.loss_type == "huber":
            # Huber loss for robustness
            return nn.SmoothL1Loss()(predictions, targets)
        
        else:
            # Default to MSE
            return nn.MSELoss()(predictions, targets)
    
    def _fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
            reference_indices: np.ndarray, **kwargs) -> None:
        """Fit the non-linear mapping model.
        
        Args:
            source_embeddings: Source embedding space
            target_embeddings: Target embedding space
            reference_indices: Reference point indices for training
            **kwargs: Additional arguments
        """
        logger.info(f"Fitting non-linear mapping with {len(reference_indices)} reference points")
        
        # Extract reference embeddings
        source_ref = source_embeddings[reference_indices]
        target_ref = target_embeddings[reference_indices]
        
        # Convert to tensors
        source_ref_tensor = torch.from_numpy(source_ref).float().to(self.device)
        target_ref_tensor = torch.from_numpy(target_ref).float().to(self.device)
        
        # Create model
        self._create_model(source_ref.shape[1], target_ref.shape[1])
        
        # Create data loaders
        train_loader, val_loader = self._create_dataloader(source_ref_tensor, target_ref_tensor)
        
        # Training loop
        training_losses = []
        validation_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.nonlinear_config.num_epochs // 10  # Early stopping patience
        
        epoch_bar = trange(self.config.nonlinear_config.num_epochs, desc="Training")
        for epoch in epoch_bar:
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            for batch_source, batch_target in train_loader:
                # Forward pass
                predictions = self.model(batch_source)
                loss = self._compute_loss(predictions, batch_target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_source, batch_target in val_loader:
                        predictions = self.model(batch_source)
                        loss = self._compute_loss(predictions, batch_target)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                validation_losses.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Step learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record losses
            avg_train_loss = epoch_train_loss / num_train_batches
            training_losses.append(avg_train_loss)
            
            # Update progress bar
            if val_loader is not None:
                epoch_bar.set_description(
                    f"Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}"
                )
            else:
                epoch_bar.set_description(f"Train: {avg_train_loss:.6f}")
            
            # Log every 20 epochs
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.nonlinear_config.num_epochs}, "
                           f"Train Loss: {avg_train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience and epoch > self.config.nonlinear_config.num_epochs // 2:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.is_fitted = True
        self.metadata = {
            'reference_size': len(reference_indices),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'num_epochs_trained': len(training_losses),
            'loss_type': self.config.nonlinear_config.loss_type,
            'final_train_loss': training_losses[-1] if training_losses else None,
            'final_val_loss': validation_losses[-1] if validation_losses else None,
            'best_val_loss': best_val_loss,
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }
        
        logger.info(f"Non-linear mapping training completed. "
                   f"Final train loss: {training_losses[-1]:.6f}")
    
    def _transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using the fitted non-linear model.
        
        Args:
            embeddings: Embeddings to transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed embeddings
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before transformation")
        
        # Convert to tensor
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        
        # Transform in evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Process in batches
            batch_size = self.config.nonlinear_config.batch_size
            num_samples = len(embeddings)
            transformed_list = []
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch = embeddings_tensor[i:end_idx]
                
                # Transform batch
                transformed_batch = self.model(batch)
                transformed_list.append(transformed_batch.cpu().numpy())
            
            # Concatenate all batches
            transformed = np.concatenate(transformed_list, axis=0)
        
        return transformed
    
    def save(self, path) -> None:
        """Save the fitted non-linear mapping model."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save unfitted model")
        
        super().save(path)
        
        # Save model state
        save_path = Path(path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, save_path / "model_checkpoint.pth")
        
        # Save model architecture info
        model_info = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        }
        
        import json
        with open(save_path / "model_architecture.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Saved non-linear mapping model to {save_path}")
    
    @classmethod
    def load(cls, path) -> 'NonLinearMappingStrategy':
        """Load a fitted non-linear mapping model."""
        instance = NonLinearMappingStrategy(MappingConfig())
        load_path = Path(path)
        
        # Load model architecture info
        import json
        with open(load_path / "model_architecture.json", "r") as f:
            model_info = json.load(f)
        
        # Update instance attributes
        instance.hidden_dims = model_info['hidden_dims']
        instance.dropout_rate = model_info['dropout_rate']
        
        # Create model
        instance._create_model(model_info['input_dim'], model_info['output_dim'])
        
        # Load model checkpoint
        if instance.model is not None:
            checkpoint = torch.load(load_path / "model_checkpoint.pth", 
                                   map_location=instance.device, weights_only=True)
            
            instance.model.load_state_dict(checkpoint['model_state_dict'])
            
            if instance.optimizer and checkpoint['optimizer_state_dict']:
                instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if instance.scheduler and checkpoint['scheduler_state_dict']:
                instance.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            instance.model.eval()
        
        logger.info(f"Loaded non-linear mapping model from {load_path}")
        instance.is_fitted = True
        return instance 

    @classmethod
    def check_fit(cls, path) -> bool:
        """Check if the model is fitted."""
        load_path = Path(path)
        return load_path.exists() and (load_path / "model_architecture.json").exists() and (load_path / "model_checkpoint.pth").exists()