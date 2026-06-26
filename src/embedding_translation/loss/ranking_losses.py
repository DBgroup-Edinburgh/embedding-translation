import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np

class ListNetLoss(nn.Module):
    """ListNet loss implementation for both binary and graded relevance."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        pred_probs = F.softmax(pred_scores / self.temperature, dim=1)
        target_probs = F.softmax(target_scores / self.temperature, dim=1)
        return -torch.sum(target_probs * torch.log(pred_probs + 1e-8))

class ListMLELoss(nn.Module):
    """ListMLE loss implementation."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        # Sort target scores in descending order
        _, target_indices = torch.sort(target_scores, descending=True)
        sorted_pred_scores = torch.gather(pred_scores, 1, target_indices)
        
        # Compute loss
        pred_scores_exp = torch.exp(sorted_pred_scores / self.temperature)
        cumsum = torch.cumsum(pred_scores_exp, dim=1)
        loss = torch.sum(sorted_pred_scores / self.temperature - torch.log(cumsum + 1e-8))
        return -loss

class RankNetLoss(nn.Module):
    """RankNet loss implementation."""
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        # Create pairs
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)
        target_diff = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)
        
        # Compute loss
        loss = torch.log(1 + torch.exp(-self.sigma * pred_diff)) * (target_diff > 0).float()
        return loss.mean()

class OrdinalLoss(nn.Module):
    """Ordinal loss implementation."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        # Convert scores to ordinal classes
        target_classes = torch.floor(target_scores * (self.num_classes - 1)).long()
        
        # Compute cumulative probabilities
        probs = torch.sigmoid(pred_scores)
        cum_probs = torch.cumprod(probs, dim=1)
        
        # Compute loss
        loss = -torch.sum(target_classes * torch.log(cum_probs + 1e-8) + 
                         (1 - target_classes) * torch.log(1 - cum_probs + 1e-8))
        return loss

class LambdaRankLoss(nn.Module):
    """Improved LambdaRank loss implementation."""
    def __init__(self, sigma: float = 1.0, k: int = 10):
        super().__init__()
        self.sigma = sigma
        self.k = k  # 只考虑前k个文档
        
    def compute_dcg(self, scores: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Compute DCG for a batch of scores.
        
        Args:
            scores: Relevance scores (batch_size x list_size)
            k: Number of documents to consider
            
        Returns:
            DCG values for each query
        """
        if k is None:
            k = scores.size(1)
            
        # 计算位置折扣
        discounts = 1.0 / torch.log2(torch.arange(2, k + 2).float().to(scores.device))
        discounts = discounts.unsqueeze(0)  # (1 x k)
        
        # 计算 DCG
        dcg = torch.sum(scores[:, :k] * discounts, dim=1)
        return dcg
        
    def compute_ndcg(self, pred_scores: torch.Tensor, target_scores: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Compute NDCG for a batch of predictions and targets.
        
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
            k: Number of documents to consider
            
        Returns:
            NDCG values for each query
        """
        if k is None:
            k = pred_scores.size(1)
            
        # 计算预测排序的 DCG
        pred_dcg = self.compute_dcg(pred_scores, k)
        
        # 计算理想 DCG（使用目标分数排序）
        ideal_dcg = self.compute_dcg(target_scores, k)
        
        # 计算 NDCG
        ndcg = pred_dcg / (ideal_dcg + 1e-8)
        return ndcg
        
    def compute_lambda(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute lambda values for each document pair.
        
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
            
        Returns:
            Lambda values for each document pair
        """
        batch_size, list_size = pred_scores.size()
        
        # 计算目标分数差异
        target_diff = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)  # (batch_size x list_size x list_size)
        
        # 计算预测分数差异
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)  # (batch_size x list_size x list_size)
        
        # 计算位置折扣
        discounts = 1.0 / torch.log2(torch.arange(2, list_size + 2).float().to(pred_scores.device))
        discounts = discounts.unsqueeze(0).unsqueeze(0)  # (1 x 1 x list_size)
        
        # 计算 NDCG 差异
        ndcg_diff = torch.abs(discounts.unsqueeze(2) - discounts.unsqueeze(1))  # (1 x list_size x list_size)
        
        # 计算 lambda 值
        lambda_ij = -self.sigma * (1 / (1 + torch.exp(self.sigma * pred_diff))) * ndcg_diff * (target_diff > 0).float()
        
        return lambda_ij
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        # 计算 lambda 值
        lambda_ij = self.compute_lambda(pred_scores, target_scores)
        
        # 计算预测分数差异
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)
        
        # 计算损失
        loss = torch.sum(lambda_ij * pred_diff)
        
        # 计算 NDCG 作为监控指标
        with torch.no_grad():
            ndcg = self.compute_ndcg(pred_scores, target_scores, self.k)
            ndcg_mean = ndcg.mean()
            
        return -loss, ndcg_mean

class ApproxNDCGLoss(nn.Module):
    """Approximate NDCG loss implementation."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        # Compute DCG
        with torch.no_grad():
            ideal_dcg = torch.sum(target_scores / torch.log2(torch.arange(2, target_scores.size(1) + 2).float().to(target_scores.device)))
        
        # Compute approximate DCG
        pred_probs = F.softmax(pred_scores / self.temperature, dim=1)
        approx_dcg = torch.sum(pred_probs * target_scores / torch.log2(torch.arange(2, target_scores.size(1) + 2).float().to(target_scores.device)))
        
        return -approx_dcg / (ideal_dcg + 1e-8)

class NeuralNDCGLoss(nn.Module):
    """Neural NDCG loss implementation."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        # Compute neural ranking
        pred_probs = F.softmax(pred_scores / self.temperature, dim=1)
        
        # Compute DCG
        dcg = torch.sum(pred_probs * target_scores / torch.log2(torch.arange(2, target_scores.size(1) + 2).float().to(target_scores.device)))
        
        # Compute ideal DCG
        ideal_dcg = torch.sum(target_scores / torch.log2(torch.arange(2, target_scores.size(1) + 2).float().to(target_scores.device)))
        
        return -dcg / (ideal_dcg + 1e-8)

class RMSELoss(nn.Module):
    """Root Mean Square Error loss implementation."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scores (batch_size x list_size)
            target_scores: Target scores (batch_size x list_size)
        """
        return torch.sqrt(torch.mean((pred_scores - target_scores) ** 2))

def get_ranking_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get ranking loss.
    
    Args:
        loss_type: Type of loss to use
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
    """
    loss_functions = {
        'listnet': ListNetLoss,
        'listmle': ListMLELoss,
        'ranknet': RankNetLoss,
        'ordinal': OrdinalLoss,
        'lambdarank': LambdaRankLoss,
        'approxndcg': ApproxNDCGLoss,
        'neuralndcg': NeuralNDCGLoss,
        'rmse': RMSELoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # 根据不同的损失函数类型处理参数
    if loss_type == 'lambdarank':
        # LambdaRank 只需要 sigma 和 k 参数
        sigma = kwargs.get('sigma', 1.0)
        k = kwargs.get('k', 10)
        return loss_functions[loss_type](sigma=sigma, k=k)
    elif loss_type in ['listnet', 'listmle', 'approxndcg', 'neuralndcg']:
        # 这些损失函数使用 temperature 参数
        temperature = kwargs.get('temperature', 1.0)
        return loss_functions[loss_type](temperature=temperature)
    elif loss_type == 'ranknet':
        # RankNet 使用 sigma 参数
        sigma = kwargs.get('sigma', 1.0)
        return loss_functions[loss_type](sigma=sigma)
    elif loss_type == 'ordinal':
        # Ordinal 需要 num_classes 参数
        num_classes = kwargs.get('num_classes', 5)
        return loss_functions[loss_type](num_classes=num_classes)
    else:
        # RMSE 不需要额外参数
        return loss_functions[loss_type]() 