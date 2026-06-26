import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .util import get_knn_faiss

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.01):
        """
        简化的 Triplet Loss:
        - 正样本：从理想 topk 结果中选择（基于 target_rel 排序）
        - 负样本：从实际检索的 topk 结果中选择不在理想 topk 中的所有文档
        
        Args:
            margin: Triplet loss 的 margin
            num_ideal_top: 理想 topk 的数量
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self,
                query_emb: torch.Tensor,     # Shape: (B, D_query)
                doc_emb_all: torch.Tensor,   # Shape: (N_corpus, D_doc) - 全量文档库
                topk_idx: torch.Tensor,      # Shape: (B, K_retrieved) - 实际检索到的 topk 索引
                target_rel: torch.Tensor     # Shape: (B, K_retrieved) - 对应的相关度标签
               ) -> torch.Tensor:
        
        B, D_query = query_emb.shape
        N_corpus, D_doc = doc_emb_all.shape
        _B_k, K_retrieved = topk_idx.shape

        doc_emb_all = torch.nn.functional.normalize(doc_emb_all, dim=1)
        query_emb = torch.nn.functional.normalize(query_emb, dim=1)

        assert D_query == D_doc, "Query 和 Doc 嵌入维度必须匹配"
        assert B == _B_k, "Batch 维度不匹配"
        
        if K_retrieved < 2:
            return torch.tensor(0.0, device=query_emb.device, requires_grad=True)

        batch_triplet_losses = []

        # scores = torch.matmul(query_emb, doc_emb_all.T)
        # _, cur_topk_idx = torch.topk(scores, k=K_retrieved, dim=1)
        score, cur_topk_idx = get_knn_faiss(query_emb, doc_emb_all, K_retrieved)

        for i in range(B):
            # 理想的 topk（基于真实相关度）
            ideal_topk_indices = topk_idx[i]  # 这些是理想的正样本
            
            # 当前模型预测的 topk
            current_model_topk = cur_topk_idx[i]
            
            # 正样本：理想 topk 中的文档（无论是否在当前模型 topk 中）
            pos_idx = ideal_topk_indices
            
            # 负样本：当前模型 topk 中但不在理想 topk 中的文档
            in_cur_not_in_ideal = ~torch.isin(current_model_topk, ideal_topk_indices)
            neg_idx = current_model_topk[in_cur_not_in_ideal]
            
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            
            # 获取嵌入
            pos_emb = doc_emb_all[pos_idx]  # [num_pos, D]
            neg_emb = doc_emb_all[neg_idx]  # [num_neg, D]
            anchor = query_emb[i]  # [D]
            
            # 向量化计算所有正负样本对的损失
            num_pos = pos_emb.size(0)
            num_neg = neg_emb.size(0)
            
            # 扩展维度以便进行批量计算
            # anchor: [1, D] -> [num_pos * num_neg, D]
            anchor_expanded = anchor.unsqueeze(0).expand(num_pos * num_neg, -1)
            
            # pos_emb: [num_pos, D] -> [num_pos * num_neg, D]
            pos_expanded = pos_emb.unsqueeze(1).expand(-1, num_neg, -1).reshape(num_pos * num_neg, -1)
            
            # neg_emb: [num_neg, D] -> [num_pos * num_neg, D]
            neg_expanded = neg_emb.unsqueeze(0).expand(num_pos, -1, -1).reshape(num_pos * num_neg, -1)
            
            # 批量计算 triplet loss
            # losses = self.loss_fn(anchor_expanded, pos_expanded, neg_expanded)
            losses = F.triplet_margin_loss(
                anchor_expanded, pos_expanded, neg_expanded, margin=self.margin, reduction='mean')
            batch_triplet_losses.append(losses)
        
        if len(batch_triplet_losses) == 0:
            return torch.tensor(0.0, device=query_emb.device, requires_grad=True)
        else:
            return torch.stack(batch_triplet_losses).mean()
