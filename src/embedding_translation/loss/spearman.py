import torch
import torch.nn as nn
import torch.nn.functional as F

class SpearmanRankLoss(nn.Module):
    def __init__(self):
        super(SpearmanRankLoss, self).__init__()

    def forward(
        self,
        query_emb: torch.Tensor,   # [B, D]
        doc_emb: torch.Tensor,     # [N, D]
        index: torch.Tensor,       # [B, K]  每个 query 对应的 top-K 文档索引
        score: torch.Tensor        # [B, K]  对应文档的相关度标签（越大越相关）
    ) -> torch.Tensor:
        B, D = query_emb.size()
        K = index.size(1)

        # 1) 提取 top-K 文档嵌入 [B, K, D]
        doc_batch = doc_emb[index]  # [B, K, D]

        # 2) 计算预测得分（内积或余弦）
        pred_scores = F.cosine_similarity(query_emb, doc_batch, dim=-1)  # [B, K]

        from torchsort import soft_rank

        pred_soft_rank = soft_rank(pred_scores, regularization_strength=1.0)
        true_soft_rank = soft_rank(score, regularization_strength=1.0)


        # 4) 计算 Spearman 相关系数
        diff_squared = (pred_soft_rank - true_soft_rank).pow(2).sum(dim=1)  # [B]
        spearman_corr = 1 - 6 * diff_squared / (K * (K**2 - 1))   # [B]

        # 5) 返回损失 = 1 - 相关系数
        loss = 1 - spearman_corr.mean()
        return loss