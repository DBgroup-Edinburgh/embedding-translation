import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LambdaRankLoss(nn.Module):
    def __init__(self, k: int = None):
        super().__init__()
        self.k = k

    def forward(
        self,
        query_emb: torch.Tensor,     # [B, D]
        doc_emb: torch.Tensor,       # [N, D] (全量库)
        topk_idx: torch.Tensor,      # [B, K] (FAISS 返回的 top-k 索引)
        target_rel: torch.Tensor     # [B, K] (与 topk_idx 对应的 relevance labels)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = query_emb.size()
        K = topk_idx.size(1)
        k = self.k or K

        # 1. 抽取 top-k 文档嵌入： doc_batch_emb: [B, K, D]
        doc_batch_emb = doc_emb[topk_idx]  # 利用索引广播，PyTorch 支持这种 [B, K] 直接索引 [N, D]

        # 2. 内积打分： [B, K]
        # query_emb = query_emb.unsqueeze(1)              # [B, 1, D]
        # query_emb = torch.nn.functional.normalize(query_emb, dim=1)
        # doc_batch_emb = torch.nn.functional.normalize(doc_batch_emb, dim=1)
        # scores = torch.sum(query_emb * doc_batch_emb, dim=-1)  # [B, K]
        scores = F.cosine_similarity(query_emb, doc_batch_emb, dim=-1)

        # 3. NDCG 计算
        with torch.no_grad():
            ideal_rel, _ = torch.sort(target_rel, dim=1, descending=True)  # [B, K]
            discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
            idcg = torch.sum(ideal_rel[:, :k] * discount.unsqueeze(0), dim=1)  # [B]

            _, rank_idx = torch.sort(scores, dim=1, descending=True)
            ranked_rel = torch.gather(target_rel, 1, rank_idx[:, :k])
            dcg = torch.sum(ranked_rel * discount.unsqueeze(0), dim=1)
            ndcg = dcg / (idcg + 1e-8)

        # 4. LambdaLoss
        rel_i = target_rel.unsqueeze(2)  # [B, K, 1]
        rel_j = target_rel.unsqueeze(1)  # [B, 1, K]
        S = torch.sign(rel_i - rel_j)    # [B, K, K]

        d_full = 1.0 / torch.log2(torch.arange(2, K + 2, device=scores.device).float())  # [K]
        d_i = d_full.view(-1, 1)
        d_j = d_full.view(1, -1)
        delta_discount = (d_i - d_j).abs()  # [K, K]
        delta_discount = delta_discount.unsqueeze(0).expand(B, -1, -1)

        delta_ndcg = (rel_i - rel_j).abs() * delta_discount / (idcg.view(B, 1, 1) + 1e-8)

        mask = torch.triu(torch.ones_like(delta_ndcg), diagonal=1)
        delta_ndcg = delta_ndcg * mask

        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B, K, K]
        pairwise_loss = F.softplus(-S * score_diff)

        loss = (delta_ndcg * pairwise_loss).sum(dim=(1, 2)) / delta_ndcg.sum(dim=(1, 2)).clamp(min=1e-6)
        final_loss = loss.mean()

        return final_loss, ndcg.mean()