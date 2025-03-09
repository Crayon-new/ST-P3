import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVCrossAttention(nn.Module):
    """BEV特征交叉注意力模块（参考CarPlanner[3](@ref)和BEVDriver[8](@ref)设计）"""
    def __init__(self, attn_dim=64):
        super().__init__()
        self.query_proj = nn.Linear(attn_dim, attn_dim)
        self.bev_proj = nn.Conv2d(attn_dim, attn_dim, 1)
        self.attn = nn.MultiheadAttention(attn_dim, 8)
        
    def forward(self, query, bev_feat, bev_mask):
        # query: [B, 64]
        # bev_feat: [B, C, H, W]
        # bev_mask: [B, 1, H, W] (分割概率图)
        
        # 特征投影
        q = self.query_proj(query).unsqueeze(0)  # [1, B, D]
        k = self.bev_proj(bev_feat).flatten(2).permute(2,0,1)  # [HW, B, D]
        v = bev_feat.flatten(2).permute(2,0,1)  # [HW, B, C]
        
        # 注意力计算（加入分割图权重）
        # attn_mask = (1 - bev_mask.flatten(1)) * -1e9  # 低概率区域mask
        attn_mask = None
        attn_out, _ = self.attn(q, k, v)
        return attn_out.squeeze(0)

class AutoRegressivePlanner(nn.Module):
    """自回归轨迹预测器（融合BEVFormer[9](@ref)和SocialLSTM[6](@ref)思想）"""
    def __init__(self, 
                 query_dim=64,
                 pred_steps=6):
        super().__init__()
        self.bev_attentions = BEVCrossAttention(query_dim)
        self.time_embed = nn.Embedding(pred_steps, query_dim)
        
        # 自回归核心（GRU+Transformer混合结构）
        self.gru = nn.GRUCell(query_dim, query_dim)
        self.transformer_decoder = nn.TransformerDecoderLayer(
            d_model=query_dim, nhead=8, dim_feedforward=512
        )
        self.loc_pred = nn.Linear(query_dim, 3)  # 预测(x,y,theta)
        
    def forward(self, init_query, bev_feats, bev_masks):
        # init_query: [B, 64] 
        # bev_feats: [B, 6, C, H, W] (6个时间步的BEV特征)
        # bev_masks: [B, 6, H, W] (分割概率图)
        if init_query is None:
            init_query = torch.zeros(bev_feats.size(0), 64, device=bev_feats.device)
 
        batch_size = init_query.size(0)
        device = init_query.device
        
        # 初始化隐藏状态
        hidden = init_query
        preds = []
        
        for t in range(6):
            # 时间嵌入编码
            time_emb = self.time_embed(torch.tensor([t], device=device))
            time_emb = time_emb.expand(batch_size, -1)
            
            # BEV特征融合（当前时间步）
            bev_attn = self.bev_attentions(hidden, bev_feats[:, t], None)

            # GRU更新状态
            gru_out = self.gru(bev_attn + time_emb, hidden)
            
            # Transformer解码
            decoder_out = self.transformer_decoder(
                gru_out.unsqueeze(0), 
                init_query.unsqueeze(0)
            ).squeeze(0)
            
            # 位置预测
            pred = self.loc_pred(decoder_out)
            preds.append(pred)
            
            # 自回归反馈
            hidden = decoder_out
        
        return torch.stack(preds, dim=1)  # [B, 6, 2]