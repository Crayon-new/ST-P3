import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

positional_encoding=dict(
    type='LearnedPositionalEncoding',
    num_feats=16,
    row_num_embed=200,
    col_num_embed=200,
    ),

class LearnedPosEmb(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats=16,
                 row_num_embed=200,
                 col_num_embed=200,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPosEmb, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, x, y):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)

        # pos = torch.cat(
        #     (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
        #         1, w, 1)),
        #     dim=-1).permute(2, 0,
        #                     1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return x_embed, y_embed

def discretize(trajs):
    '''
    trajs: torch.Tensor<float> (B, n_future, 2)   N: sample number
    '''
    B, n_future, _ = trajs.shape
    bx = torch.tensor([-49.75, -49.75], device=trajs.device)
    dx = torch.tensor([0.5, 0.5], device=trajs.device)
    bev_dimension = torch.tensor([200, 200], device=trajs.device)

    xx, yy = trajs[:,:,0], trajs[:,:,1]

    # discretize
    yi = ((yy - bx[0]) / dx[0]).long()
    yi = torch.clamp(yi,0, bev_dimension[0]-1)

    xi = ((xx - bx[1]) / dx[1]).long()
    xi = torch.clamp(xi, 0,bev_dimension[1]-1)

    return yi, xi

class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model=32, dk=32, dv=32, n_top=5):
        super().__init__()

        # 位置编码
        self.pos_emb = LearnedPosEmb()
        self.n_top = n_top
        # self.traj_pos = nn.Linear(2, d_model)

        self.d_k = dk
        self.d_v = dv
        
        self.W_Q = nn.Linear(d_model, self.d_k)
        # 不确定性图处理
 
        self.W_K = nn.Linear(1 + d_model, self.d_k)  # 假设U与PE拼接
        self.W_V = nn.Linear(1 + d_model, self.d_v)
        # 轨迹预测头
        self.planning_head = nn.Sequential(
            nn.Linear(n_top*(d_model + 2), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, traj, uncertainty):
        """
        Input:
        - traj: (B, 6, 2) 候选轨迹坐标(m)
        - seg: (B, 6, 200, 200) BEV分割概率
        - uncertainty: (B, 6, 200, 200) 预测不确定性
        Output:
        - optimized_traj: (B, 6, 2) 优化后轨迹
        """
        B, T, _ = traj.shape
        device = uncertainty.device
        traj = traj * torch.tensor([-1, 1], device=traj.device) # 转换为BEV坐标系
        gy, gx = discretize(traj) # 离散化到BEV网格

        gx_emb, gy_emb = self.pos_emb(gx, gy)
        traj_emb = torch.cat((gx_emb, gy_emb), dim=-1) # (B, 6, 32)
        Q = self.W_Q(traj_emb) # (B, 6, 32)

        # 对uncertainty 进行位置编码
        h, w = uncertainty.shape[-2:]
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        ux_emb, uy_emb = self.pos_emb(x, y)
        unc_pos_emb = torch.cat(
            (ux_emb.unsqueeze(0).repeat(h, 1, 1), uy_emb.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        unc_pos_emb = unc_pos_emb.unsqueeze(1).repeat(1, T, 1, 1, 1)

        unc_feat = torch.cat((uncertainty.view(B, T, h, w, 1), unc_pos_emb), dim=-1)

        K = self.W_K(unc_feat)  # (B,6,H,W,d_k)
        V = self.W_V(unc_feat)  # (B,6,H,W,d_v)

        # 4. 遍历每个时刻处理注意力
        optimized_traj = []
        for t in range(T):
            # 当前时刻的查询向量
            q = Q[:, t, :].unsqueeze(1)  # (B,1,d_k)
            
            # 当前时刻的Key/Value
            K_t = K[:, t].view(B, -1, self.d_k)  # (B,H*W,d_k)
            V_t = V[:, t].view(B, h, w, self.d_v)
            
            # 计算注意力分数
            scores = torch.bmm(q, K_t.transpose(1,2)) / (self.d_k**0.5)  # (B,1,H*W)
            scores = scores.squeeze(1)  # (B,H*W)
            attn_weights = F.softmax(scores, dim=-1)
            
            # 选择Top-n位置
            top_weights, top_indices = torch.topk(attn_weights, self.n_top, dim=-1)  # (B,n_top)
            h_idx = top_indices // w # (B,n_top)
            w_idx = top_indices % w   # (B,n_top)

            # 收集对应的Value和相对位置
            batch_idx = torch.arange(B, device=device)[:, None].expand(-1, self.n_top)
            v_selected = V_t[batch_idx, h_idx, w_idx]  # (B,n_top,d_v)

            # 加权
            v_selected = v_selected * top_weights.unsqueeze(-1)

            # 计算相对坐标
            curr_gx = gx[:, t]  # (B,)
            curr_gy = gy[:, t]  # (B,)
            rel_h = h_idx - curr_gx.unsqueeze(-1)  # (B,n_top)
            rel_w = w_idx - curr_gy.unsqueeze(-1)  # (B,n_top)
            rel_pos = torch.stack([rel_h, rel_w], dim=-1).float()  # (B,n_top,2)
            
            # 拼接特征并输入规划头
            combined = torch.cat([v_selected, rel_pos], dim=-1)  # (B,n_top,d_v+2)
            combined = combined.view(B, -1)  # (B,n_top*(d_v+2))
            delta = self.planning_head(combined)  # (B,2)

            optimized_traj.append(delta)
        
        # 合并各时刻结果
        optimized_traj = torch.stack(optimized_traj, dim=1)  # (B,6,2)
        # 可以加一个限制
        return traj + optimized_traj

if __name__ == '__main__':
    # importlib.import_module('stp3.models.transformer')
    importlib.import_module('mmdet.models.utils')
    traj = torch.randn(2, 6, 2) # B, T, 2
    uncertainty = torch.randn(2, 6, 1, 200, 200) # B, T, 1, 200, 200
    traj_finetuner = TrajectoryTransformer(32)
    optimized_traj = traj_finetuner(traj, uncertainty)
    # print(optimized_traj.shape)