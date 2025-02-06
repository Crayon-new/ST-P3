# Copyright (c) Zijian Huang. All rights reserved.
# ---------------------------------------------

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.cnn.bricks.transformer import build_positional_encoding
from .time_encoding import build_time_encoding
import numpy as np
import torch


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class FutureDecoder(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False,
                 positional_encoding=None,
                 time_encoding=None,
                 dataset_type='nuscenes', **kwargs):

        super(FutureDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        # self.time_encoding = build_time_encoding(time_encoding)
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                prev_bev,
                uncertainty,
                n_past,
                n_futures,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            prev_bev (Tensor): previous bev feature with shape
                `(bs, n_post(3), c, h, w)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].

        """

        bs = prev_bev.size(0)
        embed_dims = prev_bev.size(2)
        bev_h, bev_w = prev_bev.size()[-2:]
        present_bev = prev_bev[:, -1:].contiguous().squeeze(1)
        bev_mask = torch.zeros_like(present_bev)

        # bev_time = self.time_encoding(bev_mask)
        bev_pos = self.positional_encoding(bev_mask)

        bev_query = (present_bev.flatten(2, 3)).view(embed_dims*bs, -1).transpose(0, 1).view(-1, bs, embed_dims)  # (num_query, bs, embed_dims)
        # bev_query = torch.zeros([bev_h*bev_w, bs, embed_dims], device=present_bev.device)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        tempo_ref_2d = torch.stack([ref_2d.clone() for _ in range(n_past)], 1).reshape(bs*n_past, bev_h*bev_w, 1, 2)

        # key, value position embedding for cross attention
        prev_bev = prev_bev + bev_pos.unsqueeze(1)

        # key, value time embedding for cross attention
        # prev_bev = prev_bev + bev_time[:, :n_past, :, :]
        prev_bev = prev_bev.view(bs, n_past, embed_dims, -1)

        # batch first
        bev_pos = bev_pos.view(bs, embed_dims, -1).transpose(1, 2)
        bev_query = bev_query.transpose(0, 1)
        # bev_time = bev_time.view(bs, n_past+n_futures, embed_dims, -1)
        prev_bev = prev_bev.permute(0, 1, 3, 2).reshape(bs*n_past, bev_h * bev_w, embed_dims) # (bs*3, 40000, 64)

        intermediate = []
        # Autoregression
        for seq in range(n_futures):
            # time_embed = bev_time[:, n_past+seq, :, :].squeeze(1)
            # time_embed = time_embed.transpose(1, 2)
            # bev_query = bev_query + time_embed
            for lid, layer in enumerate(self.layers):
                output = layer(
                    bev_query,
                    prev_bev,
                    prev_bev,
                    uncertainty,
                    pos_embed=bev_pos,
                    ref_2d=ref_2d,
                    tempo_ref_2d=tempo_ref_2d,
                    bev_h=bev_h,
                    bev_w=bev_h,
                    **kwargs)
                bev_query = output
            intermediate.append(output)
        intermediate = torch.stack(intermediate, 1).view(bs, n_futures, bev_h, bev_w, embed_dims)
        intermediate = intermediate.permute(0, 1, 4, 2, 3)
        return intermediate
