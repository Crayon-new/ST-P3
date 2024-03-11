# Copyright (c) Zijian Huang. All rights reserved.
# ---------------------------------------------

_dim_ = 64
_pos_dim_ = _dim_//2
_time_dim = _dim_
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
n_past = 3
n_future = 4
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

decoder = dict(
    type='FutureDecoder',
    num_layers=6,
    pc_range=point_cloud_range,
    num_points_in_pillar=4,
    return_intermediate=False,
    positional_encoding=dict(
        type='LearnedPositionalEncoding',
        num_feats=_pos_dim_,
        row_num_embed=bev_h_,
        col_num_embed=bev_w_,
    ),
    time_encoding=dict(
        type='LearnedTimeEncoding',
        time_seq_len=n_past+n_future,
        num_feats=_time_dim,
    ),
    transformerlayers=dict(
        type='CustomTransformerLayer',
        attn_cfgs=[
            dict(
                type='CustomSelfAttention',
                embed_dims=64,
                num_heads=8,
                dropout=0.1,
            ),
            dict(
                type='TemporalCrossAttention',
                embed_dims=64,
                num_heads=8,
                dropout=0.1,
            ),
        ],
        ffn_cfgs=dict(
             type='FFN',
             embed_dims=64,
             feedforward_channels=256,
             num_fcs=2,
             ffn_drop=0.1,
             act_cfg=dict(type='ReLU', inplace=True),
             ),
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm')))
