import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.utils import Registry, build_from_cfg


TIME_ENCODING = Registry('time encoding')


def build_time_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, TIME_ENCODING, default_args)


@TIME_ENCODING.register_module()
class LearnedTimeEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        time_seq_len (int): The length of time sequence that need to be embedded. (len_past + len+future).
        num_feats (int, optional): The embedding feature channel number.
            Default 128.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 time_seq_len,
                 num_feats,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedTimeEncoding, self).__init__(init_cfg)
        self.time_seq_len = time_seq_len
        self.time_embed = nn.Embedding(time_seq_len, num_feats)
        self.num_feats = num_feats

    def forward(self, mask):
        """Forward function for `LearnedTimeEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            time (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        bs = mask.shape[0]
        time_embed = self.time_embed(torch.arange(self.time_seq_len, device=mask.device))
        time = time_embed.repeat(bs, h, w, 1, 1).permute(0, 3, 4, 1, 2)
        return time

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'num_time_seq={self.time_seq_len}, '
        return repr_str


# test time encoding
if __name__ == '__main__':
    time_encoding = LearnedTimeEncoding(num_time_seq=4, num_feats=64)
    mask = torch.zeros(10, 200, 200)
    time = time_encoding(mask)
    print(time.shape)