"""
Source: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/max_vit.py

Modified to accept 3D inputs and handle them in a 2.5D fashion.
"""

from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from pytorch_med_imaging.networks.layers.MBConv import MBConv25d
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Attention25d(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, depth, window_height, window_width, window_depth, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y z w1 w2 w3 d -> (b x y z) (w1 w2 w3) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2 w3) d -> b w1 w2 w3 (h d)', w1 = window_height, w2 = window_width, w3 = window_depth)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y z) ... -> b x y z ...', x = height, y = width, z = depth)


class MaxViT(nn.Module):
    r"""

    Args:
        in_ch (int):
            Number of input channels.
        out_ch (int):
            Number of output channels.
        dim (int):
            Base dimensionality for the internal representations.
        depth (tuple of ints):
            Number of transformer blocks for each stage.
        dim_head (int, optional):
            Dimensionality of the transformer heads. Default is 32.
        dim_conv_stem (int, optional):
            Dimensionality of the convolutional stem. Default is None.
        window_size (int, optional):
            Window size for the efficient block and grid-like attention. Default is 7.
        mbconv_expansion_rate (int, optional):
            Expansion rate for the MBConv25d layer. Default is 4.
        mbconv_shrinkage_rate (float, optional):
            Shrinkage rate for the MBConv25d layer. Default is 0.25.
        dropout (float, optional):
            Dropout rate for the transformer layers. Default is 0.1.

    .. note::
        * The convolution kernels are modified to be 2.5D ([k, k] -> [k, k, 1])
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv3d(in_ch, dim_conv_stem, [3, 3, 1], stride = [2, 2, 1], padding = 0),
            nn.Conv3d(dim_conv_stem, dim_conv_stem, [3, 3, 1], padding = [1, 1, 0])
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block = nn.Sequential(
                    MBConv25d(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) (z w3)-> b x y z w1 w2 w3 d', w1 = w, w2 = w, w3 =1),  # block-like attention
                    PreNormResidual(layer_dim, Attention25d(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)'),

                    Rearrange('b d (w1 x) (w2 y) (z w3) -> b x y z w1 w2 w3 d', w1 = w, w2 = w, w3=1),  # grid-like attention
                    PreNormResidual(layer_dim, Attention25d(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)'),
                )

                self.layers.append(block)

        # mlp head out
        self.mlp_head = nn.Sequential(
            Reduce('b d h w z -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], in_ch)
        )

    def forward(self, x):
        x = self.conv_stem(x)

        for stage in self.layers:
            x = stage(x)

        return self.mlp_head(x)
#
if __name__ == '__main__':
    with torch.no_grad():
        net = MaxViT(1, 1, 64, (2, 2, 5, 2), window_size=5).cuda()
        test_input = torch.rand([2, 1, 320, 320, 21]).cuda()
        print(net(test_input).shape)