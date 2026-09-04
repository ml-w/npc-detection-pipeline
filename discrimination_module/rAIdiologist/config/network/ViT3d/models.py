"""
Code copied from https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch
Modified by alabamagan, redistributed under the MIT License.

Summary of modifications:
    * Add docstrings and comments
    * Fix bug owing to `grid` option
    * Implement 2.5d embedding that reduce the last dimension differently
    * Implement 3D images to prediction network
    * Implement RAN encoder

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
from .configs import *
from ..slicewise_ran import RAN_25D
from pytorch_med_imaging.networks.layers import ResidualBlock3d, Conv3d as Conv3d_pmi
from pytorch_med_imaging.networks.layers.StandardLayers3D import MaskedSequential3d
from ..attention_residual import AttentionModule_25d
from typing import Optional
from torch.distributions.normal import Normal
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return rearrange(x, 'b s h d -> b h s d')

    def forward(self, hidden_states):
        # Apply query, key, and value linear layers
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Reshape and permute query, key, and value layers for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores as dot product between query and key layers
        attention_scores = torch.matmul(query_layer, rearrange(key_layer, 'b h s d -> b h d s'))
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Compute attention probabilities using softmax
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        # Apply dropout to attention probabilities
        attention_probs = self.attn_dropout(attention_probs)

        # Compute context layer as dot product between attention probabilities and value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = rearrange(context_layer, 'b h s d -> b s h d').contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # Apply output linear layer
        attention_output = self.out(context_layer)
        # Apply dropout to attention output
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings_Img2Img(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings_Img2Img, self).__init__()
        self.config = config
        in_ch = config.in_ch
        down_25d = config.config_as_25d
        down_factor = config.down_factor
        patch_size = _triple(config.patches["size"])

        # Calculate the number of patches
        n_patches = int((img_size[0]/2**down_factor // patch_size[0]) * (img_size[1]/2**down_factor // patch_size[1]) *
                        (img_size[2]/2**down_factor // patch_size[2] if not down_25d else img_size[2] // patch_size[2]))

        # Initialize CNN encoder
        self.hybrid_model = CNNEncoder(config, n_channels=in_ch)
        in_channels = config['encoder_channels'][-1]

        # Initialize patch embeddings with 3D convolution
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        # Initialize position embeddings as learnable parameters！
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        # Initialize dropout layer
        self.dropout = Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        # Get CNN features
        x, features = self.hybrid_model(x)

        # Generate patch embeddings (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = self.patch_embeddings(x)

        # Flatten and transpose dimensions (B, n_patches, hidden)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        # Combine patch embeddings with position embeddings (B, n_patches, hidden)
        embeddings = x + self.position_embeddings

        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Return embeddings and features
        return embeddings, features


class Embeddings(nn.Module):
    """Construct the embeddings from patch and position embeddings.

    This class handles the creation of token embeddings for input volumes by either creating patch
    embeddings directly or using a hybrid approach that incorporates a prior convolutional encoder
    (e.g., RANEncoder). Position embeddings are added to the patch embeddings to retain positional
    information. A classification token is also concatenated to the sequence of embeddings.

    Args:
        config (PMIBaseCFG):
            An object containing the model configuration parameters.
        img_size (tuple):
            A tuple containing the dimensions (height, width, depth) of the input volumes.
        in_channels (int, optional):
            The number of input channels. Default is 3.

    Attributes:
        hybrid (bool):
            Indicates if the model uses a hybrid approach with a convolutional encoder.
        hybrid_model (nn.Module, optional):
            The convolutional encoder model used in the hybrid approach.
        patch_embeddings (nn.Conv3d):
            The 3D convolution layer used for creating patch embeddings.
        position_embeddings (nn.Parameter):
            The learnable position embeddings.
        cls_token (nn.Parameter):
            The learnable classification token.
        dropout (nn.Dropout):
            The dropout layer for regularization.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        in_ch = config.in_ch
        down_25d = config.config_as_25d
        down_factor = 3 # RAN down factor

        if hasattr(config.patches, "grid"):
            # Determine patch size based on grid size
            grid_size = list(_triple(config.patches["grid"]))
            if config.config_as_25d:
                # for 2.5d, the slice dimension is not down sampled by Down25d
                gz = img_size[2] // grid_size[2]
                grid_size[2] = img_size[2] // gz
            else:
                gz = img_size[2] // grid_size[2] // 2 ** (down_factor + 1) # +1 from in-conv of RAN
            patch_size = (img_size[0] // 2 ** (down_factor + 1) // grid_size[0],
                          img_size[1] // 2 ** (down_factor + 1) // grid_size[1],
                          gz)
            if any([ps == 0 for ps in patch_size]):
                msg = "Grid size setting error! Some dimensions are 0."
                raise RuntimeError(msg)
            n_patches = grid_size[0] * grid_size[1] * grid_size[2]
            self.hybrid = True
        else:
            # Determine grid size based on patch size
            patch_size = list(_triple(config.patches["size"]))
            grid_size = (img_size[0] // patch_size[0],
                         img_size[1] // patch_size[1],
                         img_size[2] // patch_size[2])
            n_patches = grid_size[0] * grid_size[1] * grid_size[2]
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = RANEncoder(config, n_channels=in_ch)
            in_ch = config['encoder_channels'][-1]
        # self.patch_embeddings = Conv3d(in_channels=in_ch,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        self.grid_size = grid_size
        self.patch_size = patch_size

        self.patch_embeddings = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (z p3) -> b (z h w) (p3 p1 p2 c)',
                      p1 = patch_size[0], p2 = patch_size[1], p3 = patch_size[2]),
            nn.LayerNorm(in_ch * patch_size[0] * patch_size[1] * patch_size[2]),
            nn.Linear(in_ch * patch_size[0] * patch_size[1] * patch_size[2], config.hidden_size)
        )
        # position embedding should also include the cls_token, note they need to be learnable
        # but it is initialized according to original BERT
        pos = torch.arange(0, n_patches + 1 + grid_size[2]).unsqueeze(1).float()
        pos_embedding = torch.zeros(1, n_patches + 1 + grid_size[2], config.hidden_size)
        div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * -(math.log(10000.0) / config.hidden_size))
        pos_embedding[..., 0::2] = torch.sin(pos * div_term)
        pos_embedding[..., 1::2] = torch.cos(pos * div_term)
        self.position_embeddings = nn.Parameter(pos_embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # This token separates the embedded patches from different slices.
        self.sep_token = nn.Parameter(torch.rand(1, 1, config.hidden_size))

        # Drop out before output
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        B = x.shape[0]
         # classification token. i.e., output prediction is triggered by this zero token
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x, _ = self.hybrid_model(x)
        x = self.patch_embeddings(x)

        # Insearch slice seperation tokens
        x = rearrange(x, 'b (n_slice n_patches_per_slice) h -> b n_slice n_patches_per_slice h',
                      n_slice = self.grid_size[2], n_patches_per_slice= self.grid_size[0] * self.grid_size[1])
        b, n_slice, n_patches_per_slice, h = x.shape
        sep_tokens = repeat(self.sep_token, '1 1 h -> b n_slice 1 h', b = b, n_slice = n_slice, h = h)
        x = torch.cat((x, sep_tokens), dim=2) # Attach to the end of patches
        x = rearrange(x, 'b ns np h -> b (ns np) h')
        # Classificaiton token is attached to the head
        x = torch.cat((cls_tokens, x), dim=1) # (B, n_patches+1, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, None # None is for catering `Transformer.forward()`

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        choice = ['img2img', 'img2pred']
        if config.type == 'img2img':
            embeddings = Embeddings_Img2Img
        elif config.type == 'img2pred':
            embeddings = Embeddings
        else:
            msg = f"Type must be one of {choice}. Got {config.type} instead."
            raise AttributeError(msg)

        self.embeddings = embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        """Decoder block for a 3D U-Net architecture.

        This class represents a single decoder block of a 3D U-Net, which
        upsamples the input tensor and performs two 3D convolutions with ReLU
        activations. Optional skip connections can also be provided.

        Attributes:
            conv1 (Conv3dReLU):
                First 3D convolution layer followed by ReLU activation.
            conv2 (Conv3dReLU):
                Second 3D convolution layer followed by ReLU activation.
            up (nn.Upsample):
                Upsampling layer with trilinear interpolation.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            skip_channels (int, optional):
                Number of additional channels from the skip connection. Defaults to 0.
            use_batchnorm (bool, optional):
                Whether to use batch normalization in the convolution layers. Defaults to True.
        """
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_factor = config.down_factor
        head_channels = config.conv_first_channel
        self.img_size = img_size
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        l, h, w = (self.img_size[0]//2**self.down_factor//self.patch_size[0]), (self.img_size[1]//2**self.down_factor//self.patch_size[1]), (self.img_size[2]//2**self.down_factor//self.patch_size[2])
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                #print(skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down25d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        """CNN Encoder module for generating hierarchical features.

        This class represents a 3D CNN encoder that generates hierarchical features
        from the input data. The encoder consists of an initial double convolution
        followed by a series of downsampling layers and additional max-pooling
        for deeper features.

        Attributes:
            n_channels (int):
                Number of input channels.
            down_num (int):
                Number of additional max-pooling layers.
            inc (DoubleConv
                ): Initial double convolution layer.
            down1 (Down):
                First downsampling layer.
            down2 (Down):
                Second downsampling layer.
            width (int):
                Width of the last encoder channel.

        Args:
            config (Config):
                Configuration object containing model parameters.
            n_channels (int, optional):
                Number of input channels. Defaults to 2.
        """
        super(CNNEncoder, self).__init__()
        self.down25d = config.config_as_25d
        down = Down if not self.down25d else Down25d

        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = DoubleConv(n_channels, encoder_channels[0])
        self.down1 = down(encoder_channels[0], encoder_channels[1])
        self.down2 = down(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]
    def forward(self, x):
        """Forward pass of the CNN encoder.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W, D), where
                B is the batch size, C is the number of channels, D is the depth,
                H is the height, and W is the width.

        Returns:
            torch.Tensor:
                Output tensor of shape (B, C_out, H, W, D),
                where C_out is the number of output channels.
            List[torch.Tensor]:
                List of hierarchical features generated by
                the CNN encoder, ordered from high to low resolution. They can
                be fed to the decoder through shortcuts for img2img usage.
        """
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        feats = self.down2(x2)
        features.append(feats)
        feats_down = feats
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)
            features.append(feats_down)
        return feats, features[::-1]


class RANEncoder(nn.Module):
    def __init__(self, config,
                 n_channels=1,
                 first_conv_ch: Optional[int] = 64):
        super(RANEncoder, self).__init__()
        if not config.config_as_25d:
            raise AttributeError("This encoder only supports 25d operations")
        self.n_channels = n_channels
        dropout = config.encoder_dropout_rate
        encoder_channels = config.encoder_channels

        self.in_conv1  = Conv3d_pmi(n_channels,
                                    encoder_channels[0],
                                    kern_size = [7, 7, 1], stride = [2, 2, 1], padding = [3, 3, 0])
        self.in_conv2  = ResidualBlock3d(encoder_channels[0]    , encoder_channels[1])
        self.att1      = AttentionModule_25d(encoder_channels[1], encoder_channels[1], stage = 0)
        self.att2      = AttentionModule_25d(encoder_channels[2], encoder_channels[2], stage = 1)
        self.att3      = AttentionModule_25d(encoder_channels[3], encoder_channels[3], stage = 2)
        self.r1        = ResidualBlock3d(encoder_channels[1], encoder_channels[2], p = dropout / 2.)
        self.r2        = ResidualBlock3d(encoder_channels[2], encoder_channels[3], p = dropout / 1.)
        self.out_conv  = ResidualBlock3d(encoder_channels[3], encoder_channels[4], p = dropout)
        self.out_res   = nn.Sequential(*[ResidualBlock3d(encoder_channels[4], encoder_channels[4], p = dropout)
                                         for i in range(2)])

    def forward(self,
                x: torch.Tensor):
        r"""Expect input :math:`(B × in_{ch} × H × W × S)`, output (B × out_ch)

        Args:
            x (torch.Tensor):
                This should be a float tensor input with shape :math:`(B × in_{ch} × H × W × S)`.

        """
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)

        # x = nnf.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])

        # Resume dimension
        x = self.in_conv2(x)
        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)
        x = self.out_conv(x)
        x = self.out_res(x)
        return x, None

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class ViTVNetImg2Img(nn.Module):
    def __init__(self, config, img_size=(64, 256, 256), int_steps=7, vis=False):
        super(ViTVNetImg2Img, self).__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config, img_size)
        self.reg_head = RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config['n_dims'],
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(img_size)
        self.config = config
        #self.integrate = VecInt(img_size, int_steps)
    def forward(self, x):

        source = x[:,0:1,:,:]

        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        flow = self.reg_head(x)
        #flow = self.integrate(flow)
        out = self.spatial_trans(source, flow)
        return out, flow


class ViTVNetImg2Pred(nn.Module):
    def __init__(self, config, num_classes=1, img_size=(64, 256, 256), int_steps=7, vis=False):
        super(ViTVNetImg2Pred, self).__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, num_classes)
        )

        self.config = config
        self.vis = vis
        self.num_classes = num_classes
        #self.integrate = VecInt(img_size, int_steps)
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x, attention, _ = self.transformer(x)  # (B, n_patch, hidden)
        x = self.head(x[:, 0])
        x = x.view(B, self.num_classes)
        if self.vis:
            return x, attention
        else:
            return x

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

CONFIGS = {
    'ViT3d-V-Net': get_3DReg_config(),
    'ViT3d-Img2Pred': get_3DImg2Pred_config()
}

