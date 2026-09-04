import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from  pytorch_med_imaging.networks.layers import ResidualBlock3d, DoubleConv3d, Conv3d
from pytorch_med_imaging.networks.layers.StandardLayers3D import MaskedSequential3d
from .attention_residual import AttentionModule_25d, AttentionModule_25d_recur
from .spacecutter import LogisticCumulativeLink
import pprint



class RAN_25D(nn.Module):
    r"""This is a 2.5D network that will process the image volume and give a prediction of specified channel.

    This is a modification of the Residual attention network, which was originally developed for 2D classification. In
    this network, the input convolution is modified to have a 3D kernel with a size of 7×7×3 size. The stride is set
    to 2×2×1 to also perform downsample, saving some of the memories. The rest of the residual blocks all have 3d
    kernels.

    The attention modules are configured to be mostly 3D, with only the softmax branch set to have a kernel size of
    3×3×1. For the residual blocks within the attention modules, the kernels are still 3D.

    The dropouts were configured to increase with depth, this prevents the propagation of zeros to easily disable the
    training of the whole network.

    Args:
        in_ch (int):
            Number of in channels.
        out_ch (int):
            Number of out channels.
        first_conv_ch (int, Optional):
            Number of output channels of the first conv layer.
        save_mask (bool, Optional):
            If `True`, the attention mask of the attention modules will be saved to the CPU memory.
            Default to `False`.
        save_weight (bool, Optional):
            If `True`, the slice attention would be saved for use (CPU memory). Default to `False`.
        exclude_fc (bool, Optional):
            If `True`, the output FC layer would be excluded.
        dropout (float, Optional):
            Dropouts for residual blocks. Default to 0.1.
        return_top (bool, Optional):
            If `True`, also return the fc_output of the slicewise prediction. See :method:`forward_top` for more.

    """
    _strats_dict = {
        'max': 0,
        'min': 1,
        'mean': 2,
        'avg': 2
    }
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_ch: Optional[int] = 64,
                 exclude_fc: Optional[bool] = False,
                 sigmoid_out: Optional[bool] = False,
                 reduce_strats: Optional[str] = 'max',
                 dropout: Optional[float] = 0.1,
                 return_top: Optional[bool] = False) -> None:
        super(RAN_25D, self).__init__()

        self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[7, 7, 3], stride=[2, 2, 1], padding=[3, 3, 1])
        self.exclude_top = exclude_fc # Normally you don't have to use this.
        self.return_top = return_top # Normally you don't have to use this
        self.sigmoid_out = sigmoid_out
        if return_top and exclude_fc:
            assert not self.exclude_top, "Cannot return top when exclude_top is True."

        # RAN
        # Note: using the same dropout rate might cause the network unable to converge because the zeros from
        #       high level dropouts propagates to low level layers, which also does dropout. If the drop out
        #       is too high, there will be too much zeros in the low level layers.
        self.in_conv2  = ResidualBlock3d(first_conv_ch, 256, p = dropout)
        self.att1      = AttentionModule_25d(256      , 256 , stage = 0      )
        self.r1        = ResidualBlock3d(256          , 512 , p     = dropout / 4.)
        self.att2      = AttentionModule_25d(512      , 512 , stage = 1      )
        self.r2        = ResidualBlock3d(512          , 1024, p     = dropout / 4.)
        self.att3      = AttentionModule_25d(1024     , 1024, stage = 2      )
        self.out_conv1 = ResidualBlock3d(1024         , 2048, p     = dropout / 2. )
        self.out_conv2 = nn.Sequential(*([ResidualBlock3d(2048, 2048, p = dropout) for i in range(2)]))

        # Output layer
        self.out_bn = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )
        self.out_fc1 = nn.Linear(2048, out_ch)

        # Reduce strategy
        if reduce_strats not in self._strats_dict:
            raise AttributeError("Reduce strategy is incorrect.")
        # self.register_buffer('reduce_strats', torch.Tensor([self._strats_dict[reduce_strats]]))
        self.reduce_strats = self._strats_dict[reduce_strats]

    @staticmethod
    def get_nonzero_slices(x: torch.Tensor):
        r"""This method computes the nonzero slices in the input. Assumes input shape follows the convention of
        :math:`(B × C × H × W × S)` and that the padding is done only for dimension :math:`S`.

        Args:
            x (torch.Tensor):
                Tensor with shape :math:`(B × C × H × W × S)`.

        Returns:
            dict: ``nonzero_slices`` - Keys are mini-batch index and values are the bottom and top slices pair
            list: ``top_slices`` - List of top index of non-zero slices.
        """
        # compute non-zero slices from seg if it is not None
        _tmp = x.detach().cpu()
        sum_slice = _tmp.sum(dim=[1, 2, 3])  # (B x S)
        # Raise error if everything is zero in any of the minibatch
        if 0 in list(sum_slice.sum(dim=[1])):
            msg = f"An item in the mini-batch is completely empty or have no segmentation:\n" \
                  f"{sum_slice.sum(dim=[1])}"
            raise ArithmeticError(msg)
        padding_mask = sum_slice != 0  # (B x S)
        where_non0 = torch.argwhere(padding_mask)
        nonzero_slices = {i.cpu().item(): (where_non0[where_non0[:, 0] == i][:, 1].min().cpu().item(),
                                           where_non0[where_non0[:, 0] == i][:, 1].max().cpu().item()) for i in
                          where_non0[:, 0]}
        # Calculate the loading bounds
        added_radius = 0
        bot_slices = [max(0, nonzero_slices[i][0] - added_radius) for i in range(len(nonzero_slices))]
        top_slices = [min(x.shape[-1], nonzero_slices[i][1] + added_radius) for i in range(len(nonzero_slices))]
        return nonzero_slices, top_slices

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
        nonzero_slice, _ = self.get_nonzero_slices(x)

        x = self.in_conv1(x)

        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])

        # Resume dimension
        x = self.in_conv2(x)
        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)
        x = self.out_conv1(x)
        x = self.out_conv2(x)

        # order of slicewise attention and max pool makes no differences because pooling is within slice
        x = F.adaptive_max_pool3d(x, [1, 1, None]).squeeze() # x: (B x C x S)
        if x.dim() < 3:
            x = x.unsqueeze(0)

        # If `exclude_top` is set to True, directly return the reduced slicewise prediction.
        if not self.exclude_top:
            if self.return_top:
                # Copy the slicewie output (B x C x S)
                sw_prediction = x
            # Fed features through final output FC
            x = self.forward_top(B, nonzero_slice, x)

        if self.sigmoid_out:
            x = torch.sigmoid(x)
        if not self.return_top:
            return x
        else:
            return x, sw_prediction

    def forward_top(self, B, nonzero_slice, x) -> torch.Tensor:
        r"""For regular forward, this is the final layer of output, where the extracted features are fed through the
        output FC layer. If :attr:`excluded_top` is set to `True`, this function is skipped, and the output from sw
        layer is directly returned.

        Args:
            B (int):
                Mini-batch size
            nonzero_slice (dict):
                Non-zero slice got from :func:`get_nonzero_slices`
            x (torch.Tensor):
                Tensor.

        Returns:
            torch.Tensor

        """
        # expect input x: (B x C x S)
        seq_len = [nonzero_slice[n][1] for n in range(len(nonzero_slice))]
        # x = self.out_bn(x).permute(0, 2, 1).contiguous() # x: (B x S x C)
        # out_bn is affecting the performance. Differences between slices should not be reduced. Maybe can change to
        # group norm later.
        x = x.permute(0, 2, 1).contiguous()

        if self.reduce_strats == 0:
            x = torch.concat([x[i, a + 1:b].max(dim=0, keepdim=True).values for i, (a, b) in nonzero_slice.items()])
        elif self.reduce_strats == 1:
            x = torch.concat([x[i, a + 1:b].min(dim=0, keepdim=True).values for i, (a, b) in nonzero_slice.items()])
        elif self.reduce_strats == 2:
            x = torch.concat([x[i, a + 1:b].mean(dim=0, keepdim=True) for i, (a, b) in nonzero_slice.items()])
        elif self.reduce_strates == 3:
            pass

        x = self.out_fc1(x)
        # Get best prediction across the slices, note that first conv layer is [3 × 3 × 3] that eats the first and
        # the last slice, so we are not going to take them into account. nonzero_slice is index, so no need to -1
        if x.dim() < 2:
            x.view(B, -1)
        return x


class SlicewiseAttentionRAN(RAN_25D):
    r"""This modification adds a branch that computes an attention weight of for each input slice. The attention
    weights are then used to weight the deep features reducing the extracted features by slices using MaxPool.

    .. note::
        - Different to original RAN_25, this version also skipped `out_conv2`, which is a sequential residual blocks.
        - Here's a list of major differences between this implementation and the original old SWRAN:
            1. `self.out_conv1` kernel size and stride is different. `kern_size=[3, 3, 1] -> [7, 7, 3]` [Note: I change
                it back becuase its better]
            2. The old SWRAN uses axis permutation to prepare tensor for multiplication with `in_sw` weights, but
               in this implementation we use einops for better interpretability.


    Attributes:
        in_ch (int):
            Number of in channels.
        out_ch (int):
            Number of out channels.
        first_conv_ch (int, Optional):
            Number of output channels of the first conv layer.
        save_mask (bool, Optional):
            If `True`, the attention mask of the attention modules will be saved to the CPU memory.
            Default to `False`.
        save_weight (bool, Optional):
            If `True`, the slice attention would be saved for use (CPU memory). Default to `False`.
        exclude_fc (bool, Optional):
            If `True`, the output FC layer would be excluded.
        dropout (float, Optional):
            Drop out for residual blocks. Default to 0.1.
        return_top (bool, Optional):
            If `True`, this will returns also the slicewise prediction. Default to `False`.

    """
    _strats_dict = {
        'max': 0,
        'min': 1,
        'mean': 2,
        'avg': 2,
        'attn': 3,
    }
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_ch: Optional[int] = 64,
                 save_mask: Optional[bool] = False,
                 save_weight: Optional[bool] = False,
                 exclude_fc: Optional[bool] = False,
                 sigmoid_out: Optional[bool] = False,
                 reduce_strats: Optional[str] = 'max',
                 dropout: Optional[float] = 0.1,
                 return_top: Optional[bool] = False):

        super(SlicewiseAttentionRAN, self).__init__(in_ch=in_ch, out_ch=out_ch, first_conv_ch=first_conv_ch,
                                                    exclude_fc=exclude_fc, sigmoid_out=sigmoid_out,
                                                    reduce_strats=reduce_strats, dropout=dropout,
                                                    return_top=return_top)

        self.save_weight=save_weight
        # self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0])
        self.exclude_top = exclude_fc # Normally you don't have to use this.
        self.sigmoid_out = sigmoid_out

        # Slicewise attention layer
        self.in_sw = nn.Sequential(
            nn.MaxPool3d([2, 2, 1]),
            DoubleConv3d(int(first_conv_ch),
                         int(first_conv_ch * 2),
                         kern_size=[3, 3, 1], padding=0, dropout=0.1, activation='leaky_relu'),
            nn.MaxPool3d([2, 2, 1]),
            DoubleConv3d(int(first_conv_ch * 2), 1, kern_size=1, padding=0, dropout=0.1, activation='leaky_relu'),
            nn.AdaptiveAvgPool3d([1, 1, None])
        )
        self.x_w = None

    def forward(self, x):
        r"""Expect input (B x in_ch x H x W x S), output (B x out_ch)"""
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slice, _ = self.get_nonzero_slices(x)

        x = self.in_conv1(x)

        # Construct slice weight
        x_w = self.in_sw(x).view(B, -1)
        if self.save_weight:
            self.x_w = x_w.data.cpu()

        # Resume dimension
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        # * Apply slicewise weight before final conv
        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])
        x = x * einops.rearrange(x_w, 'b sc -> b 1 1 1 sc').expand_as(x)

        x = self.out_conv1(x)
        # order of slicewise attention and max pool makes no differences because pooling is within slice
        # x = x = x * x_w.view([x.shape[0], 1, 1, 1, -1]).expand_as(x)
        x = F.adaptive_max_pool3d(x, [1, 1, None]).squeeze()
        # else:
        # x = F.adaptive_avg_pool3d(x, [1, 1, None]).squeeze()

        if x.dim() < 3:
            x = x.unsqueeze(0)

        if not self.exclude_top:
            if self.return_top:
                sw_prediction = x
            x = self.forward_top(B, nonzero_slice, x)
        if self.sigmoid_out:
            x = torch.sigmoid(x)
        if self.return_top:
            return x, sw_prediction
        else:
            return x

    def get_mask(self):
        #[[B,H,W,D],[B,H,W,D],[B,H,W,]]
        return [r.get_mask() for r in [self.att1, self.att2, self.att3]]

    def get_slice_attention(self):
        if not self.x_w is None:
            while self.x_w.dim() < 2:
                self.x_w = self.x_w.unsqueeze(0)
            return self.x_w
        else:
            print("Attention weight was not saved!")
            return None


class SWRAN_Block(SlicewiseAttentionRAN):
    def __init__(self, *args, grid_size=[4, 4], **kwargs):
        r"""
        Args:
            grid_size (list of int):
                The grid size for adaptive maxpool. Performance would be better if the original
                image dimension is divisible by the specified grid_size.

        See Also:
              :class:`SlicewiseAttentionRAN`

        """
        super().__init__(*args, **kwargs)
        in_ch = args[0]
        first_conv_ch = kwargs.get('first_conv_ch', 64)
        self.pre_top_maxpool = nn.AdaptiveMaxPool3d((*grid_size, None))
        self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0])


    def forward(self, x):
        r"""Expect input (B x in_ch x H x W x S), output (B x out_ch)"""
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slice, _ = self.get_nonzero_slices(x)

        x = self.in_conv1(x)

        # Construct slice weight
        x_w = self.in_sw(x).view(B, -1)
        if self.save_weight:
            self.x_w = x_w.data.cpu()

        # Resume dimension
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        # * Apply slicewise weight before final conv
        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])
        x = x * einops.rearrange(x_w, 'b sc -> b 1 1 1 sc').expand_as(x)

        x = self.out_conv1(x)
        x = self.pre_top_maxpool(x) # this is for designed for ViT

        if x.dim() < 3:
            x = x.unsqueeze(0)

        if not self.exclude_top:
            if self.return_top:
                sw_prediction = x
            x = self.forward_top(B, nonzero_slice, x)
        if self.sigmoid_out:
            x = torch.sigmoid(x)
        if self.return_top:
            return x, sw_prediction
        else:
            return x

    def forward_top(self, B, nonzero_slice, x) -> torch.Tensor:
        # expect input x: (B x C x S)
        x = F.adaptive_max_pool3d(x, [1, 1, None]).squeeze(2,3)
        return super().forward_top(B, nonzero_slice, x)


class SWRAN_Block_Focused(SWRAN_Block):
    def __init__(self, *args, **kwargs):
        r"""This focused network is used to zoom in the primary tumor. It performs a more focused inspection instead of
        reading the entire image.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x):
        r"""Expect input (B x in_ch x H x W x S), output (B x out_ch)"""
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slice, _ = self.get_nonzero_slices(x)

        x = self.in_conv1(x)

        # Construct slice weight
        x_w = self.in_sw(x).view(B, -1)
        if self.save_weight:
            self.x_w = x_w.data.cpu()

        # Resume dimension
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        # * Apply slicewise weight before final conv
        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])
        x = x * einops.rearrange(x_w, 'b sc -> b 1 1 1 sc').expand_as(x)

        x = self.out_conv1(x)
        x = self.pre_top_maxpool(x) # this is for designed for ViT

        if x.dim() < 3:
            x = x.unsqueeze(0)

        if not self.exclude_top:
            if self.return_top:
                sw_prediction = x
            x = self.forward_top(B, nonzero_slice, x)
        if self.sigmoid_out:
            x = torch.sigmoid(x)
        if self.return_top:
            return x, sw_prediction
        else:
            return x
