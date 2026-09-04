import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from typing import *


class DenseBlock_down(nn.Module):
    """Implementation for SCDenseNet Denseblock.

    This denseblock is for the downwards transition. Note that number of layers
    were not explicitly state, but calculated from

    Input:  B x C1 x W x H x D
    Output: B x C2 x W x H x D

    ..note::
        From the original paper, the first encoder layer output is 192 x 192 x 16 x 48, so k_0 = 48, based on
        imprical trial, the number of layer/growth needs to be varied per layer. One should be fixed at 8, and
        the other should increase from 3 to 7. It's typical to have the same growth rate, so the number of layer
        is taken as the variable here. The output channel of each dense block is 72, 104, 144, 182, 248. The last
        dense transition seems to be special, as the output is 392. I am going to assume the growth rate maintained
        at 8 and the number of layers increased to 182

    Args:
        in_channels (int): Number of input channels.
        growth_rate (int): Number of channels to add per layer.
        num_layers (int): Number of convolutional layers in the block.
    """

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super(DenseBlock_down, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                    nn.BatchNorm3d(growth_rate),
                    nn.LeakyReLU(inplace=True)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dense block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W, D) where
                N is batch size, C is number of channels, H is height,
                W is width, and D is depth.

        Returns:
            torch.Tensor: Output tensor after passing through the dense block.
        """
        for layer in self.layers:
            out = layer(x)
            x = torch.cat((x, out), dim=1)  # Concatenate along the channel dimension
        return x

class Transitionblock_down_1(nn.Module):
    """Implementation for SCDenseNet: Transitionblock_down.

    This block reduces the number of channels and downsamples the input
    feature maps using max pooling. This transition block does not touch
    the third dimension (Depth).

    ..note::
        Type of pool was not specified in the original paper, Maxpool is
        used here because it's most typical.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after transition.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Transitionblock_down_1, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(kernel_size=[2, 2, 1], stride=[2, 2, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transition block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W, D) where
                N is batch size, C is number of channels, H is height,
                W is width, and D is depth.

        Returns:
            torch.Tensor: Output tensor after passing through the transition block.
        """
        x = self.conv(x)
        x = self.pool(x)
        return x

class Transitionblock_down_2(nn.Module):
    """Implementation for SCDenseNet: Transitionblock_down.

    This block reduces the number of channels and downsamples the input
    feature maps using max pooling. This transition block does not touch
    the third dimension (Depth).

    ..note::
        Type of pool was not specified in the original paper, Maxpool is
        used here because it's most typical.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after transition.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Transitionblock_down_2, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transition block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W, D) where
                N is batch size, C is number of channels, H is height,
                W is width, and D is depth.

        Returns:
            torch.Tensor: Output tensor after passing through the transition block.
        """
        x = self.conv(x)
        x = self.pool(x)
        return x


class Transitionblock_up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_D: Optional[bool]=True):
        super(Transitionblock_up, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.upsample_D = upsample_D


    def forward(self, x_shortcut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # print(f'{x.shape = }')
        # print(f'{x_shortcut.shape = }')
        if self.upsample_D:
            x = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=True)
        else:
            x = F.upsample(x, scale_factor=[2, 2, 1], mode='trilinear', align_corners=True)
            # Specifical situation where the final dimension is a single dimension
        # print(f'Upsampled {x.shape = }')
        # Check if the final dimension matches as single dimension can occur
        dim_diff = [s1 - s2 for s1, s2 in zip(x_shortcut.shape[-3:], x.shape[-3:])]
        if any(dim_diff):
            x = F.pad(x, [0, dim_diff[-1], 0, dim_diff[-2], 0, dim_diff[-3]])

        x = torch.concat([x_shortcut, x], dim=1) # concat at channel dimension
        x = self.conv(x)
        return x

class Transitionblock_last(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Transitionblock_last, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_shortcut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = F.upsample(x, scale_factor=[2, 2, 1], mode='trilinear', align_corners=True)
        x = torch.concat([x_shortcut, x], dim=1) # concat at channel dimension
        x = self.conv(x)
        return x


class SCDenseNet(nn.Module):
    def __init__(self):
        r"""Self-constrained DenseNet implemented w.r.t. Ke et al. 2020 Oral Oncology.

        Network structure:
            ----------------------------------------------------------------------------------------
                            Layer (type)               Output Shape         Param #     Tr. Param #
            ========================================================================================
                                Conv3d-1      [4, 48, 384, 384, 16]           1,344           1,344
                           BatchNorm3d-2      [4, 48, 384, 384, 16]              96              96
                             LeakyReLU-3      [4, 48, 384, 384, 16]               0               0
                             MaxPool3d-4      [4, 48, 192, 192, 16]               0               0
                       DenseBlock_down-5      [4, 72, 192, 192, 16]          36,360          36,360
                Transitionblock_down_1-6        [4, 72, 96, 96, 16]           5,256           5,256
                       DenseBlock_down-7       [4, 104, 96, 96, 16]          72,672          72,672
                Transitionblock_down_2-8        [4, 104, 48, 48, 8]          10,920          10,920
                       DenseBlock_down-9        [4, 144, 48, 48, 8]         129,720         129,720
               Transitionblock_down_1-10        [4, 144, 24, 24, 8]          20,880          20,880
                      DenseBlock_down-11        [4, 192, 24, 24, 8]         212,688         212,688
               Transitionblock_down_2-12        [4, 192, 12, 12, 4]          37,056          37,056
                      DenseBlock_down-13        [4, 248, 12, 12, 4]         326,760         326,760
               Transitionblock_down_1-14          [4, 248, 6, 6, 4]          61,752          61,752
                      DenseBlock_down-15          [4, 392, 6, 6, 4]       1,229,040       1,229,040
                    AdaptiveAvgPool3d-16          [4, 392, 1, 1, 1]               0               0
                            Rearrange-17                   [4, 392]               0               0
                               Linear-18                   [4, 512]         201,216         201,216
                            LayerNorm-19                   [4, 512]           1,024           1,024
                            LeakyReLU-20                   [4, 512]               0               0
                              Dropout-21                   [4, 512]               0               0
                               Linear-22                     [4, 1]             513             513
                              Sigmoid-23                     [4, 1]               0               0
                   Transitionblock_up-24        [4, 360, 12, 12, 4]         210,600         210,600
                   Transitionblock_up-25        [4, 288, 24, 24, 8]         145,440         145,440
                   Transitionblock_up-26        [4, 224, 48, 48, 8]          88,032          88,032
                   Transitionblock_up-27       [4, 168, 96, 96, 16]          49,896          49,896
                 Transitionblock_last-28     [4, 128, 192, 192, 16]         746,624         746,624
                               Conv3d-29       [4, 1, 192, 192, 16]             129             129
                             Upsample-30       [4, 1, 384, 384, 16]               0               0
            ========================================================================================

        """
        super(SCDenseNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv3d(1, 48, kernel_size=3, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2, 2, 1))
        )

        # Encoder
        self.down1     = DenseBlock_down(48 , 8, 3)
        self.down2     = DenseBlock_down(72 , 8, 4)
        self.down3     = DenseBlock_down(104, 8, 5)
        self.down4     = DenseBlock_down(144, 8, 6)
        self.down5     = DenseBlock_down(192, 8, 7)
        self.down_last = DenseBlock_down(248, 8, 18)

        self.down_trans1 = Transitionblock_down_1(72 , 72)
        self.down_trans2 = Transitionblock_down_2(104, 104)
        self.down_trans3 = Transitionblock_down_1(144, 144)
        self.down_trans4 = Transitionblock_down_2(192, 192)
        self.down_trans5 = Transitionblock_down_1(248, 248)

        # Decoder
        self.up_trans5 = Transitionblock_up(192 + 392 , 360 , False)
        self.up_trans4 = Transitionblock_up(144 + 360 , 288)
        self.up_trans3 = Transitionblock_up(104 + 288 , 224 , False)
        self.up_trans2 = Transitionblock_up(72 + 224  , 168)
        self.up_trans1 = Transitionblock_last(48 + 168, 128)

        # Classification output layers, not specified in original paper
        self.classification_layer = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Rearrange('b c 1 1 1 -> b c'),
            nn.Linear(392, 512),
            nn.LayerNorm(512, eps=1e-6),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Segmentation output layers
        self.segmentation_layer = nn.Sequential(
            nn.Conv3d(128, 1, kernel_size=1),
            nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True),
        )


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.in_conv(x)  # x0: 48 x 192 x 192 x 16
        # print(f"{x0.shape = }")

        x1 = self.down_trans1(self.down1(x0))  # x1: 72 x 96 x 96 x 16
        # print(f"{x1.shape = }")

        x2 = self.down_trans2(self.down2(x1))  # x2
        # print(f"{x2.shape = }")

        x3 = self.down_trans3(self.down3(x2))  # x3
        # print(f"{x3.shape = }")

        x4 = self.down_trans4(self.down4(x3))  # x4
        # print(f"{x4.shape = }")

        x5 = self.down_trans5(self.down5(x4))  # x5
        # print(f"{x5.shape = }")

        x6 = self.down_last(x5)  # x6 is bottle neck
        # print(f"{x6.shape = }")

        class_out = self.classification_layer(x6)
        # print(f"{class_out.shape = }")

        x = self.up_trans5(x4, x6)
        x = self.up_trans4(x3, x)
        x = self.up_trans3(x2, x)
        x = self.up_trans2(x1, x)
        x = self.up_trans1(x0, x)

        seg_out = self.segmentation_layer(x)
        # print(f"{seg_out.shape = }")
        return class_out, torch.sigmoid(seg_out) * class_out.view(-1, 1, 1, 1, 1).expand_as(seg_out)
