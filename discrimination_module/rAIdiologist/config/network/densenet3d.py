import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                               kernel_size=[3,3,1], stride=1, padding=[1,1,0], bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        new_features = self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(x))))))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        # Each _DenseLayer already concatenates its input with newly produced features.
        # We only need to pass the tensor sequentially through layers and return the final tensor.
        for layer in self.children():
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features,
                                kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=[2,2,1], stride=[2,2,1])

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseNet3D(nn.Module):
    """
    3D DenseNet implementation for medical image classification.
    
    Args:
        growth_rate (int): How many filters to add each layer (k in paper)
        block_config (tuple): How many layers in each pooling block
        num_init_features (int): The number of filters to learn in the first convolution layer
        bn_size (int): Multiplicative factor for number of bottle neck layers
        drop_rate (float): Dropout rate after each dense layer
        num_classes (int): Number of classification classes
        in_channels (int): Number of input channels
    """
    
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        num_classes: int = 1,
        in_channels: int = 1
    ):
        super(DenseNet3D, self).__init__()
        
        # First convolution
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=[7,7,3], stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[3,3,1], stride=[2,2,1], padding=[1,1,0])
        )
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        # Linear layer
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )
        
        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
        # For compatibility
        self.register_buffer('_mode', torch.IntTensor([0]))
    
    def forward(self, x):
        # Unsqueeze during inference
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        features = self.features(x)
        out = self.classifier(features)
        return out


def get_densenet3d(model_depth: str = '121', **kwargs):
    """
    Factory function to create DenseNet3D models of different depths.
    
    Args:
        model_depth (str): One of '121', '169', '201', or '264'
        **kwargs: Additional arguments to pass to DenseNet3D
        
    Returns:
        DenseNet3D: The requested model
    """
    if model_depth == '121':
        model = DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    elif model_depth == '169':
        model = DenseNet3D(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    elif model_depth == '201':
        model = DenseNet3D(growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    elif model_depth == '264':
        model = DenseNet3D(growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)
    else:
        raise ValueError(f"Unsupported DenseNet depth: {model_depth}. Choose from '121', '169', '201', or '264'.")
    
    return model


def get_densenet3d_121():
    """Returns a DenseNet3D-121 model with default parameters for medical imaging."""
    m = get_densenet3d('121')
    m.set_mode = lambda x: 0  # Does nothing
    return m