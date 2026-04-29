"""
This module contains the baseline MobileNetV1 architecture and all subsequent 
topological iterations tested during Stage 1, culminating in the optimized 
WideLinearMobileNet.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

# =============================================================================
# --- CORE / BASELINE ARCHITECTURE ---
# =============================================================================

class DepthSeparableConv2d(nn.Module):
    """
    Standard Depthwise Separable Convolution block for MobileNetV1.
    Splits spatial filtering (depthwise) and channel mixing (pointwise) 
    into two sequential steps to reduce computational complexity.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True) # Standard non-linear activation
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicConv2d(nn.Module):
    """Standard Convolution block with Batch Normalization and ReLU."""
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def mobilenet(alpha: float = 1.0, class_num: int = 100) -> nn.Module:
    """
    Builder function for the standard MobileNetV1 baseline.
    Used to establish the control metrics before optimization.
    """
    class MobileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
                DepthSeparableConv2d(int(32 * alpha), int(64 * alpha), 3, padding=1, bias=False)
            )

            # Downsample: spatial size /2
            self.conv1 = nn.Sequential(
                DepthSeparableConv2d(int(64 * alpha), int(128 * alpha), 3, stride=2, padding=1, bias=False),
                DepthSeparableConv2d(int(128 * alpha), int(128 * alpha), 3, padding=1, bias=False)
            )

            # Downsample: spatial size /2
            self.conv2 = nn.Sequential(
                DepthSeparableConv2d(int(128 * alpha), int(256 * alpha), 3, stride=2, padding=1, bias=False),
                DepthSeparableConv2d(int(256 * alpha), int(256 * alpha), 3, padding=1, bias=False)
            )

            # Downsample: spatial size /2
            self.conv3 = nn.Sequential(
                DepthSeparableConv2d(int(256 * alpha), int(512 * alpha), 3, stride=2, padding=1, bias=False),
                DepthSeparableConv2d(int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False),
                DepthSeparableConv2d(int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False),
                DepthSeparableConv2d(int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False),
                DepthSeparableConv2d(int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False),
                DepthSeparableConv2d(int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False)
            )

            # Downsample: spatial size /2
            self.conv4 = nn.Sequential(
                DepthSeparableConv2d(int(512 * alpha), int(1024 * alpha), 3, stride=2, padding=1, bias=False),
                DepthSeparableConv2d(int(1024 * alpha), int(1024 * alpha), 3, padding=1, bias=False)
            )

            self.fc = nn.Linear(int(1024 * alpha), class_num)

        def forward(self, x: Tensor) -> Tensor:
            x = self.stem(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return MobileNet()


# =============================================================================
# --- MODULAR ABLATION STUDY BLOCKS (Phase 2) ---
# =============================================================================

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA).
    Replaces fully connected SE blocks with a 1D convolution to preserve
    direct channel correspondence while providing dynamic feature weighting.
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """
    Mixes features between groups to break the isolation of depthwise layers.
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class BaselineModifiedConv2d(nn.Module):
    """
    Modular block used for Phase 2 ablation studies. 
    Allows toggling of Linear Bottlenecks, ECA, and Channel Shuffle 
    to isolate the performance impact of each mechanism.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, 
                 use_linear: bool = False, use_eca: bool = False, use_shuffle: bool = False, **kwargs):
        super().__init__()
        self.use_shuffle = use_shuffle
        
        # Standard Baseline Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True) 
        )
        
        # Toggle: ECA Attention
        self.eca = ECABlock(input_channels) if use_eca else nn.Identity()
        
        # Toggle: Linear Bottleneck (Prevents manifold collapse)
        if use_linear:
            self.pointwise = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1),
                nn.BatchNorm2d(output_channels)
                # Note: No ReLU here to preserve low-dimensional projection
            )
        else:
            self.pointwise = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True) # Standard Baseline ReLU
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.eca(x)
        x = self.pointwise(x)
        
        # Toggle: Channel Shuffle
        if self.use_shuffle:
            x = channel_shuffle(x, groups=2) 
            
        return x


def build_modified_baseline(use_linear: bool = False, use_eca: bool = False, use_shuffle: bool = False) -> nn.Module:
    """Dynamically replaces standard baseline blocks with ablation blocks."""
    model = mobilenet()
    for name, module in model.named_modules():
        if isinstance(module, DepthSeparableConv2d):
             in_ch = module.depthwise[0].in_channels
             out_ch = module.pointwise[0].out_channels
             k_size = module.depthwise[0].kernel_size
             stride = module.depthwise[0].stride
             padding = module.depthwise[0].padding
             
             new_block = BaselineModifiedConv2d(
                 in_ch, out_ch, k_size, 
                 stride=stride, padding=padding,
                 use_linear=use_linear, use_eca=use_eca, use_shuffle=use_shuffle
             )
             module.__class__ = BaselineModifiedConv2d
             module.__dict__.update(new_block.__dict__)
             
    return model


# =============================================================================
# --- ADVANCED TOPOLOGY ATTEMPT (Failed due to LR constraint) ---
# =============================================================================

class AdvancedResidualBlock(nn.Module):
    """
    Combines Linear Bottlenecks, SiLU Activation, and Residual Connections.
    Proved too complex to optimize stably with a strict 0.1 learning rate.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        
        self.use_res_connect = stride == 1 and input_channels == output_channels
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size, stride=stride, 
                      padding=padding, groups=input_channels, bias=bias),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(inplace=True) # Smoother loss landscape attempt
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=bias),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        
        if self.use_res_connect:
            return x + out
        return out


def build_advanced_v1() -> nn.Module:
    """Dynamically replaces blocks with advanced residual implementations."""
    model = mobilenet()
    for name, module in model.named_modules():
        if isinstance(module, DepthSeparableConv2d):
             in_ch = module.depthwise[0].in_channels
             out_ch = module.pointwise[0].out_channels
             k_size = module.depthwise[0].kernel_size
             stride = module.depthwise[0].stride
             padding = module.depthwise[0].padding
             
             new_block = AdvancedResidualBlock(
                 in_ch, out_ch, k_size, 
                 stride=stride, padding=padding, bias=False
             )
             module.__class__ = AdvancedResidualBlock
             module.__dict__.update(new_block.__dict__)
             
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        out_features = model.fc.out_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, out_features)
        )
             
    return model


# =============================================================================
# --- FINAL STAGE 1 MODEL (Peak Accuracy: 73.95%) ---
# =============================================================================

class WideLinearMobileNet(nn.Module):
    """
    The definitive optimal architecture for the Stage 1 constraints.
    Combines Linear Bottlenecks to preserve feature manifolds with a 
    Width Multiplier (alpha=2.0) to drastically expand parameter capacity.
    """
    def __init__(self, class_num: int = 100, alpha: float = 2.0):
        super().__init__()
        
        def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def linear_separable_conv(inp: int, oup: int, stride: int) -> nn.Sequential:
            """Depthwise Separable block with a Linear Bottleneck (No ReLU on pointwise)"""
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # Linear Bottleneck (No ReLU to prevent manifold collapse)
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

        self.model = nn.Sequential(
            conv_bn(3, int(32 * alpha), 1), 
            linear_separable_conv(int(32 * alpha), int(64 * alpha), 1),
            linear_separable_conv(int(64 * alpha), int(128 * alpha), 2),
            linear_separable_conv(int(128 * alpha), int(128 * alpha), 1),
            linear_separable_conv(int(128 * alpha), int(256 * alpha), 2),
            linear_separable_conv(int(256 * alpha), int(256 * alpha), 1),
            linear_separable_conv(int(256 * alpha), int(512 * alpha), 2),
            linear_separable_conv(int(512 * alpha), int(512 * alpha), 1),
            linear_separable_conv(int(512 * alpha), int(512 * alpha), 1),
            linear_separable_conv(int(512 * alpha), int(512 * alpha), 1),
            linear_separable_conv(int(512 * alpha), int(512 * alpha), 1),
            linear_separable_conv(int(512 * alpha), int(512 * alpha), 1),
            linear_separable_conv(int(512 * alpha), int(1024 * alpha), 2),
            linear_separable_conv(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Dropout added to prevent the widened network from overfitting
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(int(1024 * alpha), class_num)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x