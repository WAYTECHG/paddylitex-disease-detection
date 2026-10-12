"""Lightweight Gated Dense-IDGT backbone for Ultralytics YOLOv8.

Place this file at:
    ultralytics/nn/modules/hybrid_densenet_idgt.py

Then expose HybridDenseNetIDGTBackbone in:
    ultralytics/nn/modules/__init__.py
    ultralytics/nn/tasks.py
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Lightweight squeeze-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


class HybridDenseLayer(nn.Module):
    """DenseNet bottleneck layer with optional grouped 3x3 convolution."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 2,
        drop_rate: float = 0.0,
        use_grouped: bool = True,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, inter_channels, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)

        groups = 1
        if use_grouped:
            # Conservative grouping to avoid invalid channel combinations.
            if inter_channels % 2 == 0 and growth_rate % 2 == 0:
                groups = 2

        self.conv2 = nn.Conv2d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class GatedDenseBlock(nn.Module):
    """Dense block with optional periodic channel attention on newly generated features."""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        gate_every: int = 3,
        use_grouped: bool = True,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleDict()
        self.use_gate = use_gate
        self.gate_indices = set(range(gate_every - 1, num_layers, gate_every)) if gate_every > 0 else set()

        for i in range(num_layers):
            self.layers.append(
                HybridDenseLayer(
                    num_input_features=num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    use_grouped=use_grouped,
                )
            )
            if use_gate and i in self.gate_indices:
                self.gates[str(i)] = ChannelAttention(growth_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for i, layer in enumerate(self.layers):
            new_feature = layer(torch.cat(features, dim=1))
            # Gate the newly generated feature before it is accumulated.
            if self.use_gate and i in self.gate_indices:
                new_feature = self.gates[str(i)](new_feature)
            features.append(new_feature)
        return torch.cat(features, dim=1)


class StandardTransition(nn.Module):
    """Standard DenseNet-style transition: BN-ReLU-1x1Conv-AvgPool."""

    def __init__(self, num_input_features: int, min_channels: int = 32) -> None:
        super().__init__()
        out_channels = max(num_input_features // 2, min_channels)
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IDGTTransition(nn.Module):
    """Integrated Dense-Ghost Transition layer.

    The layer compresses dense features first, applies channel attention in the
    compressed space, generates cheap ghost-style features, fuses them, and then
    downsamples by average pooling.
    """

    def __init__(self, num_input_features: int, compression_divisor: int = 4, min_bottleneck: int = 32) -> None:
        super().__init__()
        bottleneck = max(num_input_features // compression_divisor, min_bottleneck)
        self.bottleneck = bottleneck
        self.out_channels = bottleneck * 2

        self.pre = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bottleneck, kernel_size=1, bias=False),
        )
        self.attention = ChannelAttention(bottleneck, reduction=4)
        self.cheap = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, groups=bottleneck, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, bottleneck * 2, kernel_size=1, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(bottleneck * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck * 3, self.out_channels, kernel_size=1, bias=False),
        )
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.attention(x)
        ghost = self.cheap(x)
        x = self.fuse(torch.cat([x, ghost], dim=1))
        return self.downsample(x)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class HybridDenseNetIDGTBackbone(nn.Module):
    """Lightweight DenseNet-IDGT backbone that returns P3, P4, and P5 feature maps.

    Args order for YOLO YAML:
        [growth_rate, block_config, bn_size, drop_rate, gate_every,
         p3_channels, p4_channels, p5_channels,
         use_grouped, use_gate, use_idgt, compression_divisor]
    """

    def __init__(
        self,
        growth_rate: int = 24,
        block_config: Sequence[int] = (6, 8, 16, 8),
        bn_size: int = 2,
        drop_rate: float = 0.0,
        gate_every: int = 3,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,
        use_grouped: bool = True,
        use_gate: bool = True,
        use_idgt: bool = True,
        compression_divisor: int = 4,
    ) -> None:
        super().__init__()
        if len(block_config) != 4:
            raise ValueError("block_config must contain four integers, e.g. [6, 8, 16, 8]")

        self.out_channels = [p3_channels, p4_channels, p5_channels]
        self.use_grouped = use_grouped
        self.use_gate = use_gate
        self.use_idgt = use_idgt

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1, output stride 8 after transition.
        c0 = 64
        self.block1 = GatedDenseBlock(block_config[0], c0, bn_size, growth_rate, drop_rate, gate_every, use_grouped, use_gate)
        c1 = c0 + block_config[0] * growth_rate
        self.trans1 = self._make_transition(c1, use_idgt, compression_divisor)
        c1t = self.trans1.out_channels
        self.p3_proj = nn.Conv2d(c1t, p3_channels, kernel_size=1, bias=False)

        # Stage 2, output stride 16 after transition.
        self.block2 = GatedDenseBlock(block_config[1], c1t, bn_size, growth_rate, drop_rate, gate_every, use_grouped, use_gate)
        c2 = c1t + block_config[1] * growth_rate
        self.trans2 = self._make_transition(c2, use_idgt, compression_divisor)
        c2t = self.trans2.out_channels
        self.p4_proj = nn.Conv2d(c2t, p4_channels, kernel_size=1, bias=False)

        # Stage 3, output stride 32 after transition.
        self.block3 = GatedDenseBlock(block_config[2], c2t, bn_size, growth_rate, drop_rate, gate_every, use_grouped, use_gate)
        c3 = c2t + block_config[2] * growth_rate
        self.trans3 = self._make_transition(c3, use_idgt, compression_divisor)
        c3t = self.trans3.out_channels

        # Stage 4, keep stride 32.
        self.block4 = GatedDenseBlock(block_config[3], c3t, bn_size, growth_rate, drop_rate, gate_every, use_grouped, use_gate)
        c4 = c3t + block_config[3] * growth_rate
        self.norm_final = nn.BatchNorm2d(c4)
        self.relu_final = nn.ReLU(inplace=True)
        self.p5_proj = nn.Conv2d(c4, p5_channels, kernel_size=1, bias=False)

        self.apply(_init_weights)

    @staticmethod
    def _make_transition(num_input_features: int, use_idgt: bool, compression_divisor: int) -> nn.Module:
        if use_idgt:
            return IDGTTransition(num_input_features, compression_divisor=compression_divisor, min_bottleneck=32)
        return StandardTransition(num_input_features, min_channels=32)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)       # stride 4

        x = self.block1(x)
        x = self.trans1(x)     # stride 8
        p3 = self.p3_proj(x)

        x = self.block2(x)
        x = self.trans2(x)     # stride 16
        p4 = self.p4_proj(x)

        x = self.block3(x)
        x = self.trans3(x)     # stride 32
        x = self.block4(x)
        x = self.relu_final(self.norm_final(x))
        p5 = self.p5_proj(x)

        return [p3, p4, p5]
