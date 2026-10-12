"""Shuffle-DenseNet backbone variants for Ultralytics YOLOv8.

Place this file at:
    ultralytics/nn/modules/shuffle_densenet.py

Supported variants through YAML arguments:
    1. Shuffle-GConv:
       GroupConv + Channel Shuffle, no gate

    2. Shuffle-Gated:
       GroupConv + Channel Shuffle + bounded feature-preserving gate

    3. Shuffle-Enhance:
       GroupConv + Channel Shuffle + enhance-only gate

    4. MS-Shuffle-GConv:
       GroupConv + Channel Shuffle + optional dual-dilation 3x3 dense convolution

    5. Residual-ECA Shuffle-GConv:
       GroupConv + Channel Shuffle + block-level residual ECA enhancement

YAML argument order:
    [growth_rate, block_config, bn_size, drop_rate, gate_every,
     p3_channels, p4_channels, p5_channels,
     use_grouped, use_shuffle, use_gate, use_bounded_gate,
     shuffle_groups, gate_lambda, use_enhance_gate, use_msconv,
     use_block_eca, eca_lambda, eca_kernel_size]

The final arguments are optional and default to False/safe values, so previous
Shuffle-GConv, Shuffle-Gated, Shuffle-Enhance, and MS-Shuffle-GConv YAML files
still work.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ShuffleDenseNetBackbone"]


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels across groups with zero learnable parameters."""
    if groups <= 1:
        return x

    b, c, h, w = x.size()
    if c % groups != 0:
        return x

    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ChannelAttentionGate(nn.Module):
    """Standard squeeze-excitation style gate with scale range [0, 1]."""

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
        attention = self.fc(self.avg_pool(x))
        return x * attention


class FeaturePreservingGate(nn.Module):
    """Bounded channel gate that preserves weak lesion features.

    Formula:
        x_hat = x * (1 + lambda * (2a - 1))

    If lambda = 0.5, the scale range is [0.5, 1.5].
    """

    def __init__(self, channels: int, reduction: int = 4, gate_lambda: float = 0.5) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gate_lambda = float(gate_lambda)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.fc(self.avg_pool(x))
        scale = 1.0 + self.gate_lambda * (2.0 * attention - 1.0)
        return x * scale


class EnhanceOnlyGate(nn.Module):
    """Enhance-only channel gate.

    Formula:
        x_hat = x * (1 + lambda * a)

    It only enhances useful channels and does not suppress weak features.
    """

    def __init__(self, channels: int, reduction: int = 4, gate_lambda: float = 0.25) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gate_lambda = float(gate_lambda)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.fc(self.avg_pool(x))
        scale = 1.0 + self.gate_lambda * attention
        return x * scale


class ResidualECAGate(nn.Module):
    """Residual Efficient Channel Attention.

    This version avoids nn.Conv1d so it is safer with Ultralytics MuSGD.

    Formula:
        x_hat = x * (1 + lambda * attention)

    It only enhances channel features and does not directly suppress weak lesion features.
    """

    def __init__(self, channels: int, kernel_size: int = 3, eca_lambda: float = 0.10) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = int(kernel_size)
        self.eca_lambda = float(eca_lambda)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Store as 1D parameter for optimizer compatibility.
        # It is reshaped to [1, 1, K] only during forward.
        self.weight = nn.Parameter(torch.zeros(self.kernel_size))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = self.avg_pool(x)                  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)   # [B, 1, C]

        weight = self.weight.view(1, 1, self.kernel_size)
        y = F.conv1d(
            y,
            weight,
            padding=(self.kernel_size - 1) // 2,
        )

        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1) # [B, C, 1, 1]

        scale = 1.0 + self.eca_lambda * y
        return x * scale


class ShuffleDenseLayer(nn.Module):
    """DenseNet bottleneck layer with GroupConv, Channel Shuffle, and optional MSConv."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 2,
        drop_rate: float = 0.0,
        use_grouped: bool = True,
        use_shuffle: bool = True,
        shuffle_groups: int = 2,
        use_msconv: bool = False,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, inter_channels, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)

        groups = 1
        if use_grouped and inter_channels % shuffle_groups == 0 and growth_rate % shuffle_groups == 0:
            groups = shuffle_groups
        elif use_grouped and inter_channels % 2 == 0 and growth_rate % 2 == 0:
            groups = 2

        self.conv_groups = groups
        self.use_shuffle = bool(use_shuffle and groups > 1)
        self.shuffle_groups = groups

        branch1_channels = growth_rate // 2
        branch2_channels = growth_rate - branch1_channels

        self.use_msconv = bool(
            use_msconv
            and branch1_channels > 0
            and branch2_channels > 0
            and branch1_channels % groups == 0
            and branch2_channels % groups == 0
        )

        if self.use_msconv:
            self.conv2_d1 = nn.Conv2d(
                inter_channels,
                branch1_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=groups,
                bias=False,
            )
            self.conv2_d2 = nn.Conv2d(
                inter_channels,
                branch2_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                groups=groups,
                bias=False,
            )
        else:
            self.conv2 = nn.Conv2d(
                inter_channels,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=groups,
                bias=False,
            )

        self.drop_rate = float(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.relu2(self.norm2(out))

        # Backward compatibility:
        # Old Shuffle-GConv checkpoints do not have self.use_msconv.
        use_msconv = bool(getattr(self, "use_msconv", False))

        if use_msconv:
            out = torch.cat(
                [
                    self.conv2_d1(out),
                    self.conv2_d2(out),
                ],
                dim=1,
            )
        else:
            out = self.conv2(out)

        if self.use_shuffle:
            out = channel_shuffle(out, self.shuffle_groups)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class ShuffleGatedDenseBlock(nn.Module):
    """Dense block with GroupConv, Channel Shuffle, optional gate, and optional MSConv."""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        gate_every: int = 3,
        use_grouped: bool = True,
        use_shuffle: bool = True,
        use_gate: bool = True,
        use_bounded_gate: bool = True,
        shuffle_groups: int = 2,
        gate_lambda: float = 0.5,
        use_enhance_gate: bool = False,
        use_msconv: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleDict()

        self.use_gate = bool(use_gate)
        self.use_bounded_gate = bool(use_bounded_gate)
        self.use_enhance_gate = bool(use_enhance_gate)
        self.gate_indices = set(range(gate_every - 1, num_layers, gate_every)) if gate_every > 0 else set()

        for i in range(num_layers):
            self.layers.append(
                ShuffleDenseLayer(
                    num_input_features=num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    use_grouped=use_grouped,
                    use_shuffle=use_shuffle,
                    shuffle_groups=shuffle_groups,
                    use_msconv=use_msconv,
                )
            )

            if self.use_gate and i in self.gate_indices:
                if self.use_enhance_gate:
                    self.gates[str(i)] = EnhanceOnlyGate(
                        growth_rate,
                        reduction=4,
                        gate_lambda=gate_lambda,
                    )
                elif self.use_bounded_gate:
                    self.gates[str(i)] = FeaturePreservingGate(
                        growth_rate,
                        reduction=4,
                        gate_lambda=gate_lambda,
                    )
                else:
                    self.gates[str(i)] = ChannelAttentionGate(
                        growth_rate,
                        reduction=4,
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for i, layer in enumerate(self.layers):
            new_feature = layer(torch.cat(features, dim=1))
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


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class ShuffleDenseNetBackbone(nn.Module):
    """Shuffle-DenseNet lightweight backbone returning P3, P4, and P5.

    Args order for YOLO YAML:
        [growth_rate, block_config, bn_size, drop_rate, gate_every,
         p3_channels, p4_channels, p5_channels,
         use_grouped, use_shuffle, use_gate, use_bounded_gate,
         shuffle_groups, gate_lambda, use_enhance_gate, use_msconv,
         use_block_eca, eca_lambda, eca_kernel_size]
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
        use_shuffle: bool = True,
        use_gate: bool = True,
        use_bounded_gate: bool = True,
        shuffle_groups: int = 2,
        gate_lambda: float = 0.5,
        use_enhance_gate: bool = False,
        use_msconv: bool = False,
        use_block_eca: bool = False,
        eca_lambda: float = 0.10,
        eca_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if len(block_config) != 4:
            raise ValueError("block_config must contain four integers, e.g. [6, 8, 16, 8]")

        self.out_channels = [p3_channels, p4_channels, p5_channels]
        self.use_block_eca = bool(use_block_eca)

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
        self.block1 = ShuffleGatedDenseBlock(
            block_config[0],
            c0,
            bn_size,
            growth_rate,
            drop_rate,
            gate_every,
            use_grouped,
            use_shuffle,
            use_gate,
            use_bounded_gate,
            shuffle_groups,
            gate_lambda,
            use_enhance_gate,
            use_msconv,
        )
        c1 = c0 + block_config[0] * growth_rate
        self.trans1 = StandardTransition(c1, min_channels=32)
        c1t = self.trans1.out_channels
        self.p3_proj = nn.Conv2d(c1t, p3_channels, kernel_size=1, bias=False)

        # Stage 2, output stride 16 after transition.
        self.block2 = ShuffleGatedDenseBlock(
            block_config[1],
            c1t,
            bn_size,
            growth_rate,
            drop_rate,
            gate_every,
            use_grouped,
            use_shuffle,
            use_gate,
            use_bounded_gate,
            shuffle_groups,
            gate_lambda,
            use_enhance_gate,
            use_msconv,
        )
        c2 = c1t + block_config[1] * growth_rate
        self.trans2 = StandardTransition(c2, min_channels=32)
        c2t = self.trans2.out_channels
        self.p4_proj = nn.Conv2d(c2t, p4_channels, kernel_size=1, bias=False)

        # Stage 3, output stride 32 after transition.
        self.block3 = ShuffleGatedDenseBlock(
            block_config[2],
            c2t,
            bn_size,
            growth_rate,
            drop_rate,
            gate_every,
            use_grouped,
            use_shuffle,
            use_gate,
            use_bounded_gate,
            shuffle_groups,
            gate_lambda,
            use_enhance_gate,
            use_msconv,
        )
        c3 = c2t + block_config[2] * growth_rate
        self.trans3 = StandardTransition(c3, min_channels=32)
        c3t = self.trans3.out_channels

        # Stage 4, keep stride 32.
        self.block4 = ShuffleGatedDenseBlock(
            block_config[3],
            c3t,
            bn_size,
            growth_rate,
            drop_rate,
            gate_every,
            use_grouped,
            use_shuffle,
            use_gate,
            use_bounded_gate,
            shuffle_groups,
            gate_lambda,
            use_enhance_gate,
            use_msconv,
        )
        c4 = c3t + block_config[3] * growth_rate
        self.norm_final = nn.BatchNorm2d(c4)
        self.relu_final = nn.ReLU(inplace=True)
        self.p5_proj = nn.Conv2d(c4, p5_channels, kernel_size=1, bias=False)

        if self.use_block_eca:
            self.p3_eca = ResidualECAGate(p3_channels, kernel_size=eca_kernel_size, eca_lambda=eca_lambda)
            self.p4_eca = ResidualECAGate(p4_channels, kernel_size=eca_kernel_size, eca_lambda=eca_lambda)
            self.p5_eca = ResidualECAGate(p5_channels, kernel_size=eca_kernel_size, eca_lambda=eca_lambda)

        self.apply(_init_weights)

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

        # Backward compatibility:
        # Old checkpoints do not have self.use_block_eca or p*_eca modules.
        if bool(getattr(self, "use_block_eca", False)):
            p3 = self.p3_eca(p3)
            p4 = self.p4_eca(p4)
            p5 = self.p5_eca(p5)

        return [p3, p4, p5]
