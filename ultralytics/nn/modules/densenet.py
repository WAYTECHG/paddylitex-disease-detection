from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ema import EMA
from .lsk import LSKblock


def _init_weights(module: nn.Module) -> None:
    """Kaiming initialization for convolutions and standard BN init."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class RefinementStage(nn.Module):
  """Applies official LSK and/or EMA refinement modules to feature maps."""

  def __init__(
      self, channels: int, use_lsk: bool = False, use_ema: bool = False
  ):
    super().__init__()
    layers = []
    if use_lsk:
      layers.append(LSKblock(channels))
    if use_ema:
      layers.append(EMA(channels))
    self.refiner = nn.Sequential(*layers) if layers else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.refiner(x)

class _DenseLayer(nn.Module):
    """DenseNet bottleneck layer."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_feature = self.conv1(self.relu1(self.norm1(x)))
        new_feature = self.conv2(self.relu2(self.norm2(new_feature)))

        if self.drop_rate > 0:
            new_feature = F.dropout(new_feature, p=self.drop_rate, training=self.training)

        return torch.cat([x, new_feature], dim=1)


class _DenseBlock(nn.Module):
    """Stack of DenseNet bottleneck layers."""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DenseLayer(
                    num_input_features=num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """Standard DenseNet Transition Layer."""

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.relu(self.norm(x))))


class DenseNetBackbone(nn.Module):
    """
    Unified DenseNet Backbone with Boolean Toggles for:
    - use_lsk: Enables LSK spatial attention
    - use_ema: Enables EMA channel attention
    """

    def __init__(
        self,
        growth_rate: int | list[int] | tuple[int, ...] = 20,
        block_config: tuple[int, int, int, int] | list[int] = (6, 8, 12, 6),
        bn_size: int = 2,
        drop_rate: float = 0.0,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,
        use_lsk: bool = False,
        use_ema: bool = False,
    ) -> None:
        super().__init__()

        block_config = tuple(block_config)

        if isinstance(growth_rate, (int, float)):
            self.growth_rates = (int(growth_rate),) * 4
        elif isinstance(growth_rate, (list, tuple)):
            self.growth_rates = tuple(int(g) for g in growth_rate)

        # STEM
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        c0 = 64

        # Block 1
        self.block1 = _DenseBlock(block_config[0], c0, bn_size, self.growth_rates[0], drop_rate)
        c1 = c0 + block_config[0] * self.growth_rates[0]
        self.trans1 = _Transition(c1, c1 // 2)
        c2_in = c1 // 2

        # Block 2
        self.block2 = _DenseBlock(block_config[1], c2_in, bn_size, self.growth_rates[1], drop_rate)
        c2 = c2_in + block_config[1] * self.growth_rates[1]
        self.trans2 = _Transition(c2, c2 // 2)
        c3_in = c2 // 2

        # Block 3
        self.block3 = _DenseBlock(block_config[2], c3_in, bn_size, self.growth_rates[2], drop_rate)
        c3 = c3_in + block_config[2] * self.growth_rates[2]
        self.trans3 = _Transition(c3, c3 // 2)
        c4_in = c3 // 2

        # Block 4
        self.block4 = _DenseBlock(block_config[3], c4_in, bn_size, self.growth_rates[3], drop_rate)
        c4 = c4_in + block_config[3] * self.growth_rates[3]

        # Head Projections
        self.p3_conv = nn.Conv2d(c2_in, p3_channels, kernel_size=1, bias=False)
        self.p4_conv = nn.Conv2d(c3_in, p4_channels, kernel_size=1, bias=False)
        self.p5_conv = nn.Conv2d(c4, p5_channels, kernel_size=1, bias=False)

        # Refinement Modules (LSK, EMA, or both)
        self.p3_refiner = RefinementStage(p3_channels, use_lsk=use_lsk, use_ema=use_ema)
        self.p4_refiner = RefinementStage(p4_channels, use_lsk=use_lsk, use_ema=use_ema)
        self.p5_refiner = RefinementStage(p5_channels, use_lsk=use_lsk, use_ema=use_ema)

        self.out_channels = [p3_channels, p4_channels, p5_channels]
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)

        x = self.block1(x)
        x = self.trans1(x)
        p3 = self.p3_conv(x)

        x = self.block2(x)
        x = self.trans2(x)
        p4 = self.p4_conv(x)

        x = self.block3(x)
        x = self.trans3(x)
        x = self.block4(x)
        p5 = self.p5_conv(x)

        # Apply spatial/channel refinements
        p3 = self.p3_refiner(p3)
        p4 = self.p4_refiner(p4)
        p5 = self.p5_refiner(p5)

        return [p3, p4, p5]