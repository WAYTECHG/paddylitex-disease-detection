# Ultralytics custom module for Thesis: DenseLite + LSK + EMA
# Place this file at: ultralytics/nn/modules/denselite_lsk_ema.py
# Scope: backbone-only enhancement. It wraps your existing DenseNet-style backbone classes.

import torch
import torch.nn as nn

from .densenet import DenseNetBackbone
from .hybrid_densenet_idgt import HybridDenseNetIDGTBackbone
from .shuffle_densenet import ShuffleDenseNetBackbone


class ConvBNAct(nn.Module):
    """Small Conv-BN-SiLU helper used only inside the custom backbone refiner."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LSKBlock(nn.Module):
    """
    Lightweight Large Selective Kernel-style spatial refinement.

    It is shape-preserving:
        [B, C, H, W] -> [B, C, H, W]

    Role for your thesis:
    - keeps DenseNet feature reuse untouched;
    - adds larger spatial context inside the backbone wrapper;
    - helps the model focus on wider lesion/leaf context before YOLO neck fusion.
    """

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 8)

        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dw7_dilated = nn.Conv2d(
            channels, channels, kernel_size=7, padding=9, dilation=3, groups=channels, bias=False
        )

        self.reduce_local = ConvBNAct(channels, hidden, k=1, p=0)
        self.reduce_large = ConvBNAct(channels, hidden, k=1, p=0)
        self.spatial_select = nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=True)
        self.proj = ConvBNAct(hidden, channels, k=1, p=0, act=False)

        # Residual scaling keeps training stable at the start.
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        local = self.dw5(x)
        large = self.dw7_dilated(local)

        local = self.reduce_local(local)
        large = self.reduce_large(large)

        merged = torch.cat([local, large], dim=1)
        avg_attn = torch.mean(merged, dim=1, keepdim=True)
        max_attn, _ = torch.max(merged, dim=1, keepdim=True)
        select = torch.sigmoid(self.spatial_select(torch.cat([avg_attn, max_attn], dim=1)))

        selected = local * select[:, 0:1, :, :] + large * select[:, 1:2, :, :]
        selected = self.proj(selected)
        return x + self.gamma * selected


class EMARefiner(nn.Module):
    """
    Efficient Multi-Scale Attention-inspired grouped feature refiner.

    This is a lightweight, shape-preserving attention block:
        [B, C, H, W] -> [B, C, H, W]

    It uses grouped channel processing with height-wise and width-wise contextual gates,
    plus a local 3x3 branch.
    """

    def __init__(self, channels, groups=8):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1

        self.groups = groups
        self.group_channels = channels // groups

        self.conv1x1 = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(
            self.group_channels,
            self.group_channels,
            kernel_size=3,
            padding=1,
            groups=1,
            bias=False,
        )
        self.gn = nn.GroupNorm(1, self.group_channels)
        self.out_proj = ConvBNAct(channels, channels, k=1, p=0, act=False)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.groups
        cg = self.group_channels

        xg = x.reshape(b * g, cg, h, w)

        # Directional context gates.
        x_h = xg.mean(dim=3, keepdim=True)  # [B*g, Cg, H, 1]
        x_w = xg.mean(dim=2, keepdim=True).transpose(2, 3)  # [B*g, Cg, W, 1]
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        gate_h, gate_w = torch.split(hw, [h, w], dim=2)
        gate_w = gate_w.transpose(2, 3)

        gated = self.gn(xg * torch.sigmoid(gate_h) * torch.sigmoid(gate_w))
        local = self.conv3x3(xg)

        # Lightweight cross-branch attention.
        weight_gated = torch.softmax(gated.mean(dim=(2, 3), keepdim=True), dim=1)
        weight_local = torch.softmax(local.mean(dim=(2, 3), keepdim=True), dim=1)
        refined = gated * weight_local + local * weight_gated

        refined = refined.reshape(b, c, h, w)
        refined = self.out_proj(refined)
        return x + self.gamma * refined


class LSKEMAFeatureRefiner(nn.Module):
    """
    Applies LSK + EMA to P3/P4/P5 features returned by the DenseNet-style backbone.

    It is lazy-built during YOLO's dummy forward pass, so it adapts automatically to
    the output channel sizes defined in your YAML: P3=64, P4=128, P5=256.
    """

    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        self.refiners = nn.ModuleList()
        self._built = False

    def _build(self, features):
        self.refiners = nn.ModuleList(
            nn.Sequential(
                LSKBlock(int(f.shape[1])),
                EMARefiner(int(f.shape[1]), groups=self.groups),
            )
            for f in features
        )
        self._built = True
        self.to(features[0].device)

    def forward(self, features):
        if not isinstance(features, (list, tuple)):
            return features
        if len(features) < 3:
            return features
        if not self._built:
            self._build(features)

        refined = []
        for i, f in enumerate(features):
            refined.append(self.refiners[i](f) if i < len(self.refiners) else f)
        return refined if isinstance(features, list) else tuple(refined)


class LSKEMADenseNetBackbone(DenseNetBackbone):
    """DenseNetBackbone + LSK/EMA backbone-output refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsk_ema_refiner = LSKEMAFeatureRefiner(groups=8)

    def forward(self, x):
        features = super().forward(x)
        return self.lsk_ema_refiner(features)


class LSKEMAHybridDenseNetIDGTBackbone(HybridDenseNetIDGTBackbone):
    """HybridDenseNetIDGTBackbone + LSK/EMA backbone-output refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsk_ema_refiner = LSKEMAFeatureRefiner(groups=8)

    def forward(self, x):
        features = super().forward(x)
        return self.lsk_ema_refiner(features)


class LSKEMAShuffleDenseNetBackbone(ShuffleDenseNetBackbone):
    """ShuffleDenseNetBackbone + LSK/EMA backbone-output refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsk_ema_refiner = LSKEMAFeatureRefiner(groups=8)

    def forward(self, x):
        features = super().forward(x)
        return self.lsk_ema_refiner(features)


class ModifiedDenseNetBackbone(LSKEMAHybridDenseNetIDGTBackbone):
    """
    Thesis-friendly alias for the current proposed DenseNet-style backbone.

    Current implementation:
        Compact DenseNet-lite backbone + LSK spatial refinement + EMA feature refinement.

    Why this class exists:
    - The YAML can use a stable, proposal-friendly name: ModifiedDenseNetBackbone.
    - The internal design can evolve during ablation without exposing older implementation names
      such as HybridDenseNetIDGTBackbone in the YAML.

    Reproducibility note:
    Once final experiments are reported, freeze this class implementation or create a versioned
    class such as ModifiedDenseNetBackboneV2 before changing the architecture again.
    """

    pass
