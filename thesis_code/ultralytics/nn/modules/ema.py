# import torch
# from torch import nn


# class EMA(nn.Module):
#     """Efficient Multi-Scale Attention (EMA) Module - Dynamic & Robust Version."""

#     def __init__(self, channels: int, c2: int | None = None, factor: int = 32):
#         super().__init__()
#         # Dynamic group selection to guarantee channels % groups == 0
#         groups = factor
#         while groups > 1 and (channels % groups != 0 or channels // groups == 0):
#             groups //= 2

#         self.groups = max(1, groups)
#         self.group_channels = channels // self.groups

#         self.softmax = nn.Softmax(dim=-1)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))

#         self.gn = nn.GroupNorm(self.group_channels, self.group_channels)
#         self.conv1x1 = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, stride=1, padding=1)

#     @torch.amp.autocast('cuda', enabled=False)  # Prevents FP16 overflow in matmul/softmax
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = x.size()
#         group_x = x.reshape(b * self.groups, self.group_channels, h, w)  # b*g, c//g, h, w

#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))

#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)

#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, self.group_channels, -1)  # b*g, c//g, hw

#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, self.group_channels, -1)  # b*g, c//g, hw

#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)

import torch
from torch import nn


class EMA(nn.Module):
  """Efficient Multi-Scale Attention (EMA) Module - Dynamic & Robust Version."""

  def __init__(self, channels: int, c2: int | None = None, factor: int = 32):
    super().__init__()
    groups = factor
    while groups > 1 and (channels % groups != 0 or channels // groups == 0):
      groups //= 2

    self.groups = max(1, groups)
    self.group_channels = channels // self.groups

    self.softmax = nn.Softmax(dim=-1)
    self.agp = nn.AdaptiveAvgPool2d((1, 1))
    self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
    self.pool_w = nn.AdaptiveAvgPool2d((1, None))

    self.gn = nn.GroupNorm(self.group_channels, self.group_channels)
    self.conv1x1 = nn.Conv2d(
        self.group_channels,
        self.group_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    self.conv3x3 = nn.Conv2d(
        self.group_channels,
        self.group_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )

  @torch.amp.autocast('cuda', dtype=torch.float32)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_type = x.dtype
    x = x.float()  # Explicitly cast input to FP32

    b, c, h, w = x.size()
    group_x = x.reshape(
        b * self.groups, self.group_channels, h, w
    )  # b*g, c//g, h, w

    x_h = self.pool_h(group_x)
    x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
    hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))

    x_h, x_w = torch.split(hw, [h, w], dim=2)
    x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
    x2 = self.conv3x3(group_x)

    x11 = self.softmax(
        self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
    )
    x12 = x2.reshape(b * self.groups, self.group_channels, -1)

    x21 = self.softmax(
        self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
    )
    x22 = x1.reshape(b * self.groups, self.group_channels, -1)

    weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
        b * self.groups, 1, h, w
    )
    out = (group_x * weights.sigmoid()).reshape(b, c, h, w)

    return out.to(
        orig_type
    )  # Return tensor in original precision for downstream YOLO layers