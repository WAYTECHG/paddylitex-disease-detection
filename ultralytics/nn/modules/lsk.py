import torch
from torch import nn


class LSKblock(nn.Module):
    """Large Selective Kernel (LSK) Module - Stable Implementation."""

    def __init__(self, dim: int, c2: int | None = None):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        
        # Batch normalization stabilizes unbounded conv output before multiplication
        self.bn = nn.BatchNorm2d(dim)

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        x = x.float()  # Explicitly cast input to FP32

        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]
        
        # Apply sigmoid to strictly bound spatial attention weights to [0, 1]
        attn = self.conv(attn).sigmoid()
        
        out = x * attn

        return out.to(orig_type)


LSKBlock = LSKblock