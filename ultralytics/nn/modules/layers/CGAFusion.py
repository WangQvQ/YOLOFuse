# 论文名称：DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention
# 论文地址：https://arxiv.org/abs/2301.04805
# 代码地址：https://github.com/cecret3350/DEA-Net/tree/main
# 博客地址：https://blog.csdn.net/weixin_43694096/article/details/137383380

import torch
from torch import nn
from einops.layers.torch import Rearrange

__all__ = ["CGAFusion"]


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(
            2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange("b c t h w -> b (c t) h w")(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.a = 0.5
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # print(x1.shape)
        # print(x2.shape)
        initial = torch.add(x1, x2, alpha=self.a)
        # print(initial.shape)

        cattn = self.ca(initial)
        # print(f"cattn is {cattn.shape}")
        sattn = self.sa(initial)
        # print(f"sattn is {sattn.shape}")

        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x1 + (1 - pattn2) * x2
        result = self.conv(result)
        return result
