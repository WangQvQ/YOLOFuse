import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from einops import rearrange

class DEFA(nn.Module):
    """Dynamic Enhanced Fusion Attention"""
    def __init__(self, channels=512, reduction=16, num_heads=8, dynamic_k=5):
        super().__init__()
        # 1. 动态核参数学习
        self.dynamic_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 4 * channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * channels, dynamic_k**2, 1),
            nn.Softmax(dim=1)
        )

        # 2. 交叉注意力机制
        self.cross_attn = MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # 3. 门控融合（hidden 最少为 1）
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1),
            nn.Sigmoid()
        )

        # 4. 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        self.dynamic_k = dynamic_k

    def forward(self, x):
        vi_feat, ir_feat = x  # 两路输入特征

        # --- 动态核参数学习 ---
        kernel_weights = self.dynamic_conv(vi_feat + ir_feat)  # [B, k^2, 1, 1]
        b, k2, _, _ = kernel_weights.size()
        k = self.dynamic_k

        # --- 交叉注意力交互 ---
        vi_seq = rearrange(vi_feat, 'b c h w -> b (h w) c')
        ir_seq = rearrange(ir_feat, 'b c h w -> b (h w) c')
        fused_seq, _ = self.cross_attn(vi_seq, ir_seq, ir_seq)
        fused_feat = rearrange(fused_seq, 'b (h w) c -> b c h w', h=vi_feat.size(2))

        # --- 动态卷积融合 ---
        U = F.unfold(
            fused_feat,
            kernel_size=k,
            padding=k // 2
        )  # [B, C * k^2, N]
        B, CK2, N = U.shape
        # 重新 reshape 权重并扩展到 N
        kw = kernel_weights.view(B, k * k, 1).expand(-1, -1, N)
        # 爱因斯坦求和融合
        dynamic_feat = torch.einsum('bkn,bckn->bcn', kw, U.view(B, -1, k * k, N))
        dynamic_feat = F.fold(
            dynamic_feat,
            output_size=vi_feat.shape[2:],
            kernel_size=1,
            stride=1
        )  # [B, C, H, W]

        # --- 门控融合 ---
        gates = self.gate(torch.cat([vi_feat, ir_feat], dim=1))  # [B, 2, H, W]
        w_vi, w_ir = gates[:, 0:1], gates[:, 1:2]
        out_vi = w_ir * dynamic_feat + vi_feat
        out_ir = w_vi * dynamic_feat + ir_feat

        # --- 融合两路输出 & 特征增强 ---
        fused = out_vi + out_ir  # 将两路融合
        return self.enhance(fused)


if __name__ == "__main__":
    # 测试维度 (B=2, C=512, H=80, W=80)
    vi = torch.randn(2, 512, 80, 80)
    ir = torch.randn(2, 512, 80, 80)
    model = DEFA(channels=512, reduction=16, num_heads=8, dynamic_k=5)
    out = model((vi, ir))
    print(f"Output shape: {out.shape}")  # 输出应为 (2, 512, 80, 80)
