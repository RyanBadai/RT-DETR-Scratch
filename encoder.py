import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention-based Intra-scale Feature Interaction (AIFI)
class AIFI(nn.Module):
    def __init__(self, dim):
        super(AIFI, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        x = x.view(B, C, seq_len).permute(0, 2, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

# Cross-scale Feature-fusion Module (CCFM)
class CCFM(nn.Module):
    def __init__(self, channels):
        super(CCFM, self).__init__()
        self.conv_s4 = nn.Conv2d(channels[0], channels[2], kernel_size=1)
        self.conv_s5 = nn.Conv2d(channels[1], channels[2], kernel_size=1)
        self.fusion = nn.Conv2d(channels[2] * 3, channels[2], kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(channels[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s4, s5, s6):
        s4 = F.interpolate(self.conv_s4(s4), size=s6.shape[2:], mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.conv_s5(s5), size=s6.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([s4, s5, s6], dim=1)
        fused = self.fusion(fused)
        fused = self.norm(fused)
        fused = self.relu(fused)
        return fused

# Efficient Hybrid Encoder
class EfficientHybridEncoder(nn.Module):
    def __init__(self, channels=[128, 256, 512]):
        super(EfficientHybridEncoder, self).__init__()
        self.aifi = AIFI(channels[2])
        self.ccfm = CCFM(channels)

    def forward(self, features):
        s4, s5, s6 = features
        fused = self.ccfm(s4, s5, s6)
        output = self.aifi(fused)
        return output