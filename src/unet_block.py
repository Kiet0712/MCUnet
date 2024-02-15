import torch
import torch.nn as nn
import torch.nn.functional as F
from double_conv import DoubleConv
from attention import Attention_block

class Down(nn.Module):
    def __init__(self,cfg,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(cfg,in_channels, out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)
class NormalUp(nn.Module):
    def __init__(self,cfg,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels,in_channels//2,2,2)
        self.conv = DoubleConv(cfg,in_channels,out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)
class AttentionUp(nn.Module):
    def __init__(self,cfg,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels,in_channels//2,2,2)
        self.conv = DoubleConv(cfg,in_channels,out_channels)
        self.attention = Attention_block(cfg,in_channels//2,in_channels//2,in_channels//4)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x2 = self.attention(x1,x2)
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)
