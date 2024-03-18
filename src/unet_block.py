import torch
import torch.nn as nn
import torch.nn.functional as F
from double_conv import DoubleConv
from attention import Attention_block
from coordconv import CoordConv3d,TransposeCoordConv3d,CoordConv2d,TransposeCoordConv2d

class Down(nn.Module):
    def __init__(self,cfg,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2) if cfg.MODEL.CONV_TYPE[-2:]=='3d' else nn.MaxPool2d(2),
            DoubleConv(cfg,in_channels, out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)
class NormalUp(nn.Module):
    def __init__(self,cfg,in_channels,out_channels):
        super().__init__()
        if cfg.MODEL.CONV_TYPE == 'normal3d':
            conv_type = nn.Conv3d
            conv_transpose = nn.ConvTranspose3d
        elif cfg.MODEL.CONV_TYPE == 'coord3d':
            conv_type = CoordConv3d
            conv_transpose = TransposeCoordConv3d
        elif cfg.MODEL.CONV_TYPE == 'normal2d':
            conv_type = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        elif cfg.MODEL.CONV_TYPE == 'coord2d':
            conv_type = CoordConv2d
            conv_transpose = TransposeCoordConv2d
        self.up = conv_transpose(in_channels,in_channels//2,2,2)
        self.conv = DoubleConv(cfg,in_channels,out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)
class AttentionUp(nn.Module):
    def __init__(self,cfg,in_channels,out_channels):
        super().__init__()
        if cfg.MODEL.CONV_TYPE == 'normal3d':
            conv_type = nn.Conv3d
            conv_transpose = nn.ConvTranspose3d
        elif cfg.MODEL.CONV_TYPE == 'coord3d':
            conv_type = CoordConv3d
            conv_transpose = TransposeCoordConv3d
        elif cfg.MODEL.CONV_TYPE == 'normal2d':
            conv_type = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        elif cfg.MODEL.CONV_TYPE == 'coord2d':
            conv_type = CoordConv2d
            conv_transpose = TransposeCoordConv2d
        self.up = conv_transpose(in_channels,in_channels//2,2,2)
        self.conv = DoubleConv(cfg,in_channels,out_channels)
        self.attention = Attention_block(cfg,in_channels//2,in_channels//2,in_channels//4)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x2 = self.attention(x1,x2)
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)
