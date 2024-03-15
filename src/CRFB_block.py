import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv3d,TransposeCoordConv3d,CoordConv2d,TransposeCoordConv2d

class CRFB(nn.Module):
    def __init__(self,cfg,top_channels,bottom_channels):
        super().__init__()
        top_channels = int(top_channels)
        bottom_channels = int(bottom_channels)
        if cfg.MODEL.NORM=='IN3d':
            norm = nn.InstanceNorm3d
        elif cfg.MODEL.NORM == 'BN3d':
            norm = nn.BatchNorm3d
        elif cfg.MODEL.NORM=='IN2d':
            norm = nn.InstanceNorm2d
        elif cfg.MODEL.NORM == 'BN2d':
            norm = nn.BatchNorm2d
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
        self.conv = nn.Sequential(
            conv_type(top_channels,top_channels//4,1,1,0,bias=False),
            norm(top_channels//4),
            nn.ReLU(inplace=True),
            conv_type(top_channels//4,bottom_channels,2,2,0)
        )
        self.deconv = nn.Sequential(
            conv_type(bottom_channels,bottom_channels//4,1,1,0,bias=False),
            norm(bottom_channels//4),
            nn.ReLU(inplace=True),
            conv_transpose(bottom_channels//4,top_channels,2,2,0)
        )
    def forward(self,top,bottom):
        return F.relu(self.deconv(bottom)+top,True),F.relu(self.conv(top)+bottom,True)