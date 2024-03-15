import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv3d,TransposeCoordConv3d,CoordConv2d,TransposeCoordConv2d


class DoubleConv(nn.Module):
    def __init__(self,cfg,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if cfg.MODEL.NORM=='IN3d':
            norm_type = nn.InstanceNorm3d
        elif cfg.MODEL.NORM == 'BN3d':
            norm_type = nn.BatchNorm3d
        elif cfg.MODEL.NORM=='IN2d':
            norm_type = nn.InstanceNorm2d
        elif cfg.MODEL.NORM == 'BN2d':
            norm_type = nn.BatchNorm2d
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
        self.double_conv = nn.Sequential(
            conv_type(in_channels,mid_channels,3,1,1,bias=False),
            norm_type(mid_channels),
            nn.ReLU(True),
            conv_type(mid_channels,out_channels,3,1,1,bias=False),
            norm_type(out_channels)
        )
        self.conv_residual = False
        if cfg.MODEL.DOUBLE_CONV_RESIDUAL:
            self.conv_skip_connection = nn.Sequential(
                    conv_type(in_channels,out_channels,1,1)
            )
            self.conv_residual = True
    def forward(self,x):
        output = self.double_conv(x)
        if self.conv_residual:
            output = F.relu(output+self.conv_skip_connection(x),inplace=True)
            return output
        else:
            return output
