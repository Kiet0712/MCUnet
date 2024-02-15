import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    def __init__(self,cfg,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if cfg.MODEL.DOUBLE_CONV_TYPE == 'normal':
            conv_type = nn.Conv3d
        if cfg.MODEL.NORM == 'IN':
            norm_type = nn.InstanceNorm3d
        elif cfg.MODEL.NORM == 'BN':
            norm_type = nn.BatchNorm3d
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
