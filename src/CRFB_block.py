import torch
import torch.nn as nn
import torch.nn.functional as F


class CRFB(nn.Module):
    def __init__(self,cfg,top_channels,bottom_channels):
        super().__init__()
        top_channels = int(top_channels)
        bottom_channels = int(bottom_channels)
        if cfg.MODEL.NORM=='IN':
            norm = nn.InstanceNorm3d
        else:
            norm = nn.BatchNorm3d
        self.conv = nn.Sequential(
            nn.Conv3d(top_channels,top_channels//4,1,1,0,bias=False),
            norm(top_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv3d(top_channels//4,bottom_channels,2,2,0)
        )
        self.deconv = nn.Sequential(
            nn.Conv3d(bottom_channels,bottom_channels//4,1,1,0,bias=False),
            norm(bottom_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(bottom_channels//4,top_channels,2,2,0)
        )
    def forward(self,top,bottom):
        return F.relu(self.deconv(bottom)+top,True),F.relu(self.conv(top)+bottom,True)