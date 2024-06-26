import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv3d,CoordConv2d
class Attention_block(nn.Module):
    def __init__(self,cfg,F_g,F_l,F_int):
        super().__init__()
        if cfg.MODEL.NORM == 'IN3d':
            norm_type = nn.InstanceNorm3d
        elif cfg.MODEL.NORM == 'BN3d':
            norm_type = nn.BatchNorm3d
        elif cfg.MODEL.NORM == 'IN2d':
            norm_type = nn.InstanceNorm2d
        elif cfg.MODEL.NORM == 'BN2d':
            norm_type = nn.BatchNorm2d
        if cfg.MODEL.CONV_TYPE == 'normal3d':
            conv_type = nn.Conv3d
        elif cfg.MODEL.CONV_TYPE == 'coord3d':
            conv_type = CoordConv3d
        elif cfg.MODEL.CONV_TYPE == 'normal2d':
            conv_type = nn.Conv2d
        elif cfg.MODEL.CONV_TYPE == 'coord2d':
            conv_type = CoordConv2d
        self.W_g = nn.Sequential(
            conv_type(F_g,F_int,1,bias=False),
            norm_type(F_int)
        )
        self.W_x = nn.Sequential(
            conv_type(F_l,F_int,1,bias=False),
            norm_type(F_int)
        )
        self.psi = nn.Sequential(
            conv_type(F_int,1,1,bias=False),
            norm_type(1),
            nn.Sigmoid()
        )
        self.relu_ = nn.ReLU(inplace=True)
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_block = self.relu_(g1+x1)
        psi_block = self.psi(psi_block)
        return x*psi_block