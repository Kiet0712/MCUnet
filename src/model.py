import torch
import torch.nn as nn
import torch.nn.functional as F
from double_conv import DoubleConv
from outconv import OutConv
from CRFBNet import CRFBNet
from multipath import MultiPath
from unet_block import Down,NormalUp,AttentionUp
class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        num_feature_start = cfg.MODEL.NUM_FEATURES_START_UNET
        self.inc = DoubleConv(cfg,4,num_feature_start)
        self.down1 = Down(cfg,num_feature_start,num_feature_start*2)
        self.down2 = Down(cfg,num_feature_start*2,num_feature_start*4)
        self.down3 = Down(cfg,num_feature_start*4,num_feature_start*8)
        self.down4 = Down(cfg,num_feature_start*8,num_feature_start*16)
        if cfg.MODEL.ATTENTION_UP:
            Up = AttentionUp
        else:
            Up = NormalUp
        self.up1 = Up(cfg,num_feature_start*16,num_feature_start*8)
        self.up2 = Up(cfg,num_feature_start*8,num_feature_start*4)
        self.up3 = Up(cfg,num_feature_start*4,num_feature_start*2)
        self.up4 = Up(cfg,num_feature_start*2,num_feature_start)
        self.HAVE_CRFB_NET = False
        self.MULTIPATH = False
        if cfg.MODEL.CRFBNET:
            self.CRFBnet = CRFBNet(cfg,num_feature_start,cfg.MODEL.CRFBNET_DEPTH)
            self.HAVE_CRFB_NET = True
        if cfg.MODEL.MULTI_PATH_COMBINE:
            self.MULTIPATH = True
            self.MP1 = MultiPath(cfg,num_feature_start*8)
            self.MP2 = MultiPath(cfg,num_feature_start*4)
            self.MP3 = MultiPath(cfg,num_feature_start*2)
            self.MP4 = MultiPath(cfg,num_feature_start)
        self.outconv = OutConv(cfg,num_feature_start,cfg.N_CLASS,cfg.CHANNEL_IN)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.HAVE_CRFB_NET:
            x1,x2,x3,x4 = self.CRFBnet(x1,x2,x3,x4)
        if self.MULTIPATH:
            x = self.up1(x5,self.MP1(x1,x2,x3,x4))
            x = self.up2(x,self.MP2(x1,x2,x3,x4))
            x = self.up3(x,self.MP3(x1,x2,x3,x4))
            x = self.up4(x,self.MP4(x1,x2,x3,x4))
            return self.outconv(x)
        else:
            x = self.up1(x5,x4)
            x = self.up2(x,x3)
            x = self.up3(x,x2)
            x = self.up4(x,x1)
            return self.outconv(x)