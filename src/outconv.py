import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv3d,TransposeCoordConv3d,CoordConv2d,TransposeCoordConv2d
from attention import Attention_block


class OutConv(nn.Module):
    def __init__(self,cfg, in_channels, out_channels,n_channels):
        super().__init__()
        self.MULTIHEAD_OUTPUT = cfg.MODEL.MULTIHEAD_OUTPUT
        self.OUTPUT_COORDCONV = cfg.MODEL.OUTPUT_COORDCONV
        self.SELF_GUIDE_OUTPUT = cfg.MODEL.SELF_GUIDE_OUTPUT
        if cfg.MODEL.NORM=='IN3d':
            norm = nn.InstanceNorm3d
        elif cfg.MODEL.NORM == 'BN3d':
            norm = nn.BatchNorm3d
        elif cfg.MODEL.NORM=='IN2d':
            norm = nn.InstanceNorm2d
        elif cfg.MODEL.NORM == 'BN2d':
            norm = nn.BatchNorm2d
        if cfg.MODEL.CONV_TYPE == 'normal3d':
            conv = nn.Conv3d
            conv_transpose = nn.ConvTranspose3d
        elif cfg.MODEL.CONV_TYPE == 'coord3d':
            conv = CoordConv3d
            conv_transpose = TransposeCoordConv3d
        elif cfg.MODEL.CONV_TYPE == 'normal2d':
            conv = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        elif cfg.MODEL.CONV_TYPE == 'coord2d':
            conv = CoordConv2d
            conv_transpose = TransposeCoordConv2d
        if cfg.MODEL.MULTIHEAD_OUTPUT:
            if cfg.MODEL.SELF_GUIDE_OUTPUT:
                self.reconstruct_volume_conv = nn.Sequential(
                    conv(in_channels,n_channels,kernel_size=1),
                    nn.Sigmoid() if cfg.MODEL.SELF_GUIDE_OUTPUT_SIGMOID else nn.Identity()
                )
                self.mask_head = nn.Sequential(
                    conv(in_channels+n_channels,out_channels*n_channels*2,kernel_size=1),
                    nn.Sigmoid() if cfg.MODEL.SELF_GUIDE_OUTPUT_SIGMOID else nn.Identity()
                )
                self.class_1_segment_conv = nn.Sequential(
                    conv(in_channels+n_channels*2,out_channels//3,kernel_size=1),
                    nn.Sigmoid()
                )
                self.class_2_segment_conv = nn.Sequential(
                    conv(in_channels+n_channels*2,out_channels//3,kernel_size=1),
                    nn.Sigmoid()
                )
                self.class_4_segment_conv = nn.Sequential(
                    conv(in_channels+n_channels*2,out_channels//3,kernel_size=1),
                    nn.Sigmoid()
                )
                self.attention_class_1 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                self.attention_class_2 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                self.attention_class_4 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                self.attention_reconstruct = Attention_block(cfg,in_channels,n_channels,in_channels//4)
            else:
                self.reconstruct_volume_conv = nn.Sequential(
                    conv(in_channels,n_channels,kernel_size=1),
                    nn.Sigmoid() if cfg.MODEL.SELF_GUIDE_OUTPUT_SIGMOID else nn.Identity()
                )
                self.mask_head = nn.Sequential(
                    conv(in_channels,out_channels*n_channels*2,kernel_size=1),
                    nn.Sigmoid() if cfg.MODEL.SELF_GUIDE_OUTPUT_SIGMOID else nn.Identity()
                )
                self.segment_conv = nn.Sequential(
                    conv(in_channels,out_channels,kernel_size=1),
                    nn.Sigmoid()
                )
        else:
            self.output_conv = nn.Sequential(
                conv(in_channels,out_channels,1),
                nn.Sigmoid()
            )
    def forward(self,x):
        if self.MULTIHEAD_OUTPUT:
            if self.SELF_GUIDE_OUTPUT:
                reconstruct_volume = self.reconstruct_volume_conv(x)
                mask_head = self.mask_head(torch.cat([x,self.attention_reconstruct(x,reconstruct_volume)],dim=1))
                class_1_guide = torch.cat([mask_head[:,0:4,:,:,:],mask_head[:,12:16,:,:,:]],dim=1)
                class_2_guide = torch.cat([mask_head[:,4:8,:,:,:],mask_head[:,16:20,:,:,:]],dim=1)
                class_4_guide = torch.cat([mask_head[:,8:12,:,:,:],mask_head[:,20:,:,:,:]],dim=1)
                class_1_segment_vol = self.class_1_segment_conv(torch.cat([x,self.attention_class_1(x,class_1_guide)],dim=1))
                class_2_segment_vol = self.class_2_segment_conv(torch.cat([x,self.attention_class_2(x,class_2_guide)],dim=1))
                class_4_segment_vol = self.class_4_segment_conv(torch.cat([x,self.attention_class_4(x,class_4_guide)],dim=1))
                segment_volume = torch.cat([class_4_segment_vol,class_1_segment_vol,class_2_segment_vol],dim=1)
                return {
                    'segment_volume':segment_volume,
                    'reconstruct_volume':reconstruct_volume,
                    'class_1_foreground':mask_head[:,0:4,:,:,:],
                    'class_2_foreground':mask_head[:,4:8,:,:,:],
                    'class_4_foreground':mask_head[:,8:12,:,:,:],
                    'class_1_background':mask_head[:,12:16,:,:,:],
                    'class_2_background':mask_head[:,16:20,:,:,:],
                    'class_4_background':mask_head[:,20:,:,:,:]
                }
            else:
                reconstruct_volume = self.reconstruct_volume_conv(x)
                mask_head = self.mask_head(x)
                segment_volume = self.segment_conv(x)
                return {
                    'segment_volume':segment_volume,
                    'reconstruct_volume':reconstruct_volume,
                    'class_1_foreground':mask_head[:,0:4,:,:,:],
                    'class_2_foreground':mask_head[:,4:8,:,:,:],
                    'class_4_foreground':mask_head[:,8:12,:,:,:],
                    'class_1_background':mask_head[:,12:16,:,:,:],
                    'class_2_background':mask_head[:,16:20,:,:,:],
                    'class_4_background':mask_head[:,20:,:,:,:]
                }
        else:
            return self.output_conv(x)
            

