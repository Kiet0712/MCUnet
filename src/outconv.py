import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv3d
from attention import Attention_block


class OutConv(nn.Module):
    def __init__(self,cfg, in_channels, out_channels,n_channels):
        super().__init__()
        self.MULTIHEAD_OUTPUT = cfg.MODEL.MULTIHEAD_OUTPUT
        self.OUTPUT_COORDCONV = cfg.MODEL.OUTPUT_COORDCONV
        self.SELF_GUIDE_OUTPUT = cfg.MODEL.SELF_GUIDE_OUTPUT
        self.SEPERATE_FEATURE = cfg.MODEL.SEPERATE_FEATURE
        if cfg.MODEL.OUTPUT_COORDCONV:
            conv = CoordConv3d
        else:
            conv = nn.Conv3d
        if cfg.MODEL.NORM=='IN':
            norm = nn.InstanceNorm3d
        else:
            norm = nn.BatchNorm3d
        if cfg.MODEL.MULTIHEAD_OUTPUT:
            if cfg.MODEL.SEPERATE_FEATURE:
                if cfg.MODEL.SELF_GUIDE_OUTPUT:
                    if cfg.DIALTED_OUTPUT:
                        self.reconstruct_volume_conv = nn.Sequential(
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,n_channels,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                        self.mask_head_conv = nn.Sequential(
                            conv(in_channels+n_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,out_channels*n_channels*2,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv_featrue = nn.Sequential(
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True)
                        )
                        self.segment_volume_conv_1 = nn.Sequential(
                            conv(in_channels+n_channels*2,out_channels//3,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv_2 = nn.Sequential(
                            conv(in_channels+n_channels*2,out_channels//3,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv_4 = nn.Sequential(
                            conv(in_channels+n_channels*2,out_channels//3,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                        self.attention_reconstruct = Attention_block(cfg,in_channels,n_channels,in_channels//4)
                        self.attention_class_1 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                        self.attention_class_2 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                        self.attention_class_4 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                    else:
                        self.reconstruct_volume_conv = nn.Sequential(
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,n_channels,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
                        self.mask_head_conv = nn.Sequential(
                            conv(in_channels+n_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,out_channels*n_channels*2,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv_featrue = nn.Sequential(
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True)
                        )
                        self.segment_volume_conv_1 = nn.Sequential(
                            conv(in_channels+n_channels*2,out_channels//3,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv_2 = nn.Sequential(
                            conv(in_channels+n_channels*2,out_channels//3,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv_4 = nn.Sequential(
                            conv(in_channels+n_channels*2,out_channels//3,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
                        self.attention_reconstruct = Attention_block(cfg,in_channels,n_channels,in_channels//4)
                        self.attention_class_1 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                        self.attention_class_2 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                        self.attention_class_4 = Attention_block(cfg,in_channels,n_channels*2,in_channels//4)
                else:
                    if cfg.DIALTED_OUTPUT:
                        self.reconstruct_volume_conv = nn.Sequential(
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,n_channels,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                        self.mask_head_conv = nn.Sequential(
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,out_channels*n_channels*2,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv = nn.Sequential(
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,out_channels,kernel_size=1,padding=1//2,bias=True),
                            nn.Sigmoid()
                        )
                    else:
                        self.reconstruct_volume_conv = nn.Sequential(
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,n_channels,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
                        self.mask_head_conv = nn.Sequential(
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,out_channels*n_channels*2,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
                        self.segment_volume_conv = nn.Sequential(
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,in_channels,3,1,1,bias=False),
                            norm(in_channels),
                            nn.ReLU(inplace=True),
                            conv(in_channels,out_channels,kernel_size=1,bias=True),
                            nn.Sigmoid()
                        )
            else:
                if cfg.MODEL.SELF_GUIDE_OUTPUT:
                    self.reconstruct_volume_conv = nn.Sequential(
                        conv(in_channels,n_channels,kernel_size=1),
                        nn.Sigmoid()
                    )
                    self.mask_head = nn.Sequential(
                        conv(in_channels+n_channels,out_channels*n_channels*2,kernel_size=1),
                        nn.Sigmoid()
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
                        nn.Conv3d(in_channels,n_channels,kernel_size=1),
                        nn.Sigmoid()
                    )
                    self.mask_head = nn.Sequential(
                        nn.Conv3d(in_channels,out_channels*n_channels*2,kernel_size=1),
                        nn.Sigmoid()
                    )
                    self.segment_conv = nn.Sequential(
                        nn.Conv3d(in_channels,out_channels,kernel_size=1),
                        nn.Sigmoid()
                    )
        else:
            self.output_conv = nn.Sequential(
                conv(in_channels,out_channels,1),
                nn.Sigmoid()
            )
    def forward(self,x):
        if self.MULTIHEAD_OUTPUT:
            if self.SEPERATE_FEATURE:
                if self.SELF_GUIDE_OUTPUT:
                    reconstruct_volume = self.reconstruct_volume_conv(x)
                    mask_head = self.mask_head_conv(torch.cat([x,self.attention_reconstruct(x,reconstruct_volume)],dim=1))
                    class_1_guide = torch.cat([mask_head[:,0:4,:,:,:],mask_head[:,12:16,:,:,:]],dim=1)
                    class_2_guide = torch.cat([mask_head[:,4:8,:,:,:],mask_head[:,16:20,:,:,:]],dim=1)
                    class_4_guide = torch.cat([mask_head[:,8:12,:,:,:],mask_head[:,20:,:,:,:]],dim=1)
                    segment_vol_feature = self.segment_volume_conv_featrue(x)
                    class_1_segment_vol = self.segment_volume_conv_1(torch.cat([segment_vol_feature,self.attention_class_1(segment_vol_feature,class_1_guide)],dim=1))
                    class_2_segment_vol = self.segment_volume_conv_2(torch.cat([segment_vol_feature,self.attention_class_2(segment_vol_feature,class_2_guide)],dim=1))
                    class_4_segment_vol = self.segment_volume_conv_4(torch.cat([segment_vol_feature,self.attention_class_4(segment_vol_feature,class_4_guide)],dim=1))
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
                    segment_volume = self.segment_volume_conv(x)
                    reconstruct_volume = self.reconstruct_volume_conv(x)
                    mask_head = self.mask_head_conv(x)
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
            

