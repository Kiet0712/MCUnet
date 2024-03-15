import torch
import torch.nn as nn
import torch.nn.functional as F
from CRFB_block import CRFB
from coordconv import CoordConv3d

class CRFBNet(nn.Module):
    def __init__(self,cfg,num_feature_start,depth):
        super().__init__()
        self.depth = depth
        self.feature_list = nn.ModuleList()
        if cfg.MODEL.NORM=='IN':
            norm = nn.InstanceNorm3d
        else:
            norm = nn.BatchNorm3d
        if cfg.MODEL.CONV_TYPE == 'normal':
            conv_type = nn.Conv3d
        elif cfg.MODEL.CONV_TYPE == 'coord':
            conv_type = CoordConv3d
        mapping_depth_channel = {
            0: num_feature_start,
            1: num_feature_start*2,
            2: num_feature_start*4,
            3: num_feature_start*8
        }
        for i in range(4):
            feature_list_i = nn.ModuleList()
            for j in range(depth):
                if j == 0:
                    feature_list_i.append(
                        nn.Sequential(
                            conv_type(mapping_depth_channel[i],mapping_depth_channel[i]//4,kernel_size=cfg.MODEL.CRFBNET_KERNEL_SIZE,padding=cfg.MODEL.CRFBNET_PADDING,bias=False),
                            norm(mapping_depth_channel[i]//4),
                            nn.ReLU(inplace=True)
                        )
                    )
                elif j == depth-1:
                    feature_list_i.append(
                        nn.Sequential(
                            conv_type(mapping_depth_channel[i]//4,mapping_depth_channel[i],kernel_size=cfg.MODEL.CRFBNET_KERNEL_SIZE,padding=cfg.MODEL.CRFBNET_PADDING,bias=False),
                            norm(mapping_depth_channel[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    feature_list_i.append(
                        nn.Sequential(
                            conv_type(mapping_depth_channel[i]//4,mapping_depth_channel[i]//4,kernel_size=cfg.MODEL.CRFBNET_KERNEL_SIZE,padding=cfg.MODEL.CRFBNET_PADDING,bias=False),
                            norm(mapping_depth_channel[i]//4),
                            nn.ReLU(inplace=True)
                        )
                    )
            self.feature_list.append(feature_list_i)
        self.CRFB_BLOCK_list = nn.ModuleList()
        for i in range(3):
            CRFB_i_list = nn.ModuleList()
            for j in range(depth):
                if j==depth-1:
                    CRFB_i_list.append(CRFB(cfg,mapping_depth_channel[i],mapping_depth_channel[i+1]))
                else:
                    CRFB_i_list.append(CRFB(cfg,mapping_depth_channel[i]//4,mapping_depth_channel[i+1]//4))
            self.CRFB_BLOCK_list.append(CRFB_i_list)
    def forward(self,x1,x2,x3,x4):
        xji = [x1,x2,x3,x4]
        for i in range(self.depth):
            for j in range(4):
                xji[j]=self.feature_list[j][i](xji[j])
            CRFB_i = []
            for j in range(3):
                x_j,x_j_plus = self.CRFB_BLOCK_list[j][i](xji[j],xji[j+1])
                CRFB_i.append([x_j,x_j_plus])
            for j in range(4):
                if j==0:
                    xji[j]=CRFB_i[j][0]
                elif j==3:
                    xji[j]=CRFB_i[j-1][1]
                else:
                    xji[j]=CRFB_i[j-1][1]+CRFB_i[j][0]
        return xji[0],xji[1],xji[2],xji[3]