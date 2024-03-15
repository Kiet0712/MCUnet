import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv3d,TransposeCoordConv3d


class MultiPath(nn.Module):
    def __init__(self,cfg,num_feature_aim):
        super().__init__()
        if cfg.MODEL.NORM=='IN':
            norm = nn.InstanceNorm3d
        else:
            norm = nn.BatchNorm3d
        if cfg.MODEL.CONV_TYPE == 'normal':
            conv_type = nn.Conv3d
            conv_transpose = nn.ConvTranspose3d
        elif cfg.MODEL.CONV_TYPE == 'coord':
            conv_type = CoordConv3d
            conv_transpose = TransposeCoordConv3d
        num_feature_start = cfg.MODEL.NUM_FEATURES_START_UNET
        num_feature_list = [num_feature_start*(2**i) for i in range(4)]
        if num_feature_aim==num_feature_list[3]:
            self.conv1 = nn.Sequential(
                conv_type(num_feature_list[0],num_feature_list[0]*2,2,2,0,bias=False),
                norm(num_feature_list[0]*2),
                nn.ReLU(inplace=True),
                conv_type(num_feature_list[0]*2,num_feature_list[0]*4,2,2,0,bias=False),
                norm(num_feature_list[0]*4),
                nn.ReLU(inplace=True),
                conv_type(num_feature_list[0]*4,num_feature_aim//4,2,2,0),
            )
            self.conv2 = nn.Sequential(
                conv_type(num_feature_list[1],num_feature_list[1]*2,2,2,0,bias=False),
                norm(num_feature_list[1]*2),
                nn.ReLU(inplace=True),
                conv_type(num_feature_list[1]*2,num_feature_aim//4,2,2,0)
            )
            self.conv3 = conv_type(num_feature_list[2],num_feature_aim//4,2,2,0)
            self.conv4 = conv_type(num_feature_list[3],num_feature_aim//4,1,1,0)
        elif num_feature_aim==num_feature_list[2]:
            self.conv1 = nn.Sequential(
                conv_type(num_feature_list[0],num_feature_list[0]*2,2,2,0,bias=False),
                norm(num_feature_list[0]*2),
                nn.ReLU(inplace=True),
                conv_type(num_feature_list[0]*2,num_feature_aim//4,2,2,0)
            )
            self.conv2 = conv_type(num_feature_list[1],num_feature_aim//4,2,2,0)
            self.conv3 = conv_type(num_feature_list[2],num_feature_aim//4,1,1,0)
            self.conv4 = conv_transpose(num_feature_list[3],num_feature_aim//4,2,2,0)
        elif num_feature_aim==num_feature_list[1]:
            self.conv1 = conv_type(num_feature_list[0],num_feature_aim//4,2,2,0)
            self.conv2 = conv_type(num_feature_list[1],num_feature_aim//4,1,1,0)
            self.conv3 = conv_transpose(num_feature_list[2],num_feature_aim//4,2,2,0)
            self.conv4 = nn.Sequential(
                conv_transpose(num_feature_list[3],num_feature_list[3]//2,2,2,0,bias=False),
                norm(num_feature_list[3]//2),
                nn.ReLU(inplace=True),
                conv_transpose(num_feature_list[3]//2,num_feature_aim//4,2,2,0)
            )
        else:
            self.conv1 = conv_type(num_feature_list[0],num_feature_aim//4,1,1,0)
            self.conv2 = conv_transpose(num_feature_list[1],num_feature_aim//4,2,2,0)
            self.conv3 = nn.Sequential(
                conv_transpose(num_feature_list[2],num_feature_list[2]//2,2,2,0,bias=False),
                norm(num_feature_list[2]//2),
                nn.ReLU(inplace=True),
                conv_transpose(num_feature_list[2]//2,num_feature_aim//4,2,2,0)
            )
            self.conv4 = nn.Sequential(
                conv_transpose(num_feature_list[3],num_feature_list[3]//2,2,2,0,bias=False),
                norm(num_feature_list[3]//2),
                nn.ReLU(inplace=True),
                conv_transpose(num_feature_list[3]//2,num_feature_list[3]//4,2,2,0,bias=False),
                norm(num_feature_list[3]//4),
                nn.ReLU(inplace=True),
                conv_transpose(num_feature_list[3]//4,num_feature_aim//4,2,2,0)
            )
    def forward(self,x1,x2,x3,x4):
        x_1_path = self.conv1(x1)
        x_2_path = self.conv2(x2)
        x_3_path = self.conv3(x3)
        x_4_path = self.conv4(x4)
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1))