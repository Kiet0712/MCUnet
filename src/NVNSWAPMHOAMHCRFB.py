import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g,F_int,1,bias=False),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l,F_int,1,bias=False),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int,1,1,bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu_ = nn.ReLU(inplace=True)
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_block = self.relu_(g1+x1)
        psi_block = self.psi(psi_block)
        return x*psi_block
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels,mid_channels,3,1,1,bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(True),
            nn.Conv3d(mid_channels,out_channels,3,1,1,bias=False),
            nn.InstanceNorm3d(out_channels)
        )
        self.conv_skip_connection = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,1,1)
        )
    def forward(self,x):
        return F.relu(self.double_conv(x)+self.conv_skip_connection(x),True)
class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels,in_channels//2,2,2)
        self.conv = DoubleConv(in_channels,out_channels)
        self.attention = Attention_block(in_channels//2,in_channels//2,in_channels//4)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x2 = self.attention(x1,x2)
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels,n_channels):
        super(OutConv, self).__init__()
        self.reconstruct_volume_conv = nn.Sequential(
            nn.Conv3d(in_channels,32,kernel_size=3,padding=2,dilation=2,bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,32,kernel_size=3,padding=2,dilation=2,bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,n_channels,kernel_size=1,padding=1//2,bias=True),
            nn.Sigmoid()
        )
        self.mask_head_conv = nn.Sequential(
            nn.Conv3d(in_channels+n_channels,32,kernel_size=3,padding=2,dilation=2,bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,32,kernel_size=3,padding=2,dilation=2,bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,out_channels*n_channels*2,kernel_size=1,padding=1//2,bias=True),
            nn.Sigmoid()
        )
        self.segment_volume_conv_featrue = nn.Sequential(
            nn.Conv3d(in_channels,32,kernel_size=3,padding=2,dilation=2,bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,32,kernel_size=3,padding=2,dilation=2,bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.segment_volume_conv_1 = nn.Sequential(
            nn.Conv3d(32+n_channels*2,out_channels//3,kernel_size=1,padding=1//2,bias=True),
            nn.Sigmoid()
        )
        self.segment_volume_conv_2 = nn.Sequential(
            nn.Conv3d(32+n_channels*2,out_channels//3,kernel_size=1,padding=1//2,bias=True),
            nn.Sigmoid()
        )
        self.segment_volume_conv_4 = nn.Sequential(
            nn.Conv3d(32+n_channels*2,out_channels//3,kernel_size=1,padding=1//2,bias=True),
            nn.Sigmoid()
        )
        self.attention_reconstruct = Attention_block(in_channels,n_channels,in_channels//4)
        self.attention_class_1 = Attention_block(in_channels,n_channels*2,in_channels//4)
        self.attention_class_2 = Attention_block(in_channels,n_channels*2,in_channels//4)
        self.attention_class_4 = Attention_block(in_channels,n_channels*2,in_channels//4)
    def forward(self, x):
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
class MP1(nn.Module):
    def __init__(self,scale):
        super(MP1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(int(64*scale),int(32*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(32*scale)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(32*scale),int(32*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(32*scale)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(32*scale),int(128*scale),2,2,0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(int(128*scale),int(64*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(64*scale)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(64*scale),int(128*scale),2,2,0)
        )
        self.conv3 = nn.Conv3d(int(256*scale),int(128*scale),2,2,0)
        self.conv4 = nn.Conv3d(int(512*scale),int(128*scale),1,1,0)
    def forward(self,x1,x2,x3,x4):
        x_1_path = self.conv1(x1)
        x_2_path = self.conv2(x2)
        x_3_path = self.conv3(x3)
        x_4_path = self.conv4(x4)
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1),True)
class MP2(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(int(64*scale),int(32*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(32*scale)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(32*scale),int(64*scale),2,2,0)
        )
        self.conv2 = nn.Conv3d(int(128*scale),int(64*scale),2,2,0)
        self.conv3 = nn.Conv3d(int(256*scale),int(64*scale),1,1,0)
        self.conv4 = nn.ConvTranspose3d(int(512*scale),int(64*scale),2,2,0)
    def forward(self,x1,x2,x3,x4):
        x_1_path = self.conv1(x1)
        x_2_path = self.conv2(x2)
        x_3_path = self.conv3(x3)
        x_4_path = self.conv4(x4)
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1),True)
class MP3(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(int(64*scale),int(32*scale),2,2,0)
        self.conv2 = nn.Conv3d(int(128*scale),int(32*scale),1,1,0)
        self.conv3 = nn.ConvTranspose3d(int(256*scale),int(32*scale),2,2,0)
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(int(512*scale),int(16*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(16*scale)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(int(16*scale),int(32*scale),2,2,0)
        )
    def forward(self,x1,x2,x3,x4):
        x_1_path = self.conv1(x1)
        x_2_path = self.conv2(x2)
        x_3_path = self.conv3(x3)
        x_4_path = self.conv4(x4)
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1),True)
class MP4(nn.Module):
    def __init__(self,scale) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(int(64*scale),int(16*scale),1,1,0)
        self.conv2 = nn.ConvTranspose3d(int(128*scale),int(16*scale),2,2,0)
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(int(256*scale),int(8*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(8*scale)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(int(8*scale),int(16*scale),2,2,0)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(int(512*scale),int(8*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(8*scale)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(int(8*scale),int(8*scale),2,2,0,bias=False),
            nn.InstanceNorm3d(int(8*scale)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(int(8*scale),int(16*scale),2,2,0)
        )
    def forward(self,x1,x2,x3,x4):
        x_1_path = self.conv1(x1)
        x_2_path = self.conv2(x2)
        x_3_path = self.conv3(x3)
        x_4_path = self.conv4(x4)
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1),True)
class CRFB(nn.Module):
    def __init__(self,top_channels,bottom_channels):
        super().__init__()
        top_channels = int(top_channels)
        bottom_channels = int(bottom_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(top_channels,top_channels//4,1,1,0,bias=False),
            nn.InstanceNorm3d(top_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv3d(top_channels//4,bottom_channels,2,2,0)
        )
        self.deconv = nn.Sequential(
            nn.Conv3d(bottom_channels,bottom_channels//4,1,1,0,bias=False),
            nn.InstanceNorm3d(bottom_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(bottom_channels//4,top_channels,2,2,0)
        )
    def forward(self,top,bottom):
        return F.relu(self.deconv(bottom)+top,True),F.relu(self.conv(top)+bottom,True)
class CRFBNetwork(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(int(64*scale),int(16*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(16*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(int(16*scale),int(16*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(16*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv3d(int(16*scale),int(16*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(16*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv3d(int(16*scale),int(64*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(64*scale),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(int(128*scale),int(32*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(32*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv3d(int(32*scale),int(32*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(32*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv3d(int(32*scale),int(32*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(32*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv2_4 = nn.Sequential(
            nn.Conv3d(int(32*scale),int(128*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(128*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(int(256*scale),int(64*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(64*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv3d(int(64*scale),int(64*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(64*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv3d(int(64*scale),int(64*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(64*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv3d(int(64*scale),int(256*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(256*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv3d(int(512*scale),int(128*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(128*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv3d(int(128*scale),int(128*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(128*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv3d(int(128*scale),int(128*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(128*scale)),
            nn.ReLU(inplace=True)
        )
        self.conv4_4 = nn.Sequential(
            nn.Conv3d(int(128*scale),int(512*scale),3,1,1,bias=False),
            nn.InstanceNorm3d(int(512*scale)),
            nn.ReLU(inplace=True)
        )
        self.CRFB1_1 = CRFB(16*scale,32*scale)
        self.CRFB1_2 = CRFB(16*scale,32*scale)
        self.CRFB1_3 = CRFB(16*scale,32*scale)
        self.CRFB1_4 = CRFB(64*scale,128*scale)
        self.CRFB2_1 = CRFB(32*scale,64*scale)
        self.CRFB2_2 = CRFB(32*scale,64*scale)
        self.CRFB2_3 = CRFB(32*scale,64*scale)
        self.CRFB2_4 = CRFB(128*scale,256*scale)
        self.CRFB3_1 = CRFB(64*scale,128*scale)
        self.CRFB3_2 = CRFB(64*scale,128*scale)
        self.CRFB3_3 = CRFB(64*scale,128*scale)
        self.CRFB3_4 = CRFB(256*scale,512*scale)
    def forward(self,x1,x2,x3,x4):
        x1_1 = self.conv1_1(x1)
        x2_1 = self.conv2_1(x2)
        x3_1 = self.conv3_1(x3)
        x4_1 = self.conv4_1(x4)
        x_crfb_1_1_1,x_crfb_1_2_1 = self.CRFB1_1(x1_1,x2_1)
        x_crfb_2_2_1,x_crfb_2_3_1 = self.CRFB2_1(x2_1,x3_1)
        x_crfb_3_3_1,x_crfb_3_4_1 = self.CRFB3_1(x3_1,x4_1)
        x1_2 = self.conv1_2(x_crfb_1_1_1)
        x2_2 = self.conv2_2(x_crfb_1_2_1+x_crfb_2_2_1)
        x3_2 = self.conv3_2(x_crfb_2_3_1+x_crfb_3_3_1)
        x4_2 = self.conv4_2(x_crfb_3_4_1)
        x_crfb_1_1_2,x_crfb_1_2_2 = self.CRFB1_2(x1_2,x2_2)
        x_crfb_2_2_2,x_crfb_2_3_2 = self.CRFB2_2(x2_2,x3_2)
        x_crfb_3_3_2,x_crfb_3_4_2 = self.CRFB3_2(x3_2,x4_2)
        x1_3 = self.conv1_3(x_crfb_1_1_2)
        x2_3 = self.conv2_3(x_crfb_1_2_2+x_crfb_2_2_2)
        x3_3 = self.conv3_3(x_crfb_2_3_2+x_crfb_3_3_2)
        x4_3 = self.conv4_3(x_crfb_3_4_2)
        x_crfb_1_1_3,x_crfb_1_2_3 = self.CRFB1_3(x1_3,x2_3)
        x_crfb_2_2_3,x_crfb_2_3_3 = self.CRFB2_3(x2_3,x3_3)
        x_crfb_3_3_3,x_crfb_3_4_3 = self.CRFB3_3(x3_3,x4_3)
        x1_4 = self.conv1_4(x_crfb_1_1_3)
        x2_4 = self.conv2_4(x_crfb_1_2_3+x_crfb_2_2_3)
        x3_4 = self.conv3_4(x_crfb_2_3_3+x_crfb_3_3_3)
        x4_4 = self.conv4_4(x_crfb_3_4_3)
        x_crfb_1_1_4,x_crfb_1_2_4 = self.CRFB1_4(x1_4,x2_4)
        x_crfb_2_2_4,x_crfb_2_3_4 = self.CRFB2_4(x2_4,x3_4)
        x_crfb_3_3_4,x_crfb_3_4_4 = self.CRFB3_4(x3_4,x4_4)
        return x_crfb_1_1_4,x_crfb_1_2_4+x_crfb_2_2_4,x_crfb_2_3_4+x_crfb_3_3_4,x_crfb_3_4_4
class NVNSWAPMHOAMHCRFB(nn.Module):
    def __init__(self, n_channels, n_classes,scale=0.5):
        super(NVNSWAPMHOAMHCRFB, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, int(64*scale)))
        self.down1 = (Down(int(64*scale), int(128*scale)))
        self.down2 = (Down(int(128*scale), int(256*scale)))
        self.down3 = (Down(int(256*scale), int(512*scale)))
        self.down4 = (Down(int(512*scale), int(1024*scale)))
        self.up1 = (Up(int(1024*scale), int(512*scale)))
        self.up2 = (Up(int(512*scale), int(256*scale)))
        self.up3 = (Up(int(256*scale), int(128*scale)))
        self.up4 = (Up(int(128*scale), int(64*scale)))
        self.outc = (OutConv(int(64*scale), n_classes,n_channels))
        self.MP1 = MP1(scale)
        self.MP2 = MP2(scale)
        self.MP3 = MP3(scale)
        self.MP4 = MP4(scale)
        self.CRFBnet = CRFBNetwork(scale)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path = self.CRFBnet(x1,x2,x3,x4)
        x = self.up1(x5, self.MP1(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        x = self.up2(x, self.MP2(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        x = self.up3(x, self.MP3(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        x = self.up4(x, self.MP4(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        output = self.outc(x)
        return output
