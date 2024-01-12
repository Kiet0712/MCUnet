import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_conv import MDynamic_conv3d
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
        return F.relu(self.double_conv(x)+self.conv_skip_connection(x))
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
            nn.Conv3d(in_channels,n_channels,kernel_size=1),
            nn.Sigmoid()
        )
        self.mask_head = nn.Sequential(
            nn.Conv3d(in_channels+n_channels,out_channels*n_channels*2,kernel_size=1),
            nn.Sigmoid()
        )
        self.class_segment_conv = nn.Sequential(
            nn.Conv3d(in_channels+n_channels*2,out_channels//3,kernel_size=1),
            nn.Sigmoid()
        )
        self.class_variant_guide = DoubleConv(n_channels*2,n_channels*2)
        self.reconstruct_variant_guide = DoubleConv(n_channels,n_channels)
        self.attention_class = Attention_block(in_channels,n_channels*2,in_channels//2)
        self.attention_reconstruct = Attention_block(in_channels,n_channels,in_channels//2)
    def forward(self, x):
        reconstruct_volume = self.reconstruct_volume_conv(x)
        mask_head = self.mask_head(torch.cat([x,self.attention_reconstruct(x,self.reconstruct_variant_guide(reconstruct_volume))],dim=1))
        class_1_guide = self.class_variant_guide(torch.cat([mask_head[:,0:4,:,:,:],mask_head[:,12:16,:,:,:]],dim=1))
        class_2_guide = self.class_variant_guide(torch.cat([mask_head[:,4:8,:,:,:],mask_head[:,16:20,:,:,:]],dim=1))
        class_4_guide = self.class_variant_guide(torch.cat([mask_head[:,8:12,:,:,:],mask_head[:,20:,:,:,:]],dim=1))
        class_1_segment_vol = self.class_segment_conv(torch.cat([x,self.attention_class(x,class_1_guide)],dim=1))
        class_2_segment_vol = self.class_segment_conv(torch.cat([x,self.attention_class(x,class_2_guide)],dim=1))
        class_4_segment_vol = self.class_segment_conv(torch.cat([x,self.attention_class(x,class_4_guide)],dim=1))
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
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1))
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
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1))
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
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1))
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
        return F.relu(torch.cat([x_1_path,x_2_path,x_3_path,x_4_path],dim=1))
class CMDCFB(nn.Module):
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
        self.dynamic_conv_top = nn.Sequential(
            MDynamic_conv3d(top_channels,top_channels,1),
            nn.InstanceNorm3d(top_channels)
        )
        self.dynamic_conv_bottom = nn.Sequential(
            MDynamic_conv3d(bottom_channels,bottom_channels,1),
            nn.InstanceNorm3d(bottom_channels)
        )
    def forward(self,top,bottom):
        return F.relu(self.dynamic_conv_top(top,self.deconv(bottom))),F.relu(self.dynamic_conv_bottom(bottom,self.conv(top)))
class CMDCFBNetwork(nn.Module):
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
        self.CMDCFB1_1 = CMDCFB(16*scale,32*scale)
        self.CMDCFB1_2 = CMDCFB(16*scale,32*scale)
        self.CMDCFB1_3 = CMDCFB(16*scale,32*scale)
        self.CMDCFB1_4 = CMDCFB(64*scale,128*scale)
        self.CMDCFB2_1 = CMDCFB(32*scale,64*scale)
        self.CMDCFB2_2 = CMDCFB(32*scale,64*scale)
        self.CMDCFB2_3 = CMDCFB(32*scale,64*scale)
        self.CMDCFB2_4 = CMDCFB(128*scale,256*scale)
        self.CMDCFB3_1 = CMDCFB(64*scale,128*scale)
        self.CMDCFB3_2 = CMDCFB(64*scale,128*scale)
        self.CMDCFB3_3 = CMDCFB(64*scale,128*scale)
        self.CMDCFB3_4 = CMDCFB(256*scale,512*scale)
    def forward(self,x1,x2,x3,x4):
        x1_1 = self.conv1_1(x1)
        x2_1 = self.conv2_1(x2)
        x3_1 = self.conv3_1(x3)
        x4_1 = self.conv4_1(x4)
        x_CMDCFB_1_1_1,x_CMDCFB_1_2_1 = self.CMDCFB1_1(x1_1,x2_1)
        x_CMDCFB_2_2_1,x_CMDCFB_2_3_1 = self.CMDCFB2_1(x2_1,x3_1)
        x_CMDCFB_3_3_1,x_CMDCFB_3_4_1 = self.CMDCFB3_1(x3_1,x4_1)
        x1_2 = self.conv1_2(x_CMDCFB_1_1_1)
        x2_2 = self.conv2_2(x_CMDCFB_1_2_1+x_CMDCFB_2_2_1)
        x3_2 = self.conv3_2(x_CMDCFB_2_3_1+x_CMDCFB_3_3_1)
        x4_2 = self.conv4_2(x_CMDCFB_3_4_1)
        x_CMDCFB_1_1_2,x_CMDCFB_1_2_2 = self.CMDCFB1_2(x1_2,x2_2)
        x_CMDCFB_2_2_2,x_CMDCFB_2_3_2 = self.CMDCFB2_2(x2_2,x3_2)
        x_CMDCFB_3_3_2,x_CMDCFB_3_4_2 = self.CMDCFB3_2(x3_2,x4_2)
        x1_3 = self.conv1_3(x_CMDCFB_1_1_2)
        x2_3 = self.conv2_3(x_CMDCFB_1_2_2+x_CMDCFB_2_2_2)
        x3_3 = self.conv3_3(x_CMDCFB_2_3_2+x_CMDCFB_3_3_2)
        x4_3 = self.conv4_3(x_CMDCFB_3_4_2)
        x_CMDCFB_1_1_3,x_CMDCFB_1_2_3 = self.CMDCFB1_3(x1_3,x2_3)
        x_CMDCFB_2_2_3,x_CMDCFB_2_3_3 = self.CMDCFB2_3(x2_3,x3_3)
        x_CMDCFB_3_3_3,x_CMDCFB_3_4_3 = self.CMDCFB3_3(x3_3,x4_3)
        x1_4 = self.conv1_4(x_CMDCFB_1_1_3)
        x2_4 = self.conv2_4(x_CMDCFB_1_2_3+x_CMDCFB_2_2_3)
        x3_4 = self.conv3_4(x_CMDCFB_2_3_3+x_CMDCFB_3_3_3)
        x4_4 = self.conv4_4(x_CMDCFB_3_4_3)
        x_CMDCFB_1_1_4,x_CMDCFB_1_2_4 = self.CMDCFB1_4(x1_4,x2_4)
        x_CMDCFB_2_2_4,x_CMDCFB_2_3_4 = self.CMDCFB2_4(x2_4,x3_4)
        x_CMDCFB_3_3_4,x_CMDCFB_3_4_4 = self.CMDCFB3_4(x3_4,x4_4)
        return x_CMDCFB_1_1_4,x_CMDCFB_1_2_4+x_CMDCFB_2_2_4,x_CMDCFB_2_3_4+x_CMDCFB_3_3_4,x_CMDCFB_3_4_4
class ASRCBGAMHCMDCFB(nn.Module):
    def __init__(self, n_channels, n_classes,scale=0.5):
        super(ASRCBGAMHCMDCFB, self).__init__()
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
        self.CMDCFBnet = CMDCFBNetwork(scale)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path = self.CMDCFBnet(x1,x2,x3,x4)
        x = self.up1(x5, self.MP1(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        x = self.up2(x, self.MP2(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        x = self.up3(x, self.MP3(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        x = self.up4(x, self.MP4(x_1_res_path,x_2_res_path,x_3_res_path,x_4_res_path))
        output = self.outc(x)
        return output
