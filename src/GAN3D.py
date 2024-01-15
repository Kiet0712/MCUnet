import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from MHCRFB import MHCRFB as MHCRFB
from AttentionMHCRFB import AMHCRFB as AMHCRFB
from ClassBaseGuideAMHCRFB import CBGAMHCRFB as CBGAMHCRFB
from RClassBaseGuideAMHCRFB import RCBGAMHCRFB as RCBGAMHCRFB
from SRCBGAMHCARFB import SRCBGAMHCARFB as SRCBGAMHCARFB
from SCBGAMHCRFB import SCBGAMHCRFB as SCBGAMHCRFB
from ASRCBGAMHCRFB import ASRCBGAMHCRFB as ASRCBGAMHCRFB
from SCASRCBGAMHCRFB import SCASRCBGAMHCRFB as SCASRCBGAMHCRFB
from SRCBDCGAMHCRFB import SRCBDCGAMHCRFB as SRCBDCGAMHCRFB
from SRCBMDCGAMHCRFB import SRCBMDCGAMHCRFB as SRCBMDCGAMHCRFB
from NSWSRCBMDCGAMHCRFB import NSWSRCBMDCGAMHCRFB as NSWSRCBMDCGAMHCRFB
from ASRCBGAMHCMDCFB import ASRCBGAMHCMDCFB as ASRCBGAMHCMDCFB
from NVNSWASRCBGAMHCRFB import NVNSWASRCBGAMHCRFB as NVNSWASRCBGAMHCRFB
from NSWASRCBGAMHCRFB import NSWASRCBGAMHCRFB as NSWASRCBGAMHCRFB
from RNVNSWASRCBGAMHCRFB import RNVNSWASRCBGAMHCRFB as RNVNSWASRCBGAMHCRFB
from NVNSWRASRCBGAMHCRFB import NVNSWRASRCBGAMHCRFB as NVNSWRASRCBGAMHCRFB
model_choice = {
    'MHCRFB':MHCRFB,
    'AMHCRFB':AMHCRFB,
    'CBGAMHCRFB':CBGAMHCRFB,
    'RCBGAMHCRFB':RCBGAMHCRFB,
    'SCBGAMHCRFB':SCBGAMHCRFB,
    'SRCBGAMHCARFB':SRCBGAMHCARFB,
    'ASRCBGAMHCRFB':ASRCBGAMHCRFB,
    'SCASRCBGAMHCRFB':SCASRCBGAMHCRFB,
    'SRCBDCGAMHCRFB':SRCBDCGAMHCRFB,
    'SRCBMDCGAMHCRFB':SRCBMDCGAMHCRFB,
    'NSWSRCBMDCGAMHCRFB':NSWSRCBMDCGAMHCRFB,
    'ASRCBGAMHCMDCFB':ASRCBGAMHCMDCFB,
    'NVNSWASRCBGAMHCRFB':NVNSWASRCBGAMHCRFB,
    'NSWASRCBGAMHCRFB':NSWASRCBGAMHCRFB,
    'RNVNSWASRCBGAMHCRFB':RNVNSWASRCBGAMHCRFB,
    'NVNSWRASRCBGAMHCRFB':NVNSWRASRCBGAMHCRFB
}
class DiscriminatorConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels,out_channels,kernel_size,stride,0),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2,True)
        )
    def forward(self,x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self,img_size,device) -> None:
        super().__init__()
        self.embedding = nn.Linear(4,img_size**3,bias=False)
        self.model = nn.Sequential(
            DiscriminatorConvBlock(5,32,4,2),
            DiscriminatorConvBlock(32,64,4,2),
            DiscriminatorConvBlock(64,128,4,2),
            DiscriminatorConvBlock(128,256,4,1),
            nn.ReflectionPad3d(1),
            nn.Conv3d(256,1,4,1,0)
        )
        self.img_size = img_size
        self.label_embedding = torch.from_numpy(
            np.array([
                [1,0,0,0],
                [1,0,0,1],
                [0,1,0,0],
                [0,1,0,1],
                [0,0,1,0],
                [0,0,1,1]
            ],dtype=np.float32)
        ).to(device)
    def forward(self,x):
        inputs = torch.cat([
            x['class_1_background'],
            x['class_1_foreground'],
            x['class_2_background'],
            x['class_2_foreground'],
            x['class_4_background'],
            x['class_4_foreground']
        ],dim=0)
        label_embed = F.leaky_relu(self.embedding(self.label_embedding),0.2,True)
        label_embed = torch.reshape(label_embed,[-1,1,self.img_size,self.img_size,self.img_size])
        inputs = torch.cat([inputs,label_embed],dim=1)
        return self.model(inputs)
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,D,inputs,label,patchGAN_fake_output,patchGAN_fake,patchGAN_true):
        inputs_for_D = {
            'class_1_background':inputs*(1-label[:,1:2,:,:,:]),
            'class_1_foreground':inputs*label[:,1:2,:,:,:],
            'class_2_background':inputs*(1-label[:,2:,:,:,:]),
            'class_2_foreground':inputs*label[:,2:,:,:,:],
            'class_4_background':inputs*(1-label[:,0:1,:,:,:]),
            'class_4_foreground':inputs*label[:,0:1,:,:,:],
        }
        patchGAN_true_output = D(inputs_for_D)
        return F.mse_loss(patchGAN_true_output,patchGAN_true)+F.mse_loss(patchGAN_fake_output,patchGAN_fake)
class GAN3D(nn.Module):
    def __init__(self,n_channels, n_classes,device,weigt_adversarial,model_choice_str,data_augmentation):
        super().__init__()
        self.G = model_choice[model_choice_str](n_channels,n_classes).to(device)
        self.D = Discriminator(128,device).to(device)
        self.optim_G = torch.optim.Adam(self.G.parameters(),2e-4,(0.5,0.999))
        self.optim_D = torch.optim.Adam(self.D.parameters(),2e-4,(0.5,0.999))
        self.device = device
        self.D_loss = DiscriminatorLoss()
        self.weigt_adversarial = weigt_adversarial
        self.data_augmentation = data_augmentation
    def save_checkpoint(self,path):
        torch.save(
            {
                'G_state_dict':self.G.state_dict(),
                'D_state_dict':self.D.state_dict(),
                'G_optim_state_dict':self.optim_G.state_dict(),
                'D_optim_state_dict':self.optim_D.state_dict()   
            },path
        )
    def load_checkpoint(self,path):
        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.optim_G.load_state_dict(checkpoint['G_optim_state_dict'])
        self.optim_D.load_state_dict(checkpoint['D_optim_state_dict'])
    def one_epoch(self,train_dataloader,mh_loss:nn.Module,epoch):
        patchGAN_fake = torch.from_numpy(np.zeros((6,1,14,14,14),dtype=np.float32)).to(self.device)
        patchGAN_true = torch.from_numpy(np.ones((6,1,14,14,14),dtype=np.float32)).to(self.device)
        running_loss = {}
        for i,data in enumerate(tqdm(train_dataloader)):
            inputs = data['img'].to(self.device)
            label = data['label'].to(self.device)
            if self.data_augmentation!=None:
                inputs,label = self.data_augmentation(inputs,label)
            self.optim_G.zero_grad()
            outputs = self.G(inputs)
            patchGAN_fake_output = self.D(outputs)
            loss_dict = mh_loss(outputs,inputs,label)
            loss_G = F.mse_loss(patchGAN_fake_output,patchGAN_true)
            loss_dict['loss_G']=loss_G
            loss_dict['loss'] = loss_dict['loss']+self.weigt_adversarial*loss_G
            loss_dict['loss'].backward()
            self.optim_G.step()
            patchGAN_fake_output = patchGAN_fake_output.detach()
            self.optim_D.zero_grad()
            loss_D = self.D_loss(self.D,inputs,label,patchGAN_fake_output,patchGAN_fake,patchGAN_true)
            loss_dict['loss_D']=loss_D
            loss_D.backward()
            self.optim_D.step()
            for key in loss_dict:
                if i==0:
                    running_loss[key]=loss_dict[key].item()
                else:
                    running_loss[key]+=loss_dict[key].item()
            if (i%100==0 and i!=0) or i==999:
                print('Epoch ' + str(epoch+1) + ', iter ' + str(i+1) + ':')
                for key in running_loss:
                    print(key + ' = ' + str(running_loss[key]/100))
                    running_loss[key]=0


