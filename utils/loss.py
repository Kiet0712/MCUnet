import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        sizep = pred.size(0)
        sizet = target.size(0)
        pred_flat = pred.view(sizep, -1)
        target_flat = target.view(sizet, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = target.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/ (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
class BceDiceLoss(nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred,target)
        diceloss = self.dice(pred,target)

        loss = diceloss + bceloss

        return {'loss':loss}
class MHLoss_SELF_GUIDE(nn.Module):
    def __init__(self,lamda_list):
        super().__init__()
        self.lamda_list = lamda_list
        self.dicebce = BceDiceLoss()
    def forward(self,data,volume,gt_volume):
        segment_volume = data['segment_volume']
        reconstruct_volume = data['reconstruct_volume']
        class_1_foreground = data['class_1_foreground']
        class_2_foreground = data['class_2_foreground']
        class_4_foreground = data['class_4_foreground']
        class_1_background = data['class_1_background']
        class_2_background = data['class_2_background']
        class_4_background = data['class_4_background']


        #calculate_main_loss
        segment_volume_loss = self.dicebce(segment_volume,gt_volume)['loss']
        reconstruct_volume_loss = F.l1_loss(reconstruct_volume,volume)
        class_1_foreground_loss = F.l1_loss(class_1_foreground,volume*gt_volume[:,1:2,:,:,:])
        class_2_foreground_loss = F.l1_loss(class_2_foreground,volume*gt_volume[:,2:,:,:,:])
        class_4_foreground_loss = F.l1_loss(class_4_foreground,volume*gt_volume[:,0:1,:,:,:])
        class_1_background_loss = F.l1_loss(class_1_background,volume*(1-gt_volume[:,1:2,:,:,:]))
        class_2_background_loss = F.l1_loss(class_2_background,volume*(1-gt_volume[:,2:,:,:,:]))
        class_4_background_loss = F.l1_loss(class_4_background,volume*(1-gt_volume[:,0:1,:,:,:]))
        #calculate_guide_loss
        reconstruct_guide_loss = F.l1_loss(reconstruct_volume,class_1_background+class_1_foreground)+\
                                 F.l1_loss(reconstruct_volume,class_2_background+class_2_foreground)+\
                                 F.l1_loss(reconstruct_volume,class_4_background+class_4_foreground)
        class_1_background_guide_loss = F.l1_loss(class_1_background,reconstruct_volume*(1-segment_volume[:,1:2,:,:,:]))
        class_1_foreground_guide_loss = F.l1_loss(class_1_foreground,reconstruct_volume*segment_volume[:,1:2,:,:,:])
        class_2_background_guide_loss = F.l1_loss(class_2_background,reconstruct_volume*(1-segment_volume[:,2:,:,:,:]))
        class_2_foreground_guide_loss = F.l1_loss(class_2_foreground,reconstruct_volume*segment_volume[:,2:,:,:,:])
        class_4_background_guide_loss = F.l1_loss(class_4_background,reconstruct_volume*(1-segment_volume[:,0:1,:,:,:]))
        class_4_foreground_guide_loss = F.l1_loss(class_4_foreground,reconstruct_volume*segment_volume[:,0:1,:,:,:])

        loss = self.lamda_list['segment_volume_loss']*segment_volume_loss+\
               self.lamda_list['reconstruct_volume_loss']*reconstruct_volume_loss+\
               self.lamda_list['class_1_foreground_loss']*class_1_foreground_loss+\
               self.lamda_list['class_2_foreground_loss']*class_2_foreground_loss+\
               self.lamda_list['class_4_foreground_loss']*class_4_foreground_loss+\
               self.lamda_list['class_1_background_loss']*class_1_background_loss+\
               self.lamda_list['class_2_background_loss']*class_2_background_loss+\
               self.lamda_list['class_4_background_loss']*class_4_background_loss+\
               self.lamda_list['reconstruct_guide_loss']*reconstruct_guide_loss+\
               self.lamda_list['class_1_background_guide_loss']*class_1_background_guide_loss+\
               self.lamda_list['class_1_foreground_guide_loss']*class_1_foreground_guide_loss+\
               self.lamda_list['class_2_background_guide_loss']*class_2_background_guide_loss+\
               self.lamda_list['class_2_foreground_guide_loss']*class_2_foreground_guide_loss+\
               self.lamda_list['class_4_background_guide_loss']*class_4_background_guide_loss+\
               self.lamda_list['class_4_foreground_guide_loss']*class_4_foreground_guide_loss
        return {
            'loss': loss,
            'segment_volume_loss':segment_volume_loss,
            'reconstruct_volume_loss':reconstruct_volume_loss,
            'class_1_foreground_loss':class_1_foreground_loss,
            'class_2_foreground_loss':class_2_foreground_loss,
            'class_4_foreground_loss':class_4_foreground_loss,
            'class_1_background_loss':class_1_background_loss,
            'class_2_background_loss':class_2_background_loss,
            'class_4_background_loss':class_4_background_loss,
            'reconstruct_guide_loss':reconstruct_guide_loss,
            'class_1_background_guide_loss':class_1_background_guide_loss,
            'class_1_foreground_guide_loss':class_1_foreground_guide_loss,
            'class_2_background_guide_loss':class_2_background_guide_loss,
            'class_2_foreground_guide_loss':class_2_foreground_guide_loss,
            'class_4_background_guide_loss':class_4_background_guide_loss,
            'class_4_foreground_guide_loss':class_4_foreground_guide_loss
        }
class MHLoss(nn.Module):
    def __init__(self,lamda_list):
        super().__init__()
        self.lamda_list = lamda_list
        self.dicebce = BceDiceLoss()
    def forward(self,data,volume,gt_volume):
        segment_volume = data['segment_volume']
        reconstruct_volume = data['reconstruct_volume']
        class_1_foreground = data['class_1_foreground']
        class_2_foreground = data['class_2_foreground']
        class_4_foreground = data['class_4_foreground']
        class_1_background = data['class_1_background']
        class_2_background = data['class_2_background']
        class_4_background = data['class_4_background']


        #calculate_main_loss
        segment_volume_loss = self.dicebce(segment_volume,gt_volume)['loss']
        reconstruct_volume_loss = F.l1_loss(reconstruct_volume,volume)
        class_1_foreground_loss = F.l1_loss(class_1_foreground,volume*gt_volume[:,1:2,:,:,:])
        class_2_foreground_loss = F.l1_loss(class_2_foreground,volume*gt_volume[:,2:,:,:,:])
        class_4_foreground_loss = F.l1_loss(class_4_foreground,volume*gt_volume[:,0:1,:,:,:])
        class_1_background_loss = F.l1_loss(class_1_background,volume*(1-gt_volume[:,1:2,:,:,:]))
        class_2_background_loss = F.l1_loss(class_2_background,volume*(1-gt_volume[:,2:,:,:,:]))
        class_4_background_loss = F.l1_loss(class_4_background,volume*(1-gt_volume[:,0:1,:,:,:]))

        loss = self.lamda_list['segment_volume_loss']*segment_volume_loss+\
               self.lamda_list['reconstruct_volume_loss']*reconstruct_volume_loss+\
               self.lamda_list['class_1_foreground_loss']*class_1_foreground_loss+\
               self.lamda_list['class_2_foreground_loss']*class_2_foreground_loss+\
               self.lamda_list['class_4_foreground_loss']*class_4_foreground_loss+\
               self.lamda_list['class_1_background_loss']*class_1_background_loss+\
               self.lamda_list['class_2_background_loss']*class_2_background_loss+\
               self.lamda_list['class_4_background_loss']*class_4_background_loss
        return {
            'loss': loss,
            'segment_volume_loss':segment_volume_loss,
            'reconstruct_volume_loss':reconstruct_volume_loss,
            'class_1_foreground_loss':class_1_foreground_loss,
            'class_2_foreground_loss':class_2_foreground_loss,
            'class_4_foreground_loss':class_4_foreground_loss,
            'class_1_background_loss':class_1_background_loss,
            'class_2_background_loss':class_2_background_loss,
            'class_4_background_loss':class_4_background_loss
        }