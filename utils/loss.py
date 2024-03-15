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

    def forward(self, pred, target,something=None):
        bceloss = self.bce(pred,target)
        diceloss = self.dice(pred,target)

        loss = diceloss + bceloss

        return {'loss':loss}
class MHLoss_SELF_GUIDE(nn.Module):
    def __init__(self,lamda_list,n_classes):
        super().__init__()
        self.lamda_list = lamda_list
        self.n_classes = n_classes
        self.dicebce = BceDiceLoss()
    def forward(self,data,gt_volume,volume):
        segment_volume = data['segment_volume']
        reconstruct_volume = data['reconstruct_volume']
        segment_volume_loss = self.dicebce(segment_volume,gt_volume)['loss']
        reconstruct_volume_loss = F.l1_loss(reconstruct_volume,volume)
        loss = {
            'seg_vol_loss':segment_volume_loss,
            'recstr_vol_loss':reconstruct_volume_loss,
        }
        reconstruct_guide_loss = 0
        for i in range(self.n_classes):
            str_class = 'class_' + str(i+1) + '_'
            short_str = 'c'+str(i+1)+'_'
            foreground_mask = data[str_class+'foreground']
            background_mask = data[str_class+'background']
            loss[short_str+'fg_loss'] = F.l1_loss(foreground_mask,volume*gt_volume[:,i:(i+1)])
            loss[short_str+'bg_loss'] = F.l1_loss(background_mask,volume*(1-gt_volume[:,i:(i+1)]))
            loss[short_str+'fg_guide_loss'] = F.l1_loss(foreground_mask,reconstruct_volume*segment_volume[:,i:(i+1)])
            loss[short_str+'bg_guide_loss'] = F.l1_loss(background_mask,reconstruct_volume*(1-segment_volume[:,i:(i+1)]))
            reconstruct_guide_loss+=F.l1_loss(reconstruct_volume,foreground_mask+data[str_class+'background'])

        loss['recstr_guide_loss'] = reconstruct_guide_loss
        sum_loss = 0
        for key in loss:
            sum_loss+=loss[key]*self.lamda_list[key]
        loss['loss']=sum_loss
        return loss
class MHLoss(nn.Module):
    def __init__(self,lamda_list,n_classes):
        super().__init__()
        self.lamda_list = lamda_list
        self.n_classes = n_classes
        self.dicebce = BceDiceLoss()
    def forward(self,data,gt_volume,volume):
        segment_volume = data['segment_volume']
        reconstruct_volume = data['reconstruct_volume']
        segment_volume_loss = self.dicebce(segment_volume,gt_volume)['loss']
        reconstruct_volume_loss = F.l1_loss(reconstruct_volume,volume)
        loss = {
            'seg_vol_loss':segment_volume_loss,
            'recstr_vol_loss':reconstruct_volume_loss,
        }
        for i in range(self.n_classes):
            str_class = 'class_' + str(i+1) + '_'
            short_str = 'c'+str(i+1)+'_'
            foreground_mask = data[str_class+'foreground']
            background_mask = data[str_class+'background']
            loss[short_str+'fg_loss'] = F.l1_loss(foreground_mask,volume*gt_volume[:,i:(i+1)])
            loss[short_str+'bg_loss'] = F.l1_loss(background_mask,volume*(1-gt_volume[:,i:(i+1)]))
        sum_loss = 0
        for key in loss:
            sum_loss+=loss[key]*self.lamda_list[key]
        loss['loss']=sum_loss
        return loss