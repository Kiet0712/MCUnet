from monai.inferers import sliding_window_inference
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from metric import calculate_metrics
from loss import MHLoss,MHLoss_SELF_GUIDE,BceDiceLoss
import matplotlib.pyplot as plt
def pad_batch1_to_compatible_size(batch):
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(2), ymax - batch.size(3), xmax - batch.size(4)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)

def make_optimizers(cfg,model):
    if cfg.SOLVER.OPTIMIZER=='adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
        return optimizer
    elif cfg.SOLVER.OPTIMIZER=='adamw':
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
        return optimizer
    else:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            nesterov=cfg.SOLVER.NESTEROV,
            momentum=cfg.SOLVER.SGD_MOMENTUM
        )
        return optimizer
def make_scheduler(cfg,optimizer):
    if cfg.SOLVER.SCHEDULER == "LambdaLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda= lambda epoch: 1 if epoch >= 1 and epoch < 5
                                else 5 if epoch >= 5 and epoch < 10
                                else 10 if epoch >= 10 and epoch < 20
                                else 20 if epoch >= 20 and epoch < 25
                                else 10 if epoch >= 25 and epoch < 30
                                else 5 if epoch >= 30 and epoch < 40
                                else 1 if epoch >= 40 and epoch < 80
                                else 0.1 if epoch >= 80 and epoch < 90
                                else 0.01
        )
        return scheduler
    elif cfg.SOLVER.SCHEDULER == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            epochs=cfg.SOLVER.MAX_EPOCHS,
            steps_per_epoch=1000,
            pct_start=cfg.SOLVER.PCT_START,
            div_factor=cfg.SOLVER.DIV_FACTOR,
            max_lr=1e-3
        )
        return scheduler
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.95
        )
        return scheduler
def make_loss_function(cfg):
    if cfg.MODEL.MULTIHEAD_OUTPUT:
        if cfg.SELF_GUIDE_LOSS:
            return MHLoss_SELF_GUIDE(
                    {
                        'segment_volume_loss':5,
                        'reconstruct_volume_loss':2,
                        'class_1_foreground_loss':2,
                        'class_2_foreground_loss':2,
                        'class_4_foreground_loss':2,
                        'class_1_background_loss':2,
                        'class_2_background_loss':2,
                        'class_4_background_loss':2,
                        'reconstruct_guide_loss':0.75,
                        'class_1_background_guide_loss':0.75,
                        'class_1_foreground_guide_loss':0.75,
                        'class_2_background_guide_loss':0.75,
                        'class_2_foreground_guide_loss':0.75,
                        'class_4_background_guide_loss':0.75,
                        'class_4_foreground_guide_loss':0.75
                    }
            )
        else:
            return MHLoss(
                {
                        'segment_volume_loss':5,
                        'reconstruct_volume_loss':2,
                        'class_1_foreground_loss':2,
                        'class_2_foreground_loss':2,
                        'class_4_foreground_loss':2,
                        'class_1_background_loss':2,
                        'class_2_background_loss':2,
                        'class_4_background_loss':2,
                        'reconstruct_guide_loss':0.75,
                        'class_1_background_guide_loss':0.75,
                        'class_1_foreground_guide_loss':0.75,
                        'class_2_background_guide_loss':0.75,
                        'class_2_foreground_guide_loss':0.75,
                        'class_4_background_guide_loss':0.75,
                        'class_4_foreground_guide_loss':0.75
                }
            )
    else:
        return BceDiceLoss()
def validation_sliding_windown(cfg,model,val_dataloader,device):
    if cfg.MODEL.MULTIHEAD_OUTPUT:
        class Wrapper:
            def __init__(self,model):
                self.model = model
            def __call__(self, x):
                return self.model(x)['segment_volume']
        model = Wrapper(model)
    result_metrics = []
    for i,data in enumerate(tqdm(val_dataloader)):
        inputs = data['img']
        crop_idx = data['crop_indices']
        inputs,pad = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.to(device)
        gt_seg_volume = data['label']
        with torch.inference_mode():
            predict_segment_volume = sliding_window_inference(
                inputs=inputs,
                roi_size=cfg.ROI_SIZE,
                sw_batch_size=cfg.SW_BATCHSIZE,
                overlap=cfg.OVERLAP,
                predictor=model
            )
        maxz,maxy,maxx = predict_segment_volume.size(2)-pad[0],predict_segment_volume.size(3)-pad[1],predict_segment_volume.size(4)-pad[2]
        predict_segment_volume = predict_segment_volume[:,:,0:maxz,0:maxy,0:maxx].cpu().numpy()
        predict_seg_volume = np.zeros(gt_seg_volume.shape)
        predict_seg_volume[:,:,slice(*crop_idx[0]),slice(*crop_idx[1]),slice(*crop_idx[2])] = predict_segment_volume
        segs = (predict_seg_volume[0]>0.5).astype('float32')
        patient_metrics_result = calculate_metrics(segs,gt_seg_volume[0].numpy()) 
        if len(patient_metrics_result)!=0:
            result_metrics.append(patient_metrics_result)
    result_metrics = np.array(result_metrics)
    mean_metrics_results = np.mean(result_metrics,axis=0)
    return mean_metrics_results
def validation_normal(cfg,model,val_dataloader,device):
    if cfg.MODEL.MULTIHEAD_OUTPUT:
        class Wrapper:
            def __init__(self,model):
                self.model = model
            def __call__(self, x):
                return self.model(x)['segment_volume']
        model = Wrapper(model)
    result_metrics = []
    for i,data in enumerate(tqdm(val_dataloader)):
        inputs = data['img']
        crop_idx = data['crop_indices']
        inputs,pad = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.to(device)
        gt_seg_volume = data['label']
        with torch.inference_mode():
            predict_segment_volume = model(inputs)
        maxz,maxy,maxx = predict_segment_volume.size(2)-pad[0],predict_segment_volume.size(3)-pad[1],predict_segment_volume.size(4)-pad[2]
        predict_segment_volume = predict_segment_volume[:,:,0:maxz,0:maxy,0:maxx].cpu().numpy()
        predict_seg_volume = np.zeros(gt_seg_volume.shape)
        predict_seg_volume[:,:,slice(*crop_idx[0]),slice(*crop_idx[1]),slice(*crop_idx[2])] = predict_segment_volume
        segs = (predict_seg_volume[0]>0.5).astype('float32')
        patient_metrics_result = calculate_metrics(segs,gt_seg_volume[0].numpy()) 
        if len(patient_metrics_result)!=0:
            result_metrics.append(patient_metrics_result)
    result_metrics = np.array(result_metrics)
    mean_metrics_results = np.mean(result_metrics,axis=0)
    return mean_metrics_results
def validation(cfg,model,val_dataloader,device):
    if cfg.VALIDATION_TYPE=='normal':
        return validation_normal(cfg,model,val_dataloader,device)
    else:
        return validation_sliding_windown(cfg,model,val_dataloader,device)
def update_PLOT(PLOT,result):
    et,tc,wt = result[0],result[1],result[2]
    for i in range(4):
        PLOT['et'][i].append(et[i])
        PLOT['tc'][i].append(tc[i])
        PLOT['wt'][i].append(wt[i])
    return PLOT
def PLOT_RESULT_GRAPH(PLOT):
    figure, axis = plt.subplots(2, 2)
    mapping_dict = {
        0:'Hausdorff Distance',
        1:'Sensitivity',
        2:'Specificity',
        3:'Dice Score'
    }
    for i in range(4):
        x = int(i%2)
        y = int((i-x)%2)
        axis[y,x].plot(PLOT['et'][i], linestyle='-', marker='o', color='r', label='ET')
        axis[y,x].plot(PLOT['tc'][i], linestyle='-', marker='o', color='r', label='TC')
        axis[y,x].plot(PLOT['wt'][i], linestyle='-', marker='o', color='r', label='WT')
        axis[0, 0].set_title(mapping_dict[i])
    plt.legend()
    plt.show()
