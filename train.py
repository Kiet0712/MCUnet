import tarfile
file = tarfile.open('/kaggle/input/brats-2021-task1/BraTS2021_Training_Data.tar')

file.extractall('./brain_images')
file.close()
import sys
path_code = ''
sys.path.append(path_code)
from torch.utils.data import DataLoader
import torch
from MHCRFB import MHCRFB as MHCRFB
from AttentionMHCRFB import AMHCRFB as AMHCRFB
from ClassBaseGuideAMHCRFB import CBGAMHCRFB as CBGAMHCRFB
from GAN3D import GAN3D as GAN3D
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
from dataset.dataset import BRATS
from utils.loss import MHLoss_1,MHLoss_2
from utils.DataAugmentationBlock import DataAugmenter
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
LOAD_CHECK_POINT = False
GAN_TRAINING = False
AUGMENTATION = False
checkpoint_path = ''
loss_choice = {
    'MHLoss_1': MHLoss_1,
    'MHLoss_2': MHLoss_2
}
loss_choice_str = ''
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
model_string = ''
csv_dir = ''
root_dir = ''
data_train = BRATS(
    csv_dir=csv_dir,
    mode='train',
    root_dir=root_dir
)
data_val = BRATS(
    csv_dir=csv_dir,
    mode='val',
    root_dir=root_dir
)
train_dataloader = DataLoader(data_train,1,True,num_workers=4,pin_memory=True)
val_dataloader = DataLoader(data_val,1,True,num_workers=4,pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_augmentation = None
if AUGMENTATION:
    data_augmentation = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).to(device)
if GAN_TRAINING:
    model = GAN3D(4,3,device,0.1,model_string,data_augmentation)
else:
    model = model_choice[model_string](4,3)
    model.to(device)
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
def calculate_metrics(predict,gt):
    labels = ["ET", "TC", "WT"]
    results = []
    for i, label in enumerate(labels):
        if np.sum(gt[i])==0:
            print('Remove sample')
            print('Non '  + label)
            return []
        preds_coords = np.argwhere(predict[i])
        targets_coords = np.argwhere(gt[i])
        haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
        tp = np.sum((predict[i]==1)&(gt[i]==1))
        tn = np.sum((predict[i]==0)&(gt[i]==0))
        fp = np.sum((predict[i]==1)&(gt[i]==0))
        fn = np.sum((predict[i]==0)&(gt[i]==1))
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        dice = 2*tp/(2*tp+fp+fn)
        results.append([
            haussdorf_dist,
            sens,
            spec,
            dice
        ])
    return results

PLOT= {
    'et':[],
    'tc':[],
    'wt':[]
}
def print_validation_result(et, tc, wt, name_metrics=["HDis      ", "Sens      ", "Spec      ", "Dice"]):
    """
    Prints the Validation result with corresponding name metrics for three numpy arrays.

    Args:
    et: A numpy array containing the first row of data.
    tc: A numpy array containing the second row of data.
    wt: A numpy array containing the third row of data.
    name_metrics: A list of strings containing the corresponding name metrics for each column.

    Returns:
    None
    """
    # Check if the number of columns and name metrics match
    if len(et) != len(tc) != len(wt) != len(name_metrics):
        raise ValueError("The number of columns and name metrics must be equal.")

    # Print the Validation result with row names
    print("Validation result:")
    print("     ", *name_metrics)  # Print header with metrics names
    print("ET:  ", *et)  # Print ET row with values
    print("TC:  ", *tc)  # Print TC row with values
    print("WT:  ", *wt)  # Print WT row with values
    PLOT['et'].append(et[3])
    PLOT['tc'].append(tc[3])
    PLOT['wt'].append(wt[3])
    plt.plot(PLOT['et'], linestyle='-', marker='o', color='r', label='ET') 
    plt.plot(PLOT['tc'], linestyle='-', marker='o', color='g', label='TC') 
    plt.plot(PLOT['wt'], linestyle='-', marker='o', color='b', label='WT') 
    plt.title('Dice score')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()
def validation(val_dataloader,model):
    result_metrics = []
    for i,data in enumerate(tqdm(val_dataloader)):
        inputs = data['img']
        crop_idx = data['crop_indices']
        inputs,pad = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.to(device)
        gt_seg_volume = data['label']
        with torch.inference_mode():
            outputs = model(inputs)
            predict_segment_volume = outputs['segment_volume']
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
    print('Validation result:')
    print_validation_result(mean_metrics_results[0],mean_metrics_results[1],mean_metrics_results[2])
loss_func = loss_choice[loss_choice_str](
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
if not GAN_TRAINING:
    optim = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,0.95)
if LOAD_CHECK_POINT:
    if not GAN_TRAINING:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        model.load_checkpoint(checkpoint_path)
def train(train_dataloader,model,loss_func,optim,epochs,save_each_epoch,checkpoint_save_path):
    for epoch in range(epochs):
        torch.backends.cudnn.benchmark = True
        if not GAN_TRAINING:
            running_loss = {}
            for i,data in enumerate(tqdm(train_dataloader)):
                inputs = data['img'].to(device)
                label = data['label'].to(device)
                if AUGMENTATION:
                    inputs,label = data_augmentation(inputs,label)
                optim.zero_grad()
                outputs = model(inputs)
                loss_cal = loss_func(outputs,inputs,label)
                loss_cal['loss'].backward()
                optim.step()
                for key in loss_cal:
                    if i==0:
                        running_loss[key]=loss_cal[key].item()
                    else:
                        running_loss[key]+=loss_cal[key].item()
                if (i%100==0 and i!=0) or i==999:
                    print('Epoch ' + str(epoch+1) + ', iter ' + str(i+1) + ':')
                    for key in running_loss:
                        print(key + ' = ' + str(running_loss[key]/100))
                        running_loss[key]=0
            if epoch%save_each_epoch==0:
                print('================================VALIDATION ' + str(epoch+1)+'================================')
                torch.backends.cudnn.benchmark = False
                validation(val_dataloader,model)
                torch.save(
                    {
                        'model_state_dict':model.state_dict(),
                        'optim_state_dict':optim.state_dict(),
                        'scheduler': scheduler.state_dict()
                    },checkpoint_save_path
                 )
            scheduler.step()
        else:
            model.one_epoch(train_dataloader,loss_func,epoch)
            if epoch%save_each_epoch==0:
                print('================================VALIDATION ' + str(epoch+1)+'================================')
                torch.backends.cudnn.benchmark = False
                validation(val_dataloader,model.G)
                model.save_checkpoint(checkpoint_save_path)
train(
    train_dataloader,
    model,
    loss_func,
    optim,
    4,
    2,
    'CHECKPOINT_' + model_string + datetime.now().strftime('%H-%M-%S_%d-%m-%Y') + '.pth'
)
import itk                                                                
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
def visualize(infor: list,mode: list):
    if mode[0]=='visualize_input_and_label':
        inputs = infor[0]
        label = infor[1]
        def visualize_input_and_label(layer):
            fig, axs = plt.subplots(2, 4,figsize=(15,8))
            #"ET", "TC", "WT"
            #'flair','t1','t1ce','t2'
            axs[0,0].imshow(inputs[0,:,:,layer],cmap='gray')
            axs[0,0].set_title('flair '+ str(layer))
            axs[0,1].imshow(inputs[1,:,:,layer],cmap='gray')
            axs[0,1].set_title('t1 ' + str(layer))
            axs[0,2].imshow(inputs[2,:,:,layer],cmap='gray')
            axs[0,2].set_title('t1ce ' + str(layer))
            axs[0,3].imshow(inputs[3,:,:,layer],cmap='gray')
            axs[0,3].set_title('t2 ' + str(layer))
            axs[1,0].imshow(label[0,:,:,layer],cmap='gray')
            axs[1,0].set_title('ET ' + str(layer))
            axs[1,1].imshow(label[1,:,:,layer],cmap='gray')
            axs[1,1].set_title('TC ' + str(layer))
            axs[1,2].imshow(label[2,:,:,layer],cmap='gray')
            axs[1,2].set_title('WT ' + str(layer))
            mask_label = np.zeros_like(label[0,:,:,layer],dtype=np.uint8)
            mask_label[np.where(label[0,:,:,layer]==1)[0],np.where(label[0,:,:,layer]==1)[1]] = 4
            mask_label[np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[0],np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[1]]=1
            mask_label[np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[0],np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[1]]=2
            axs[1,3].imshow(mask_label)
            axs[1,3].set_title('Mask label')
        interact(visualize_input_and_label,layer=(0,inputs.shape[3]-1))
    elif mode[0]=='visualize_input_label_output_model':
        if mode[1]=='one_head':
            inputs = infor[0]
            label = infor[1]
            outputs = infor[2]
            def one_head_visualize_input_label_output_model(layer):
                fig, axs = plt.subplots(3, 4,figsize=(15,12))
                axs[0,0].imshow(inputs[0,:,:,layer],cmap='gray')
                axs[0,0].set_title('flair '+ str(layer))
                axs[0,1].imshow(inputs[1,:,:,layer],cmap='gray')
                axs[0,1].set_title('t1 ' + str(layer))
                axs[0,2].imshow(inputs[2,:,:,layer],cmap='gray')
                axs[0,2].set_title('t1ce ' + str(layer))
                axs[0,3].imshow(inputs[3,:,:,layer],cmap='gray')
                axs[0,3].set_title('t2 ' + str(layer))
                axs[1,0].imshow(label[0,:,:,layer],cmap='gray')
                axs[1,0].set_title('ET ' + str(layer))
                axs[1,1].imshow(label[1,:,:,layer],cmap='gray')
                axs[1,1].set_title('TC ' + str(layer))
                axs[1,2].imshow(label[2,:,:,layer],cmap='gray')
                axs[1,2].set_title('WT ' + str(layer))
                mask_label = np.zeros_like(label[0,:,:,layer],dtype=np.uint8)
                mask_label[np.where(label[0,:,:,layer]==1)[0],np.where(label[0,:,:,layer]==1)[1]] = 4
                mask_label[np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[0],np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[1]]=1
                mask_label[np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[0],np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[1]]=2
                axs[1,3].imshow(mask_label)
                axs[1,3].set_title('Mask label ' + str(layer))
                axs[2,0].imshow(outputs[0,:,:,layer],cmap='gray')
                axs[2,0].set_title('ET predict' + str(layer))
                axs[2,1].imshow(outputs[1,:,:,layer],cmap='gray')
                axs[2,1].set_title('TC predict' + str(layer))
                axs[2,2].imshow(outputs[2,:,:,layer],cmap='gray')
                axs[2,2].set_title('WT predict' + str(layer))
                mask_label_predict = np.zeros_like(outputs[0,:,:,layer],dtype=np.uint8)
                mask_label_predict[np.where(outputs[0,:,:,layer]==1)[0],np.where(outputs[0,:,:,layer]==1)[1]] = 4
                mask_label_predict[np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[1]]=1
                mask_label_predict[np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[1]]=2
                axs[2,3].imshow(mask_label_predict)
                axs[2,3].set_title('Mask label predict ' + str(layer))
            interact(one_head_visualize_input_label_output_model,layer=(0,inputs.shape[3]-1))
        elif mode[1]=='multi_head':
            inputs = infor[0]
            label = infor[1]
            outputs = infor[2]['segment_volume'][0]
            reconstruct_volume = infor[2]['reconstruct_volume'][0]
            class_1_foreground_predict = infor[2]['class_1_foreground'][0]
            class_1_background_predict = infor[2]['class_1_background'][0]
            class_2_foreground_predict = infor[2]['class_2_foreground'][0]
            class_2_background_predict = infor[2]['class_2_background'][0]
            class_4_foreground_predict = infor[2]['class_4_foreground'][0]
            class_4_background_predict = infor[2]['class_4_background'][0]
            class_1_foreground_gt = inputs*label[1:2,:,:,:]
            class_1_background_gt = inputs*(1-label[1:2,:,:,:])
            class_2_foreground_gt = inputs*label[2:,:,:,:]
            class_2_background_gt = inputs*(1-label[2:,:,:,:])
            class_4_foreground_gt = inputs*label[0:1,:,:,:]
            class_4_background_gt = inputs*(1-label[0:1,:,:,:])
            dict_mask = {
                'class_1_foreground_predict':class_1_foreground_predict,
                'class_1_background_predict':class_1_background_predict,
                'class_2_foreground_predict':class_2_foreground_predict,
                'class_2_background_predict':class_2_background_predict,
                'class_4_foreground_predict':class_4_foreground_predict,
                'class_4_background_predict':class_4_background_predict,
                'class_1_foreground_gt':class_1_foreground_gt,
                'class_1_background_gt':class_1_background_gt,
                'class_2_foreground_gt':class_2_foreground_gt,
                'class_2_background_gt':class_2_background_gt,
                'class_4_foreground_gt':class_4_foreground_gt,
                'class_4_background_gt':class_4_background_gt
            }
            def multi_head_visualize_input_label_output_model(layer):
                fig, axs = plt.subplots(16, 4,figsize=(15,24))
                axs[0,0].imshow(inputs[0,:,:,layer],cmap='gray')
                axs[0,0].set_title('flair '+ str(layer))
                axs[0,1].imshow(inputs[1,:,:,layer],cmap='gray')
                axs[0,1].set_title('t1 ' + str(layer))
                axs[0,2].imshow(inputs[2,:,:,layer],cmap='gray')
                axs[0,2].set_title('t1ce ' + str(layer))
                axs[0,3].imshow(inputs[3,:,:,layer],cmap='gray')
                axs[0,3].set_title('t2 ' + str(layer))
                axs[1,0].imshow(reconstruct_volume[0,:,:,layer],cmap='gray')
                axs[1,0].set_title('flair reconstruct '+ str(layer))
                axs[1,1].imshow(reconstruct_volume[1,:,:,layer],cmap='gray')
                axs[1,1].set_title('t1 reconstruct ' + str(layer))
                axs[1,2].imshow(reconstruct_volume[2,:,:,layer],cmap='gray')
                axs[1,2].set_title('t1ce reconstruct ' + str(layer))
                axs[1,3].imshow(reconstruct_volume[3,:,:,layer],cmap='gray')
                axs[1,3].set_title('t2 reconstruct ' + str(layer))
                axs[2,0].imshow(label[0,:,:,layer],cmap='gray')
                axs[2,0].set_title('ET ' + str(layer))
                axs[2,1].imshow(label[1,:,:,layer],cmap='gray')
                axs[2,1].set_title('TC ' + str(layer))
                axs[2,2].imshow(label[2,:,:,layer],cmap='gray')
                axs[2,2].set_title('WT ' + str(layer))
                mask_label = np.zeros_like(label[0,:,:,layer],dtype=np.uint8)
                mask_label[np.where(label[0,:,:,layer]==1)[0],np.where(label[0,:,:,layer]==1)[1]] = 4
                mask_label[np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[0],np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[1]]=1
                mask_label[np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[0],np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[1]]=2
                axs[2,3].imshow(mask_label)
                axs[2,3].set_title('Mask label ' + str(layer))
                axs[3,0].imshow(outputs[0,:,:,layer],cmap='gray')
                axs[3,0].set_title('ET predict' + str(layer))
                axs[3,1].imshow(outputs[1,:,:,layer],cmap='gray')
                axs[3,1].set_title('TC predict' + str(layer))
                axs[3,2].imshow(outputs[2,:,:,layer],cmap='gray')
                axs[3,2].set_title('WT predict' + str(layer))
                mask_label_predict = np.zeros_like(outputs[0,:,:,layer],dtype=np.uint8)
                mask_label_predict[np.where(outputs[0,:,:,layer]==1)[0],np.where(outputs[0,:,:,layer]==1)[1]] = 4
                mask_label_predict[np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[1]]=1
                mask_label_predict[np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[1]]=2
                axs[3,3].imshow(mask_label_predict)
                axs[3,3].set_title('Mask label predict ' + str(layer))
                count = 4
                mask_value_label = {
                    '1':'TC',
                    '2':'WT',
                    '4':'ET'
                }
                for class_value in ['1','2','4']:
                    base_name = 'class_'
                    for type_mask in ['foreground,background']:
                        for type_plot in ['gt,predict']:
                            name_obj = base_name+class_value+'_'+type_mask+'_'+type_plot
                            axs[count,0].imshow(dict_mask[name_obj][0,:,:,layer],cmap='gray')
                            axs[count,0].set_title('flair class ' + mask_value_label[class_value] + ' ' + type_mask + ' ' + type_plot + str(layer)) 
                            axs[count,1].imshow(dict_mask[name_obj][1,:,:,layer],cmap='gray')
                            axs[count,1].set_title('t1 class ' + mask_value_label[class_value] + ' ' + type_mask + ' ' + type_plot + str(layer)) 
                            axs[count,2].imshow(dict_mask[name_obj][2,:,:,layer],cmap='gray')
                            axs[count,2].set_title('t1ce class ' + mask_value_label[class_value] + ' ' + type_mask + ' ' + type_plot + str(layer)) 
                            axs[count,3].imshow(dict_mask[name_obj][3,:,:,layer],cmap='gray')
                            axs[count,3].set_title('t2 class ' + mask_value_label[class_value] + ' ' + type_mask + ' ' + type_plot + str(layer)) 
                            count+=1
        else:
            pass
    else:
        pass