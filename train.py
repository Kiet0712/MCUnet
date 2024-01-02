import tarfile
file = tarfile.open('/kaggle/input/brats-2021-task1/BraTS2021_Training_Data.tar')

file.extractall('./brain_images')
file.close()
import sys
path_code = ''
sys.path.append(path_code)
from torch.utils.data import DataLoader
import torch
from MHCRFBDMPUnet3D import MHCRFBDMPUnet3D as MHCRFBDMPUnet3D
from AttentionMHCRFBDMPUnet3D import AMHCRFBDMPUnet3D as AMHCRFBDMPUnet3D
from ClassBaseGuideAMHCRFBDMPUnet3D import CBGAMHCRFBDMPUnet3D as CBGAMHCRFBDMPUnet3D
from dataset.dataset import BRATS
from utils.loss import MHLoss_1
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from datetime import datetime
LOAD_CHECK_POINT = False
checkpoint_path = ''
model_choice = {
    'MHCRFBDMPUnet3D':MHCRFBDMPUnet3D,
    'AMHCRFBDMPUnet3D':AMHCRFBDMPUnet3D,
    'CBGAMHCRFBDMPUnet3D':CBGAMHCRFBDMPUnet3D
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
train_dataloader = DataLoader(data_train,1,True)
val_dataloader = DataLoader(data_val,1,True)
model = model_choice[model_string](4,3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def validation(val_dataloader,model):
    result_metrics = []
    for i,data in enumerate(val_dataloader):
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
def train(train_dataloader,model,loss_func,optim,epochs,save_each_epoch,checkpoint_save_path):
    for epoch in range(epochs):
        running_loss = {}
        for i,data in enumerate(train_dataloader):
            inputs = data['img'].to(device)
            label = data['label'].to(device)
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
            if i%100==0 and i!=0:
                print('Epoch ' + str(epoch+1) + ', iter ' + str(i+1) + ':')
                for key in running_loss:
                    print(key + ' = ' + str(running_loss[key]/100))
                    running_loss[key]=0
        if epoch%save_each_epoch==0:
            print('================================VALIDATION ' + str(epoch+1)+'================================')
            validation(val_dataloader,model)
            torch.save(
                {
                    'model_state_dict':model.state_dict(),
                    'optim_state_dict':optim.state_dict()
                },checkpoint_save_path
            )
optim = torch.optim.Adam(model.parameters(),lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,0.95)
loss_func = MHLoss_1(
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
if LOAD_CHECK_POINT:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
train(
    train_dataloader,
    model,
    loss_func,
    optim,
    4,
    2,
    'CHECKPOINT_' + model_string + datetime.now().strftime('%H-%M-%S_%d-%m-%Y') + '.pth'
)
