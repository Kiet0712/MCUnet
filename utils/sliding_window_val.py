from monai.inferers import sliding_window_inference
import torch
import numpy as np
from tqdm.auto import tqdm
from batch_utils import pad_batch1_to_compatible_size
from metric import calculate_metrics
class Wrapper:
  def __init__(self,model,wrapper_type):
    self.model = model
    self.wrapper_type = wrapper_type
  def __call__(self, x):
    if self.wrapper_type=='normal':
       return self.model(x)['segment_volume']
    elif self.wrapper_type=='foreground_cover':
       output = self.model(x)
       segment_volume = output['segment_volume']
       class_1_foreground = torch.mean(output['class_1_foreground'],dim=1,keepdim=True)
       class_2_foreground = torch.mean(output['class_2_foreground'],dim=1,keepdim=True)
       class_4_foreground = torch.mean(output['class_4_foreground'],dim=1,keepdim=True)
       foreground = (torch.cat([class_4_foreground,class_1_foreground,class_2_foreground],dim=1)>=0.01).type(segment_volume.dtype)
       return segment_volume*foreground

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
def validation_sliding_window(val_dataloader,model,device,wrapper_type: str):
    model = Wrapper(model,wrapper_type)
    result_metrics = []
    for i,data in enumerate(tqdm(val_dataloader)):
        inputs = data['img']
        crop_idx = data['crop_indices']
        inputs,pad = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.to(device)
        gt_seg_volume = data['label']
        with torch.inference_mode():
            predict_segment_volume = sliding_window_inference(inputs,roi_size=[128,128,128],sw_batch_size=1,overlap=0.7,predictor=model)
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