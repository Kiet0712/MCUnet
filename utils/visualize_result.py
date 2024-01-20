import itk                                                                
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import numpy as np
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
            mask_label[np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[0],np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[1]]=2
            mask_label[np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[0],np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[1]]=1
            mask_label[np.where(label[0,:,:,layer]==1)[0],np.where(label[0,:,:,layer]==1)[1]] = 4
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
                mask_label[np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[0],np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[1]]=2
                mask_label[np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[0],np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[1]]=1
                mask_label[np.where(label[0,:,:,layer]==1)[0],np.where(label[0,:,:,layer]==1)[1]] = 4
                axs[1,3].imshow(mask_label)
                axs[1,3].set_title('Mask label ' + str(layer))
                axs[2,0].imshow(outputs[0,:,:,layer],cmap='gray')
                axs[2,0].set_title('ET predict' + str(layer))
                axs[2,1].imshow(outputs[1,:,:,layer],cmap='gray')
                axs[2,1].set_title('TC predict' + str(layer))
                axs[2,2].imshow(outputs[2,:,:,layer],cmap='gray')
                axs[2,2].set_title('WT predict' + str(layer))
                mask_label_predict = np.zeros_like(outputs[0,:,:,layer],dtype=np.uint8)
                mask_label_predict[np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[1]]=2
                mask_label_predict[np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[1]]=1
                mask_label_predict[np.where(outputs[0,:,:,layer]==1)[0],np.where(outputs[0,:,:,layer]==1)[1]] = 4
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
                mask_label[np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[0],np.where(np.logical_and(label[2,:,:,layer]==1, np.logical_not(label[1,:,:,layer]==1)))[1]]=2
                mask_label[np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[0],np.where(np.logical_and(label[1,:,:,layer]==1, np.logical_not(label[0,:,:,layer]==1)))[1]]=1
                mask_label[np.where(label[0,:,:,layer]==1)[0],np.where(label[0,:,:,layer]==1)[1]] = 4
                axs[2,3].imshow(mask_label)
                axs[2,3].set_title('Mask label ' + str(layer))
                axs[3,0].imshow(outputs[0,:,:,layer],cmap='gray')
                axs[3,0].set_title('ET predict' + str(layer))
                axs[3,1].imshow(outputs[1,:,:,layer],cmap='gray')
                axs[3,1].set_title('TC predict' + str(layer))
                axs[3,2].imshow(outputs[2,:,:,layer],cmap='gray')
                axs[3,2].set_title('WT predict' + str(layer))
                mask_label_predict = np.zeros_like(outputs[0,:,:,layer],dtype=np.uint8)
                mask_label_predict[np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[2,:,:,layer]==1, np.logical_not(outputs[1,:,:,layer]==1)))[1]]=2
                mask_label_predict[np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[0],np.where(np.logical_and(outputs[1,:,:,layer]==1, np.logical_not(outputs[0,:,:,layer]==1)))[1]]=1
                mask_label_predict[np.where(outputs[0,:,:,layer]==1)[0],np.where(outputs[0,:,:,layer]==1)[1]] = 4
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
                    for type_mask in ['foreground','background']:
                        for type_plot in ['gt','predict']:
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
            interact(multi_head_visualize_input_label_output_model,layer=(0,inputs.shape[3]-1))
        else:
            pass
    else:
        pass