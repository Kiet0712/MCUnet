import sys
path_code = ''
sys.path.append(path_code)
from config.defaults import _C as cfg
from dataset.dataset import make_dataloader
from utils.DataAugmentationBlock import DataAugmenter
from src.model_main import make_model
from utils.utils_main import *
import torch
from tqdm.auto import tqdm
import tarfile
import pandas as pd
cfg.DATASET.CSV_DIR = ''
cfg.DATASET.ROOT_DIR = ''
cfg.LOG_PATH = ''
def myprint(x,log_to_screen = cfg.LOG_TO_SCREEN):
    with open(cfg.LOG_PATH,"a") as f:
        print(x,file = f)
    if log_to_screen:
        print(x,file = sys.stdout)
def train(cfg,device):
    train_dataloader = make_dataloader(cfg,"train")
    val_dataloader = make_dataloader(cfg,"val")
    if cfg.DATASET.AUGMENTATION:
        data_augmentation = DataAugmenter(
            p=cfg.DATASET.AUGMENTATION_PROBA,
            noise_only=cfg.DATASET.AUGMENTATION_NOISE_ONLY,
            channel_shuffling=cfg.DATASET.AUGMENTATION_CHANNEL_SHUFFLE,
            drop_channnel=cfg.DATASET.AUGMENTATION_DROP_CHANNEL
        ).to(device)
    model = make_model(cfg).to(device)
    optim = make_optimizers(cfg,model)
    scheduler = make_scheduler(cfg,optim)
    PLOT = {
        'et':[[],[],[],[]],
        'tc':[[],[],[],[]],
        'wt':[[],[],[],[]]
    }
    current_epoch = 1
    loss_func = make_loss_function(cfg)
    if cfg.CHECKPOINT.LOAD:
        checkpoint = torch.load(cfg.CHECKPOINT.PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        PLOT = checkpoint['plot']
        current_epoch = checkpoint['current_epoch']+1
        result_plot = np.stack([np.array(PLOT['et']),np.array(PLOT['tc']),np.array(PLOT['wt'])],axis=0)
        result_plot_mean_by_class = np.mean(result_plot,axis=0)
        best_epoch_result = np.argmax(result_plot_mean_by_class[-1,:])
        best_result = result_plot[:,:,best_epoch_result]
        myprint('CHECKPOINT_LOADING FROM ' + str(cfg.CHECKPOINT.PATH),True)
        myprint('--------------------------------------------BEST_RESULT--------------------------------------------',True)
        myprint('--------------------------------------------EPOCH ' + str(best_epoch_result+1)+'--------------------------------------------',True)
        myprint(pd.DataFrame(best_result,columns = ['Hausdorff Distance','Sensitivity','Specificity','Dice Score'],index = ['ET','TC','WT']),True)
        PLOT_RESULT_GRAPH(PLOT)
        myprint('Current epoch = ' + str(current_epoch),True)
    for epoch in range(current_epoch,cfg.SOLVER.MAX_EPOCHS+1):
        torch.backends.cudnn.benchmark = True
        running_loss = {}
        loop = tqdm(enumerate(train_dataloader),total=len(train_dataloader),leave=False)
        for i,data in loop:
            loop.set_description(f"Epoch [{epoch}/{cfg.SOLVER.MAX_EPOCHS}]")
            inputs = data['img'].to(device)
            label = data['label'].to(device)
            if cfg.DATASET.AUGMENTATION:
                inputs,label = data_augmentation(inputs,label)
            optim.zero_grad()
            outputs = model(inputs)
            loss_cal = loss_func(outputs,label,inputs)
            loss_cal['loss'].backward()
            optim.step()
            string_loss = ''
            for key in loss_cal:
                if key=='loss':
                    string_loss+=key + f' = {loss_cal[key].item(): .4f}, '
                else:
                    string_loss+= key[:-5] + f' = {loss_cal[key].item(): .4f}, '
                if i==0:
                    running_loss[key]=loss_cal[key].item()
                else:
                    running_loss[key]+=loss_cal[key].item()
            string_loss = string_loss[:-2]
            loop.set_postfix_str(s=string_loss)
            if i % cfg.SOLVER.PRINT_RESULT_INTERVAL == 0 and i != 0:
                myprint('# ---------------------------------------------------------------------------- #')
                myprint('Epoch ' + str(epoch) + ', iter ' + str(i+1) + ':')
                for key in running_loss:
                    running_loss[key] = running_loss[key]/cfg.SOLVER.PRINT_RESULT_INTERVAL
                myprint(pd.DataFrame([running_loss]).transpose())
                for key in running_loss:
                    running_loss[key]=0
        scheduler.step()
        if epoch % cfg.SOLVER.EVAL_EPOCH_INTERVAL == 0:
            myprint("--------------------------------VALIDATION EPOCH " + str(epoch) + "--------------------------------",True)
            torch.backends.cudnn.benchmark = False
            result = validation(cfg,model,val_dataloader,device)
            myprint(pd.DataFrame(result,columns = ['Hausdorff Distance','Sensitivity','Specificity','Dice Score'],index = ['ET','TC','WT']))
            PLOT = update_PLOT(PLOT,result)
            if cfg.PLOT_GRAPH_RESULT:
                PLOT_RESULT_GRAPH(PLOT)
            result_plot = np.stack([np.array(PLOT['et']),np.array(PLOT['tc']),np.array(PLOT['wt'])],axis=0)
            result_plot_mean_by_class = np.mean(result_plot,axis=0)
            best_epoch_result = np.argmax(result_plot_mean_by_class[-1,:])
            if best_epoch_result==len(PLOT['et'][0])-1:
                best_checkpoint_save = 'best_checkpoint.pth'
                torch.save(
                    {
                        'model_state_dict':model.state_dict(),
                        'optim_state_dict':optim.state_dict(),
                        'current_epoch': epoch,
                        'plot': PLOT,
                        'scheduler_state_dict':scheduler.state_dict()
                    },best_checkpoint_save
                )
        if epoch%cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL==0:
            checkpoint_save_path = 'checkpoint_'+str(epoch)+'.pth'
            torch.save(
                {
                    'model_state_dict':model.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'current_epoch': epoch,
                    'plot': PLOT,
                    'scheduler_state_dict':scheduler.state_dict()
                },checkpoint_save_path
            )
file = tarfile.open('/kaggle/input/brats-2021-task1/BraTS2021_Training_Data.tar')

file.extractall('./brain_images')
file.close()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(cfg,device)