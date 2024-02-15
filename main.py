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
        current_epoch = checkpoint['current_epoch']
    for epoch in range(current_epoch,cfg.SOLVER.MAX_EPOCHS+1):
        torch.backends.cudnn.benchmark = True
        running_loss = {}
        for i,data in enumerate(tqdm(train_dataloader)):
            inputs = data['img'].to(device)
            label = data['label'].to(device)
            if cfg.DATASET.AUGMENTATION:
                inputs,label = data_augmentation(inputs,label)
            optim.zero_grad()
            outputs = model(inputs)
            loss_cal = loss_func(outputs,label,inputs)
            loss_cal['loss'].backward()
            optim.step()
            for key in loss_cal:
                if i==0:
                    running_loss[key]=loss_cal[key].item()
                else:
                    running_loss[key]+=loss_cal[key].item()
            if i % cfg.SOLVER.PRINT_RESULT_INTERVAL == 0 and i != 0:
                print('# ---------------------------------------------------------------------------- #')
                print('Epoch ' + str(epoch) + ', iter ' + str(i+1) + ':')
                for key in running_loss:
                    running_loss[key] = running_loss[key]/cfg.SOLVER.PRINT_RESULT_INTERVAL
                print(pd.DataFrame([running_loss]).transpose())
                for key in running_loss:
                    running_loss[key]=0
        scheduler.step()
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
        if epoch % cfg.SOLVER.EVAL_EPOCH_INTERVAL == 0:
            torch.backends.cudnn.benchmark = False
            result = validation(cfg,model,val_dataloader,device)
            PLOT = update_PLOT(PLOT,result)
            PLOT_RESULT_GRAPH(PLOT)
def main(args):
    #extract dataset
    file = tarfile.open('/kaggle/input/brats-2021-task1/BraTS2021_Training_Data.tar')

    file.extractall('./brain_images')
    file.close()
    cfg.DATASET.BATCH_SIZE = args.batch_size
    cfg.DATASET.CSV_DIR = args.fold_split_dataset
    cfg.DATASET.ROOT_DIR = args.root_dir
    cfg.CHECKPOINT.PATH = args.checkpoint_path
    cfg.CHECKPOINT.LOAD = args.load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(cfg,device)
    
