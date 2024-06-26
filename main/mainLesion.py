from config.Lesion.defaults import _C as cfg
from dataset.Lesion.dataset import LesionDataset, test_dataset
import torch
from tqdm.auto import tqdm
import os
from utils.share_utils import *
from src.model_main import make_model
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
class DiceMetric(nn.Module):
    def __init__(self):
        super(DiceMetric, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-6

        size = target.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/ (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = dice_score.sum()/size

        return dice_loss
torch.backends.cudnn.benchmark = True
def myprint(x,log_to_screen = cfg.LOG_TO_SCREEN):
    with open(cfg.LOG_PATH,"a") as f:
        print(x,file = f)
    if log_to_screen:
        print(x,file = sys.stdout)
def validation(cfg,model,val_dataloader,device):
    predict = []
    answer = []
    with torch.inference_mode():
        for i,data in enumerate(tqdm(val_dataloader)):
            img,mask = data
            inputs = img.to(device)
            label = mask.to(device)
            if cfg.MODEL.MULTIHEAD_OUTPUT:
                output = model(inputs)['segment_volume']
            else:
                output = model(inputs)

            predict.append(output)
            answer.append(label)
        predict = (torch.cat(predict,dim=0)>=0.5).float()
        answer = torch.cat(answer)
        dsc1 = DiceMetric()(predict,answer)
        dsc0 = DiceMetric()(1-predict,1-answer)
        return (dsc1+dsc0)/2
def train(cfg,device):
    data_train = LesionDataset(cfg.DATASET.IMG_TRAIN_PATH,cfg.DATASET.GT_TRAIN_PATH,cfg.DATASET.IMG_SIZE)
    data_val = test_dataset(cfg.DATASET.IMG_VAL_PATH,cfg.DATASET.GT_VAL_PATH,cfg.DATASET.IMG_SIZE)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=cfg.DATASET.NUM_WORKERS)
    val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.DATASET.NUM_WORKERS)
    model = make_model(cfg).to(device)
    optim = make_optimizers(cfg,model)
    scheduler = make_scheduler(cfg,optim)
    loss_func = make_loss_function(cfg)
    current_epoch = 1
    RESULT = {
        'dice': []
    }
    os.makedirs('checkpoint', exist_ok=True)
    if cfg.CHECKPOINT.LOAD:
        checkpoint = torch.load(cfg.CHECKPOINT.PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['current_epoch']+1
        RESULT = checkpoint['RESULT']
    for epoch in range(current_epoch,cfg.SOLVER.MAX_EPOCHS+1):
        running_loss = {}
        loop = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        for i,data in loop:
            loop.set_description(f"Epoch [{epoch}/{cfg.SOLVER.MAX_EPOCHS}]")
            img,mask = data
            inputs = img.to(device)
            label = mask.to(device).float()
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
            result = validation(cfg,model,val_dataloader,device)
            RESULT['dice'].append(result.item())
            if result.item() == max(RESULT['dice']):
                    torch.save(
                        {
                            'model_state_dict':model.state_dict(),
                            'optim_state_dict':optim.state_dict(),
                            'current_epoch': epoch,
                            'scheduler_state_dict':scheduler.state_dict()
                        }, f'checkpoint/best_dice.pth'
                    )
                    print(f'NEW BEST CKPTS!')
        if epoch%cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL==0:
            checkpoint_save_path = 'checkpoint/checkpoint_'+str(epoch)+'.pth'
            torch.save(
                {
                    'model_state_dict':model.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'current_epoch': epoch,
                    'scheduler_state_dict':scheduler.state_dict(),
                    'RESULT': RESULT,
                },checkpoint_save_path
            )
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train(cfg,device)