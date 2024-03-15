from config.defaults import _C as cfg
from dataset.KvasirSEG.dataset import KvasirSEG_Dataset
import torch
from tqdm.auto import tqdm
import utils.KvasirSEG.utils as ut
import os
from utils.share_utils import *
from src.model_main import make_model
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
class DiceScoreCoefficient(nn.Module):
    def __init__(self, n_classes):
        super(DiceScoreCoefficient, self).__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def fast_hist(self, label_true, label_pred, labels):
        mask = (label_true >= 0) & (label_true < labels)
        hist = np.bincount(labels * label_true[mask].astype(int) + label_pred[mask], minlength=labels ** 2,
        ).reshape(labels, labels)
        return hist

    def _dsc(self, mat):
        diag_all = np.sum(np.diag(mat))
        fp_all = mat.sum(axis=1)
        fn_all = mat.sum(axis=0)
        tp_tn = np.diag(mat)
        precision = np.zeros((self.n_classes)).astype(np.float32)
        recall = np.zeros((self.n_classes)).astype(np.float32)    
        f2 = np.zeros((self.n_classes)).astype(np.float32)

        for i in range(self.n_classes):
            if (fp_all[i] != 0)and(fn_all[i] != 0):   
                precision[i] = float(tp_tn[i]) / float(fp_all[i])
                recall[i] = float(tp_tn[i]) / float(fn_all[i])
                if (precision[i] != 0)and(recall[i] != 0):  
                     f2[i] = (2.0*precision[i]*recall[i]) / (precision[i]+recall[i])
                else:       
                    f2[i] = 0.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0

        return f2


    ### main ###
    def forward(self, output, target):
        output = np.array(output)
        target = np.array(target)
        seg = np.argmax(output,axis=1)

        for lt, lp in zip(target, seg):
            self.confusion_matrix += self.fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

        dsc = self._dsc(self.confusion_matrix)

        return dsc
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
            
            output = model(inputs)

            inputs = inputs.cpu().numpy()
            output = output.cpu().numpy()
            label = label.cpu().numpy()
            for j in range(cfg.DATASET.BATCH_SIZE):
                predict.append(output[j])
                answer.append(label[j])
        dsc = DiceScoreCoefficient(n_classes=cfg.N_CLASS)(predict, answer)
        return dsc
def train(cfg,device):
    train_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                     ut.ExtRandomRotation(degrees=90),
                                     ut.ExtRandomHorizontalFlip(),
                                     ut.ExtToTensor(),
                                     ])

    val_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                   ut.ExtToTensor(),
                                   ])
    data_train = KvasirSEG_Dataset(root = cfg.DATASET.ROOT_DIR, 
                                     dataset_type='train', 
                                     transform=train_transform) 
    data_val = KvasirSEG_Dataset(root = cfg.DATASET.ROOT_DIR, 
                                   dataset_type='val', 
                                   transform=val_transform)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=cfg.DATASET.NUM_WORKERS)
    val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.DATASET.NUM_WORKERS)
    model = make_model(cfg).to(device)
    optim = make_optimizers(cfg,model)
    scheduler = make_scheduler(cfg,optim)
    loss_func = make_loss_function(cfg)
    current_epoch = 1
    best_dice = -1
    if cfg.CHECKPOINT.LOAD:
        checkpoint = torch.load(cfg.CHECKPOINT.PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['current_epoch']+1
        best_dice = checkpoint['best_dice']
    for epoch in range(current_epoch,cfg.SOLVER.MAX_EPOCHS+1):
        running_loss = {}
        loop = tqdm(enumerate(train_dataloader),total=len(train_dataloader),leave=False)
        for i,data in loop:
            loop.set_description(f"Epoch [{epoch}/{cfg.SOLVER.MAX_EPOCHS}]")
            img,mask = data
            inputs = img.to(device)
            label = mask.to(device)
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
            myprint('dice = ' + str(result),True)
            if result > best_dice:
                best_checkpoint_save = 'best_checkpoint.pth'
                myprint('NEW BEST RESULT WILL BE SAVE IN ' + best_checkpoint_save,True)
                torch.save(
                    {
                        'model_state_dict':model.state_dict(),
                        'optim_state_dict':optim.state_dict(),
                        'current_epoch': epoch,
                        'best_dice':result,
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
                    'best_dice':best_dice,
                    'scheduler_state_dict':scheduler.state_dict()
                },checkpoint_save_path
            )
def test(cfg,device,checkpoint_path):
    # preprocceing #
    test_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                    ut.ExtToTensor(),
                                    ])
    # data loader #
    data_test = KvasirSEG_Dataset(root = cfg.DATASET.ROOT_DIR, 
                                    dataset_type='test', 
                                    transform=test_transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.DATASET.NUM_WORKERS)
    model = make_model(cfg).to(device)
    checkpoint = torch.load(cfg.CHECKPOINT.PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Dice = ' + str(validation(cfg,model,test_loader,device)))
