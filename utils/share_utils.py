import torch
from loss import MHLoss,MHLoss_SELF_GUIDE,BceDiceLoss
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
            lr_lambda= lambda epoch: 1 if epoch >= 0 and epoch < 4
                                else 5 if epoch >= 4 and epoch < 9
                                else 10 if epoch >= 9 and epoch < 19
                                else 20 if epoch >= 19 and epoch < 24
                                else 10 if epoch >= 24 and epoch < 29
                                else 5 if epoch >= 29 and epoch < 39
                                else 1 if epoch >= 39 and epoch < 79
                                else 0.1 if epoch >= 79 and epoch < 89
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
            gamma=cfg.SOLVER.EXPONENTIAL_LR_SETUP
        )
        return scheduler
def make_loss_function(cfg):
    dict_weight_loss = {}
    i = 0
    for loss in cfg.MULTI_HEAD_LOSS_NAME:
        dict_weight_loss[loss] = cfg.MULTI_HEAD_LOSS_WEIGHT[i]
        i+=1
    if cfg.MODEL.MULTIHEAD_OUTPUT:
        if cfg.SELF_GUIDE_LOSS:
            return MHLoss_SELF_GUIDE(dict_weight_loss,cfg.N_CLASS)
        else:
            return MHLoss(dict_weight_loss,cfg.N_CLASS)
                
    else:
        return BceDiceLoss()