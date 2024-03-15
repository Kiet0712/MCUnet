from BRATS import BRATS_make_dataloader
def make_dataloader(cfg,mode,name):
    if name[:5]=='BRATS':
        return BRATS_make_dataloader(cfg,mode,name)