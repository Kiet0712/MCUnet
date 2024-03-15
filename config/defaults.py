import os
from yacs.config import CfgNode as CN



# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C = CN()
_C.MODEL = CN()
_C.MODEL.CRFBNET = False
_C.MODEL.CRFBNET_DEPTH = 4
_C.MODEL.MULTIHEAD_OUTPUT = False
_C.MODEL.SELF_GUIDE_OUTPUT = False
_C.MODEL.NORM = 'IN3d' # ('IN' or 'BN') + ('2d' or '3d')
_C.MODEL.NUM_FEATURES_START_UNET = 32
_C.MODEL.CONV_TYPE = 'normal3d' # ('normal' or 'coord') + ('2d' or '3d') 
_C.MODEL.DOUBLE_CONV_RESIDUAL = True
_C.MODEL.OUTPUT_COORDCONV = False
_C.MODEL.ATTENTION_UP = True
_C.MODEL.MULTI_PATH_COMBINE = False
_C.MODEL.CRFBNET_KERNEL_SIZE = 3
_C.MODEL.CRFBNET_PADDING = 1
_C.MODEL.SELF_GUIDE_OUTPUT_SIGMOID = True
# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "adam" #other choice is adamw, sgd
_C.SOLVER.SCHEDULER = "LambdaLR" #other choice is OneCycleLR, ExponentialLR
_C.SOLVER.EXPONENTIAL_LR_SETUP = 0.8
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.WEIGHT_DECAY = 1e-6
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.SGD_MOMENTUM = 0.9
_C.SOLVER.NESTEROV = True
# AdamOneCycle
_C.SOLVER.PCT_START = 0.4
_C.SOLVER.DIV_FACTOR = 100
# EVAL AND SAVE CHECKPOINT
_C.SOLVER.EVAL_EPOCH_INTERVAL = 1
_C.SOLVER.SAVE_CHECKPOINT_INTERVAL = 1
_C.SOLVER.PRINT_RESULT_INTERVAL = 100

# ---------------------------------------------------------------------------- #
# DATASET
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.SPLIT_DIR = ''
_C.DATASET.FOLD = 0
_C.DATASET.ROOT_DIR = ''
_C.DATASET.NUM_WORKERS = 4
_C.DATASET.PIN_MEMORY = True
_C.DATASET.BATCH_SIZE = 1
_C.DATASET.AUGMENTATION = True
_C.DATASET.AUGMENTATION_PROBA = 0.8
_C.DATASET.AUGMENTATION_NOISE_ONLY = False
_C.DATASET.AUGMENTATION_CHANNEL_SHUFFLE = False
_C.DATASET.AUGMENTATION_DROP_CHANNEL = True
_C.DATASET.NAME = 'BRATS2021'
_C.DATASET.SWINUNETR_SPLIT = False
# ---------------------------------------------------------------------------- #
# CHECKPOINT
# ---------------------------------------------------------------------------- #
_C.CHECKPOINT = CN()
_C.CHECKPOINT.PATH = ''
_C.CHECKPOINT.LOAD = False

# ---------------------------------------------------------------------------- #
# VALIDATION
# ---------------------------------------------------------------------------- #
_C.VALIDATION_TYPE = 'normal' #windown for sliding_windown inference
_C.ROI_SIZE = [128,128,128]
_C.OVERLAP = 0.7
_C.SW_BATCHSIZE = 1


# ---------------------------------------------------------------------------- #
# LOSS_FUNCTION
# ---------------------------------------------------------------------------- #
_C.SELF_GUIDE_LOSS = True
_C.MULTI_HEAD_LOSS_NAME = [
    'seg_vol_loss',
    'recstr_vol_loss',
    'c1_fg_loss',
    'c2_fg_loss',
    'c4_fg_loss',
    'c1_bg_loss',
    'c2_bg_loss',
    'c4_bg_loss',
    'recstr_guide_loss',
    'c1_bg_guide_loss',
    'c1_fg_guide_loss',
    'c2_bg_guide_loss',
    'c2_fg_guide_loss',
    'c4_bg_guide_loss',
    'c4_fg_guide_loss'
]
_C.MULTI_HEAD_LOSS_WEIGHT = [5,2,2,2,2,2,2,2,0.75,0.75,0.75,0.75,0.75,0.75,0.75]
_C.N_CLASS = 3
_C.CHANNEL_IN = 4
# ---------------------------------------------------------------------------- #
# PLOT_GRAPH_RESULT
# ---------------------------------------------------------------------------- #
_C.PLOT_GRAPH_RESULT = True

# ---------------------------------------------------------------------------- #
# PLOT_GRAPH_RESULT
# ---------------------------------------------------------------------------- #
_C.LOG_PATH = ''

# ---------------------------------------------------------------------------- #
# LOG_TO_SCREEN
# ---------------------------------------------------------------------------- #
_C.LOG_TO_SCREEN = True