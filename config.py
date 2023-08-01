from yacs.config import CfgNode as CN

_CN = CN()

_CN.MODULE = CN()
# _CN.MODULE.BACKBONE_TYPE = 'ResNetFPN'
# _CN.MODULE.BACKBONE_TYPE = 'swin'
_CN.MODULE.BACKBONE_TYPE = 'swin_v1'
_CN.MODULE.INPUT_CHANNEL = 3
# _CN.MODULE.BACKBONE_TYPE = 'swin_v2'
_CN.MODULE.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.MODULE.FINE_WINDOW_SIZE = 7  # window_size in fine_level, must be odd
_CN.MODULE.FINE_CONCAT_COARSE_FEAT = True

# 1. backbone (local feature CNN) config
_CN.MODULE.RESNETFPN = CN()
_CN.MODULE.RESNETFPN.INITIAL_DIM = 128
_CN.MODULE.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# 2. coarse module config
_CN.MODULE.COARSE = CN()
_CN.MODULE.COARSE.D_MODEL = 256
_CN.MODULE.COARSE.NHEAD = 8
_CN.MODULE.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.MODULE.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.MODULE.COARSE.TEMP_BUG_FIX = True

# 3. Coarse-Matching config
_CN.MODULE.MATCH_COARSE = CN()
_CN.MODULE.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
_CN.MODULE.MATCH_COARSE.THR = 0.20
_CN.MODULE.MATCH_COARSE.BORDER_RM = 2
_CN.MODULE.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MODULE.MATCH_COARSE.TRAIN_COARSE_PERCENT = 1.0  # training tricks: save GPU memory
_CN.MODULE.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.MODULE.MATCH_COARSE.SPARSE_SPVS = True

# 4. fine module config
_CN.MODULE.FINE = CN()
_CN.MODULE.FINE.D_MODEL = 64
_CN.MODULE.FINE.NHEAD = 8
_CN.MODULE.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.MODULE.FINE.ATTENTION = 'linear'

# 4. pose module config
_CN.MODULE.POSE_NET_FLAG = 'none'

_CN.MODULE.POSE = CN()
_CN.MODULE.POSE.D_MODEL = 256
_CN.MODULE.POSE.NHEAD = 8
_CN.MODULE.POSE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.MODULE.POSE.ATTENTION = 'linear'
_CN.MODULE.POSE.SIZE = [60, 80]
_CN.MODULE.POSE.AXIS_W = 1
_CN.MODULE.POSE.TRANS_W = 1
_CN.MODULE.POSE.AXIS_W_CV = 0
_CN.MODULE.POSE.TRANS_W_CV = 0

_CN.MODULE.POSE_NEW = CN()
_CN.MODULE.POSE_NEW.D_MODEL = 256
_CN.MODULE.POSE_NEW.NHEAD = 8
_CN.MODULE.POSE_NEW.LAYER_NAMES = ['cross', 'cross'] * 2
_CN.MODULE.POSE_NEW.ATTENTION = 'linear'
_CN.MODULE.POSE_NEW.SIZE = [60, 80]
_CN.MODULE.POSE_NEW.AXIS_W = 1
_CN.MODULE.POSE_NEW.TRANS_W = 1

# 5. Losses
# -- # coarse-level
_CN.MODULE.LOSS = CN()
_CN.MODULE.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.MODULE.LOSS.COARSE_WEIGHT = 1.0
_CN.MODULE.LOSS.SPARSE_SPVS = False
# -- - -- # focal loss (coarse)
_CN.MODULE.LOSS.FOCAL_ALPHA = 0.25
_CN.MODULE.LOSS.FOCAL_GAMMA = 2.0
_CN.MODULE.LOSS.POS_WEIGHT = 1.0
_CN.MODULE.LOSS.NEG_WEIGHT = 1.0
# -- # fine-level
_CN.MODULE.LOSS.POSE_NET_FLAG = _CN.MODULE.POSE_NET_FLAG
_CN.MODULE.LOSS.POSE_LOSS_CAL_FLAG = _CN.MODULE.POSE_NET_FLAG
_CN.MODULE.LOSS.FINE_WEIGHT = 1.0
_CN.MODULE.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)
_CN.MODULE.LOSS.R_WEIGHT = 1.
_CN.MODULE.LOSS.T_WEIGHT = 1.

##############  Dataset  ##############
_CN.DATASET = CN()
_CN.DATASET.TRAINVAL_DATA_ROOT = 0
_CN.DATASET.TRAIN_DATA_ROOT = 0
_CN.DATASET.VAL_DATA_ROOT = 0
_CN.DATASET.TEST_DATA_ROOT = 0
_CN.DATASET.DATA_ENHANCE = 0
_CN.DATASET.LIGHTING_DATA = True
_CN.DATASET.IMG_READ_GRAY = False
_CN.DATASET.DEBUG = False

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = 0  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = 0  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'  # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 15, 18, 21, 24, 27]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = False
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32  # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'normal'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = 0

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

_CN.TRAINER.SEED = 114514


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
