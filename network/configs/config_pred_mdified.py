from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
_CN.BACKBONE_TYPE = 'swin_v1'
_CN.INPUT_CHANNEL = 3
_CN.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.FINE_WINDOW_SIZE = 7  # window_size in fine_level, must be odd
_CN.FINE_CONCAT_COARSE_FEAT = True

# 1. LoFTR-backbone (local feature CNN) config
_CN.RESNETFPN = CN()
_CN.RESNETFPN.INITIAL_DIM = 128
_CN.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 0.2
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKH_ITERS = 3
_CN.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.MATCH_COARSE.SKH_PREFILTER = True
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.4  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. LoFTR-fine module config
_CN.FINE = CN()
_CN.FINE.D_MODEL = 64
_CN.FINE.NHEAD = 8
_CN.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.FINE.ATTENTION = 'linear'

# 4. pose module config
_CN.POSE_NET_FLAG = 'None'

_CN.POSE = CN()
_CN.POSE.D_MODEL = 256
_CN.POSE.NHEAD = 8
_CN.POSE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.POSE.ATTENTION = 'linear'
_CN.POSE.SIZE = [60, 80]
_CN.POSE.AXIS_W = 1
_CN.POSE.TRANS_W = 1
_CN.POSE.AXIS_W_CV = 0
_CN.POSE.TRANS_W_CV = 0

_CN.POSE_NEW = CN()
_CN.POSE_NEW.D_MODEL = 256
_CN.POSE_NEW.NHEAD = 8
_CN.POSE_NEW.LAYER_NAMES = ['cross', 'cross'] * 1
_CN.POSE_NEW.ATTENTION = 'linear'
_CN.POSE_NEW.SIZE = [60, 80]
_CN.POSE_NEW.AXIS_W = 1
_CN.POSE_NEW.TRANS_W = 1



cfg_new = lower_config(_CN)
