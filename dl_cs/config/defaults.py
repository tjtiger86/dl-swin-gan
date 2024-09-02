from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.VERSION = 1

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NAME = "DLESPIRiT3D"
_C.MODEL.MODEL_TYPE = "RES"           #Terrence - RES for resnet or SE for squeeze-excitation
_C.MODEL.WEIGHTS = ""
_C.MODEL.META_ARCHITECTURE = "dlespirit"  # 'dlespirit' or 'modl'
_C.MODEL.STRATEGY = "standard"   #standard vs deepspeed

# -----------------------------------------------------------------------------
# Unrolled Model Parameters
# -----------------------------------------------------------------------------
_C.MODEL.PARAMETERS = CN()
_C.MODEL.PARAMETERS.NUM_UNROLLS = 5
_C.MODEL.PARAMETERS.NUM_RESBLOCKS = 2
_C.MODEL.PARAMETERS.NUM_SWINBLOCKS = 2
_C.MODEL.PARAMETERS.NUM_LAYERS = 12 #number of layers in swin or dit
_C.MODEL.PARAMETERS.NUM_HEADS = 6 #self attention num heads for swin or dit
_C.MODEL.PARAMETERS.RR = 16 #reduction ratio for squeeze excitation block
_C.MODEL.PARAMETERS.NUM_FEATURES = 256
_C.MODEL.PARAMETERS.DROPOUT = 0.0
_C.MODEL.PARAMETERS.NUM_EMAPS = 2

# Diffusion-specific flags
_C.MODEL.PARAMETERS.NOISE_SCHED = "linear"
_C.MODEL.PARAMETERS.LEARN_SIGMA = False

# Unrolled flags
_C.MODEL.PARAMETERS.FIX_STEP_SIZE = False
_C.MODEL.PARAMETERS.SHARE_WEIGHTS = False
_C.MODEL.PARAMETERS.SLWIN_INIT = False
_C.MODEL.PARAMETERS.GRAD_CHECKPOINT = False

# MoDL-specific flags
_C.MODEL.PARAMETERS.MODL = CN()
_C.MODEL.PARAMETERS.MODL.NUM_CG_STEPS = 10
_C.MODEL.PARAMETERS.MODL.MU = 0.1
_C.MODEL.PARAMETERS.MODL.FIX_PENALTY = False

# DSLR-specific flags
_C.MODEL.PARAMETERS.DSLR = CN()
_C.MODEL.PARAMETERS.DSLR.NUM_BASIS = 8
_C.MODEL.PARAMETERS.DSLR.BLOCK_SIZE = 16
_C.MODEL.PARAMETERS.DSLR.OVERLAPPING = True
_C.MODEL.PARAMETERS.DSLR.NUM_CG_STEPS = 10


#SWIN specific Parameters
#_C.MODEL.PARAMETERS.WINDOW_SIZE = (4, 4) #Patch size 
#_C.MODEL.PARAMETERS.NUM_HEAD = (4) #reduction ratio for squeeze excitation block
#_C.MODEL.PARAMETERS.NUM_LAYERS = (4)

# Conv block parameters
_C.MODEL.PARAMETERS.CONV_BLOCK = CN()
_C.MODEL.PARAMETERS.CONV_BLOCK.KERNEL_SIZE = (3,)
# Flag to turn on circular padding across phase encode and time dimension
_C.MODEL.PARAMETERS.CONV_BLOCK.CIRCULAR_PAD = True
# Either "relu" or "leaky_relu"
_C.MODEL.PARAMETERS.CONV_BLOCK.ACTIVATION = "relu"
# Either "none", "instance", or "batch"
_C.MODEL.PARAMETERS.CONV_BLOCK.NORM = "none"
# Use separable (2+1)D convolutions
_C.MODEL.PARAMETERS.CONV_BLOCK.SEPARABLE = True
# Use complex-valued convolutional layer
_C.MODEL.PARAMETERS.CONV_BLOCK.COMPLEX = True

# Loss function parameters
_C.MODEL.RECON_LOSS = CN()
_C.MODEL.RECON_LOSS.NAME = "complex_l1"
_C.MODEL.RECON_LOSS.RENORMALIZE_DATA = True
_C.MODEL.RECON_LOSS.LOSS_WEIGHT = False

# -----------------------------------------------------------------------------
# Dataset Paths
# -----------------------------------------------------------------------------
_C.DATASET = CN()
# List of the dataset names for training.
_C.DATASET.TRAIN = ()
# List of the dataset names for validation.
_C.DATASET.VAL = ()
# List of the dataset names for testing.
_C.DATASET.TEST = ()

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Training and Validation batch sizes
_C.DATALOADER.TRAIN_BATCH_SIZE = 1
_C.DATALOADER.VAL_BATCH_SIZE = 1
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Number of volumes to leave out during training (to simulate data scarce situations)
_C.DATALOADER.SUBSAMPLE = 1.0


# -----------------------------------------------------------------------------
# Augmentations/Transforms
# -----------------------------------------------------------------------------
_C.AUG_TRAIN = CN()
_C.AUG_TRAIN.CROP_READOUT = 64
_C.AUG_TRAIN.UNDERSAMPLE = CN()
_C.AUG_TRAIN.UNDERSAMPLE.NAME = "VDktMaskFunc"
_C.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS = (10, 15)
_C.AUG_TRAIN.UNDERSAMPLE.CALIBRATION_SIZE = 1
_C.AUG_TRAIN.UNDERSAMPLE.VD_POWER = 1.5
_C.AUG_TRAIN.UNDERSAMPLE.PERTURB_FACTOR = 0.4
_C.AUG_TRAIN.UNDERSAMPLE.ADHERE_FACTOR = 0.33
_C.AUG_TRAIN.UNDERSAMPLE.PARTIAL_KX = 0.25
_C.AUG_TRAIN.UNDERSAMPLE.PARTIAL_KY = 0.0

_C.AUG_VAL = CN()
_C.AUG_TRAIN.CROP_READOUT = 0
_C.AUG_VAL.UNDERSAMPLE = CN()
_C.AUG_VAL.UNDERSAMPLE.NAME = "VDktMaskFunc"
_C.AUG_VAL.UNDERSAMPLE.ACCELERATIONS = (10, 15)
_C.AUG_VAL.UNDERSAMPLE.CALIBRATION_SIZE = 1
_C.AUG_VAL.UNDERSAMPLE.VD_POWER = 1.5
_C.AUG_VAL.UNDERSAMPLE.PERTURB_FACTOR = 0.4
_C.AUG_VAL.UNDERSAMPLE.ADHERE_FACTOR = 0.33
_C.AUG_VAL.UNDERSAMPLE.PARTIAL_KX = 0.25
_C.AUG_VAL.UNDERSAMPLE.PARTIAL_KY = 0.0

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()

_C.OPTIMIZER.NAME = "Adam"
_C.OPTIMIZER.MAX_EPOCHS = 1000
_C.OPTIMIZER.GRAD_ACCUM_ITERS = 1
_C.OPTIMIZER.GRAD_CLIP_VAL = 0.

# Adam Parameters
_C.OPTIMIZER.ADAM = CN()
_C.OPTIMIZER.ADAM.LR = 0.0001
_C.OPTIMIZER.ADAM.BETAS = (0.9, 0.999)
_C.OPTIMIZER.ADAM.EPS = 1e-8
_C.OPTIMIZER.ADAM.WEIGHT_DECAY = 0.

# ---------------------------------------------------------------------------- #
# Learning Rate Scheduler
# ---------------------------------------------------------------------------- #

# Learning Rate Scheduler Parameters
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = "StepLR"
_C.LR_SCHEDULER.STEP_SIZE = 1000
_C.LR_SCHEDULER.GAMMA = 0.5 # decay rate

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.EVAL = CN()
# The period over which to evaluate the model during training.
# Set to 0 to disable.
_C.EVAL.RUN_EVERY_N_EPOCHS = 1

# ---------------------------------------------------------------------------- #
# Logger options
# ---------------------------------------------------------------------------- #
_C.LOGGER = CN()
_C.LOGGER.LOG_METRICS_EVERY_N_STEPS = 50
_C.LOGGER.LOG_IMAGES_EVERY_N_STEPS = 100
_C.LOGGER.LOG_PREDICTION_EVERY_N_STEPS = 500

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = ""
# Device on which to run training (-1=cpu, else=gpu)
_C.DEVICE = -1
# Random seed
_C.SEED = 1
# Option to turn on CUDNN benchmark
_C.CUDNN_BENCHMARK = False

# ---------------------------------------------------------------------------- #
# Config Description
# ---------------------------------------------------------------------------- #
_C.DESCRIPTION = CN()
# Brief description about config
_C.DESCRIPTION.BRIEF = ""
# Experiment name for logging to Weights & Biases
_C.DESCRIPTION.EXP_NAME = ""
# Tags associated with experiment.
# e.g. "fastmri_knee_mc" for fastMRI dataset; "unrolled" for using unrolled network; etc.
_C.DESCRIPTION.TAGS = ()
