# config/cfg_tadiff_net.py
from munch import DefaultMunch

# -----------------------------------------------
# model config 
network = 'TaDiff_Net' 
data_pool = ['lumiere']  

# Data directories for different datasets
data_dir = {
    'sailor': '/home/brian/project/pl_ddpm/src/data/sailor_npy',
    'lumiere': '/l/users/alaa.mohamed/datasets/lumiere_proc'
}

image_size = 192
in_channels = 13 
out_channels = 7
num_intv_time = 3
model_channels = 32
num_res_blocks = 2
channel_mult = (1, 2, 3, 4)
attention_resolutions = [8, 4]
num_heads = 4
num_classes = 81  # treat_code
max_T = 1000  # diffusion steps
ddpm_schedule = 'linear'  # 'linear', 'cosine', 'log'

# -----------------------------------------------
# optimizer, lr, loss, train config 
opt = 'adamw'  # adam, adamw, sgd, adan
lr = 5.e-3
max_epochs = 30  # total number of training epochs 
max_steps = 600  # total number of training iterations
log_every_n_steps = 50
weight_decay = 3e-5
lrdecay_cosine = True
lr_gamma = 0.585  # 0.5, 0.575, 0.65, 0.585
warmup_steps = 100
loss_type = 'ssim'  # or mse
aux_loss_w = 1.0  # NEW: weight for auxiliary segmentation loss
batch_size = 1
sw_batch = 16  # total batch = sw_batch * batch_size 
num_workers = 8  # 4, 8, 16 up to 32 normally num of CPU cores
grad_clip = 1.5
accumulate_grad_batches = 4  # simulate larger batch size to save GPU mem
n_repeat_tr = 10   # simulate larger train dataset by repeating it
n_repeat_val = 5   # simulate larger val data by repeating 
cache_rate = 0.0   # cache all data in memory (0.0 = no caching, 1.0 = full caching)
check_val_every_n_epoch = 1
# -----------------------------------------------
# I/O, system and log config for trainer (e.g. lightning)
wandb_entity = "allaaamr-mbzuai"  # Change this to your WandB username
logdir = './ckpt'
log_interval = 1
seed = 114514  # 5000, 114514, 3407
gpu_devices = '0'  # str or int e.g. '0', '0,1', '0,1,2,3'
gpu_strategy = "ddp"
gpu_accelerator = "gpu"
precision = '32'  # '32', '16-mixed', 'bf16-mixed'
val_interval_epoch = 10  # Run validation every N epochs

# Checkpointing
resume_from_ckpt = False
ckpt_best_or_last = None  # Path to checkpoint if resuming
ckpt_save_top_k = 3
ckpt_save_last = True
ckpt_monitor = "val_loss"  # val_loss or val_dice
ckpt_filename = "ckpt-{epoch:03d}-{step:06d}-{val_loss:.6f}"
ckpt_mode = "min"  # 'min' for loss, 'max' for dice

# Mode flags
do_train_only = False
do_test_only = False

# -----------------------------------------------
# Create config object
config_keys = [k for k, v in globals().items() if not k.startswith('_') and 
               isinstance(v, (int, float, bool, str, list, tuple, dict))]
config = {k: globals()[k] for k in config_keys}
config = DefaultMunch.fromDict(config)