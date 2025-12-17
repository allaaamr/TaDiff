# config/arg_parse.py
import argparse

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_args(cfg):
    parser = argparse.ArgumentParser(description='TaDiff Training Script')
    
    # ============================================
    # SYSTEM & I/O
    # ============================================
    parser.add_argument("--seed", type=int, default=cfg.seed,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu_devices", type=str, default=cfg.gpu_devices,
                        help="GPU device IDs (e.g., '0' or '0,1,2,3')")
    parser.add_argument("--gpu_strategy", type=str, default=cfg.gpu_strategy,
                        help="Training strategy (ddp, auto)")
    parser.add_argument("--gpu_accelerator", type=str, default=cfg.gpu_accelerator,
                        help="Accelerator type (gpu, cpu)")
    parser.add_argument("--logdir", type=str, default=cfg.logdir,
                        help="Directory for saving checkpoints and logs")
    parser.add_argument("--log_interval", type=int, default=cfg.log_interval,
                        help="Logging interval in steps")
    
    # ============================================
    # MODE
    # ============================================
    parser.add_argument('--do_train_only', default=cfg.do_train_only, 
                        action='store_true',
                        help="Only perform training (skip validation)")
    parser.add_argument('--do_test_only', default=cfg.do_test_only, 
                        action='store_true',
                        help="Only perform testing (skip training)")
    
    # ============================================
    # DATA
    # ============================================
    parser.add_argument("--data_pool", type=str, nargs='+', default=cfg.data_pool,
                        help="List of datasets to use (e.g., sailor lumiere)")
    parser.add_argument("--cache_rate", type=float, default=cfg.cache_rate,
                        help="Cache rate for data loading (0.0-1.0)")
    parser.add_argument("--n_repeat_tr", type=int, default=cfg.n_repeat_tr,
                        help="Number of times to repeat training data")
    parser.add_argument("--n_repeat_val", type=int, default=cfg.n_repeat_val,
                        help="Number of times to repeat validation data")
    
    # ============================================
    # MODEL
    # ============================================
    parser.add_argument("--network", type=str, default=cfg.network,
                        help="Network architecture name")
    parser.add_argument("--image_size", type=int, default=cfg.image_size,
                        help="Input image size")
    parser.add_argument("--in_channels", type=int, default=cfg.in_channels,
                        help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=cfg.out_channels,
                        help="Number of output channels")
    parser.add_argument("--model_channels", type=int, default=cfg.model_channels,
                        help="Base channel count for the model")
    parser.add_argument("--num_res_blocks", type=int, default=cfg.num_res_blocks,
                        help="Number of residual blocks per level")
    parser.add_argument("--channel_mult", type=int, nargs='+', 
                        default=list(cfg.channel_mult),
                        help="Channel multipliers for each level")
    parser.add_argument("--attention_resolutions", type=int, nargs='+',
                        default=list(cfg.attention_resolutions),
                        help="Resolutions at which to apply attention")
    parser.add_argument("--num_heads", type=int, default=cfg.num_heads,
                        help="Number of attention heads")
    parser.add_argument("--num_classes", type=int, default=cfg.num_classes,
                        help="Number of treatment classes")
    
    # ============================================
    # DIFFUSION
    # ============================================
    parser.add_argument("--max_T", type=int, default=cfg.max_T,
                        help="Maximum number of diffusion timesteps")
    parser.add_argument("--ddpm_schedule", type=str, default=cfg.ddpm_schedule,
                        choices=['linear', 'cosine', 'log'],
                        help="Noise schedule for diffusion process")
    
    # ============================================
    # TRAINING
    # ============================================
    parser.add_argument("--max_epochs", type=int, default=cfg.max_epochs,
                        help="Maximum number of training epochs")
    parser.add_argument("--max_steps", type=int, default=cfg.max_steps,
                        help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size,
                        help="Batch size for training")
    parser.add_argument("--sw_batch", type=int, default=cfg.sw_batch,
                        help="Sliding window batch size")
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers,
                        help="Number of data loading workers")
    parser.add_argument("--precision", default=cfg.precision,
                        help="Training precision (16-mixed, 32, bf16-mixed)")
    parser.add_argument("--accumulate_grad_batches", type=int, 
                        default=cfg.accumulate_grad_batches,
                        help="Number of batches for gradient accumulation")
    parser.add_argument("--grad_clip", type=float, default=cfg.grad_clip,
                        help="Gradient clipping value")
    
    # ============================================
    # OPTIMIZER
    # ============================================
    parser.add_argument("--opt", type=str, default=cfg.opt,
                        choices=['adam', 'adamw', 'sgd', 'adan'],
                        help="Optimizer type")
    parser.add_argument("--lr", type=float, default=cfg.lr,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay,
                        help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=cfg.warmup_steps,
                        help="Number of warmup steps for learning rate")
    parser.add_argument("--lrdecay_cosine", type=str2bool, default=cfg.lrdecay_cosine,
                        help="Use cosine learning rate decay")
    parser.add_argument("--lr_gamma", type=float, default=cfg.lr_gamma,
                        help="Learning rate decay gamma")
    
    # ============================================
    # LOSS
    # ============================================
    parser.add_argument("--loss_type", type=str, default=cfg.loss_type,
                        choices=['mse', 'ssim'],
                        help="Loss function type")
    parser.add_argument("--aux_loss_w", type=float, default=getattr(cfg, 'aux_loss_w', 1.0),
                        help="Weight for auxiliary loss")
    
    # ============================================
    # VALIDATION
    # ============================================
    parser.add_argument("--val_interval_epoch", type=int, 
                        default=cfg.val_interval_epoch,
                        help="Run validation every N epochs")
    
    # ============================================
    # CHECKPOINTING
    # ============================================
    parser.add_argument("--resume_from_ckpt", type=str2bool, 
                        default=cfg.resume_from_ckpt,
                        help="Resume training from checkpoint")
    parser.add_argument("--ckpt_best_or_last", type=str, 
                        default=cfg.ckpt_best_or_last,
                        help="Path to checkpoint file for resuming")
    parser.add_argument("--ckpt_save_top_k", type=int, 
                        default=cfg.ckpt_save_top_k,
                        help="Save top K checkpoints")
    parser.add_argument("--ckpt_save_last", type=str2bool, 
                        default=cfg.ckpt_save_last,
                        help="Save last checkpoint")
    parser.add_argument("--ckpt_monitor", type=str, default=cfg.ckpt_monitor,
                        help="Metric to monitor for checkpointing")
    parser.add_argument("--ckpt_filename", type=str, default=cfg.ckpt_filename,
                        help="Checkpoint filename pattern")
    parser.add_argument("--ckpt_mode", type=str, default=cfg.ckpt_mode,
                        choices=['min', 'max'],
                        help="Checkpoint monitoring mode")
    
    # ============================================
    # LOGGING (WandB)
    # ============================================
    parser.add_argument("--wandb_entity", type=str, default=cfg.wandb_entity,
                        help="WandB entity (username or team)")
    
    args = parser.parse_args()
    
    # Convert channel_mult and attention_resolutions back to tuples
    args.channel_mult = tuple(args.channel_mult)
    args.attention_resolutions = tuple(args.attention_resolutions)
    
    # Add data_dir from config (not typically passed as CLI arg since it's complex)
    args.data_dir = cfg.data_dir
    
    return args