"""
TaDiff Model Training Script - Slice-wise Processing

This script trains the TaDiff model using a slice-by-slice approach similar to test.py.
Key features:
- Processes 3D volumes slice-by-slice for efficient GPU memory usage
- Identifies tumor-containing slices for focused training
- Supports multi-GPU training with PyTorch Lightning
- Implements comprehensive logging and checkpointing
- Uses same data preparation as test.py for consistency

The script processes 3D medical volumes by:
1. Loading 3D data [B, C*S, D, H, W]
2. Reshaping to [B, S, C, H, W, D] format
3. Selecting slices with sufficient tumor content
4. Processing each slice as 2D: [B, S, C, H, W]
5. Computing loss and updating model

Example usage:
    python train_slicewise.py --gpu_devices 0 --batch_size 2 --max_epochs 100
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from monai.data import CacheDataset, DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.cfg_tadiff_net import config as default_config
from config.arg_parse import load_args
from src.tadiff_model import Tadiff_model
from src.data.data_loader import val_transforms, npz_keys
from src.evaluation.metrics import (
    # setup_metrics,
    # calculate_metrics,
    calculate_tumor_volumes,
    get_slice_indices,
    MetricsCalculator
)

class SliceWiseDataset(torch.utils.data.Dataset):
    """
    Dataset that extracts 2D slices from 3D volumes on-the-fly.
    Only returns slices with sufficient tumor content.
    """
    
    def __init__(self, file_list: List[Dict], transform, min_tumor_voxels: int = 100):
        """
        Args:
            file_list: List of file dictionaries with keys from npz_keys
            transform: MONAI transform pipeline
            min_tumor_voxels: Minimum tumor size to include slice
        """
        self.file_list = file_list
        self.transform = transform
        self.min_tumor_voxels = min_tumor_voxels
        
        # Cache 3D volumes and their valid slice indices
        self.volumes_cache = []
        self.slice_info = []  # List of (volume_idx, slice_idx) tuples
        
        print("Building slice index...")
        self._build_slice_index()
        print(f"Total valid slices: {len(self.slice_info)}")
    
    def _build_slice_index(self):
        """Pre-compute which slices have sufficient tumor content."""
        for vol_idx, file_dict in enumerate(self.file_list):
            # Load and transform volume
            data = self.transform(file_dict)
            
            # Calculate tumor volumes per slice
            labels = data['label']  # Shape: [S, H, W, D] or [H, W, D]
            if labels.dim() == 4:
                # Multi-session: [S, H, W, D]
                z_mask_size = calculate_tumor_volumes(labels)
            else:
                # Single session: [H, W, D]
                labels = labels.unsqueeze(0)
                z_mask_size = calculate_tumor_volumes(labels)
            
            # Find slices with sufficient tumor
            valid_slices = torch.where(z_mask_size > self.min_tumor_voxels)[0]
            
            # Cache volume and record valid slices
            self.volumes_cache.append(data)
            for slice_idx in valid_slices:
                self.slice_info.append((vol_idx, slice_idx.item()))
    
    def __len__(self):
        return len(self.slice_info)
    
    def __getitem__(self, idx):
        """Return a single 2D slice from cached volumes."""
        vol_idx, slice_idx = self.slice_info[idx]
        volume_data = self.volumes_cache[vol_idx]
        
        image_3d = volume_data['image']  # [C*S, D, H, W]
        label_3d = volume_data['label']  # [S, D, H, W] or [D, H, W]

        # Reorder to [C*S, H, W, D]
        image_3d = image_3d.permute(0, 2, 3, 1)       # (0: C*S, 1: D, 2: H, 3: W) -> (C*S, H, W, D)

        # Reorder labels similarly
        if label_3d.dim() == 4:
            # [S, D, H, W] -> [S, H, W, D]
            label_3d = label_3d.permute(0, 2, 3, 1)
        else:
            # [D, H, W] -> [H, W, D]
            label_3d = label_3d.permute(1, 2, 0)

        image_2d = image_3d[..., slice_idx]          # [C*S, H, W]
        label_2d = label_3d[..., slice_idx]         # [S, H, W] or [H, W]
        if label_2d.dim() == 2:
            label_2d = label_2d.unsqueeze(0)        # [1, H, W]
        
        print("image_2d ", image_2d.shape)
        return {
            'image': image_2d,
            'label': label_2d,
            'days': volume_data['days'],
            'treatment': volume_data['treatment'],
            'slice_idx': slice_idx,
            'volume_idx': vol_idx
        }


class TaDiffDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for slice-wise training."""
    
    def __init__(
        self,
        train_files: List[Dict],
        val_files: List[Dict],
        batch_size: int = 2,
        num_workers: int = 4,
        min_tumor_voxels: int = 100
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_tumor_voxels = min_tumor_voxels
        
    def setup(self, stage: Optional[str] = None):
        """Setup train and val datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = SliceWiseDataset(
                self.train_files,
                val_transforms,
                self.min_tumor_voxels
            )
            self.val_dataset = SliceWiseDataset(
                self.val_files,
                val_transforms,
                self.min_tumor_voxels
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class SliceWiseTaDiffModel(Tadiff_model):
    """
    Extended TaDiff model that processes 2D slices.
    Handles data preparation matching test.py's approach.
    """
    
    def prepare_slice_batch(self, batch: Dict[str, torch.Tensor], n_modalities: int = 3):
        """
        Prepare 2D slice batch for model input.
        Follows same logic as test.py's prepare_image_batch().
        
        Args:
            batch: Dictionary with keys 'image', 'label', 'days', 'treatment'
                   image: [B, C*S, H, W] - 2D slices
                   label: [B, S, H, W] - 2D label slices
            n_modalities: Number of imaging modalities (default: 3 for T1, T1c, FLAIR)
        
        Returns:
            Processed batch ready for model
        """
        images = batch['image']  # [B, C*S, H, W]
        labels = batch['label']  # [B, S, H, W]
        
        b, cs, h, w = images.shape
        
        # Determine number of sessions
        if labels.dim() == 3:
            n_sessions = 1
            labels = labels.unsqueeze(1)  # [B, 1, H, W]
        else:
            n_sessions = labels.shape[1]
        
        # Reshape images: [B, C*S, H, W] -> [B, S, C, H, W]
        # Assuming 4 modalities stored as [t1, t1c, flair, t2]
        images = images.view(b, 4, n_sessions, h, w)  # [B, 4, S, H, W]
        images = images.permute(0, 2, 1, 3, 4)  # [B, S, 4, H, W]
        
        # Remove T2 modality to get 3 modalities
        images = images[:, :, :n_modalities, :, :]  # [B, S, 3, H, W]
        
        return {
            'image': images,
            'label': labels,
            'days': batch['days'],
            'treatment': batch['treatment']
        }
    
    def get_loss_slicewise(self, batch: Dict[str, torch.Tensor], mode='train'):
        """
        Compute loss for 2D slice batch.
        Adapted from original get_loss() but for 2D slices.
        
        Args:
            batch: Dictionary containing 2D slice data
            mode: 'train' or 'val'
        
        Returns:
            loss, mse, dice_metric
        """
        # Prepare batch
        batch = self.prepare_slice_batch(batch)
        
        imgs = batch['image']  # [B, S, C, H, W]
        labels = batch['label']  # [B, S, H, W]
        days = batch['days']  # [B, T] where T is timepoints
        treatments = batch['treatment']  # [B, T]
        
        b, s, c, h, w = imgs.shape
        n_sess = s
        
        # Handle temporal conditioning (expects 4 timepoints)
        n_timepoints = days.shape[1]
        
        if n_timepoints >= 4:
            s1_days, s2_days, s3_days, t_days = days[:, 0], days[:, 1], days[:, 2], days[:, 3]
            treat1, treat2, treat3, treat_t = treatments[:, 0], treatments[:, 1], treatments[:, 2], treatments[:, 3]
        elif n_timepoints == 3:
            s1_days, s2_days, s3_days = days[:, 0], days[:, 1], days[:, 2]
            t_days = s3_days
            treat1, treat2, treat3 = treatments[:, 0], treatments[:, 1], treatments[:, 2]
            treat_t = treat3
        elif n_timepoints == 2:
            s1_days, s2_days = days[:, 0], days[:, 1]
            s3_days = t_days = s2_days
            treat1, treat2 = treatments[:, 0], treatments[:, 1]
            treat3 = treat_t = treat2
        else:
            s1_days = s2_days = s3_days = t_days = days[:, 0]
            treat1 = treat2 = treat3 = treat_t = treatments[:, 0]
        
        # Build conditioning vectors
        intvs = [s1_days.to(torch.float32), s2_days.to(torch.float32),
                 s3_days.to(torch.float32), t_days.to(torch.float32)]
        treat_cond = [treat1.to(torch.float32), treat2.to(torch.float32),
                      treat3.to(torch.float32), treat_t.to(torch.float32)]
        
        # Target is last session
        i_tg = -torch.ones((b,), dtype=torch.int64, device=self.device)
        
        # Extract target
        idx_b = torch.arange(b, device=imgs.device)
        idx_s = i_tg.to(imgs.device).long()
        
        gt_img = imgs[idx_b, idx_s, ...].to(torch.float32)  # [B, C, H, W]
        
        # Handle label - need to add channel dimension for consistency
        if labels.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            labels = labels.unsqueeze(1)
        elif labels.dim() == 4 and labels.shape[1] != 4:
            # [B, S, H, W] -> need to broadcast to [B, 4, H, W]
            # Use last session as target
            gt_label = labels[idx_b, idx_s, ...]  # [B, H, W]
            gt_label = gt_label.unsqueeze(1)  # [B, 1, H, W]
            # Replicate to 4 channels for compatibility
            gt_label = gt_label.repeat(1, 4, 1, 1)  # [B, 4, H, W]
        else:
            gt_label = labels[idx_b, idx_s, ...]  # [B, 4, H, W]
        
        # Sample diffusion timestep
        t = torch.randint(1, self.diffusion.T + 1, [gt_img.shape[0]], device=self.device)
        w_tg = self.alphabar[t - 1]
        
        # Add noise to target
        xt, epsilon = self.diffusion.sample(gt_img.to(torch.float32), t)
        
        # Prepare input
        imgs_input = imgs.clone()
        if labels.shape[1] != 4:
            # Create 4-channel label from single channel
            labels_input = labels.repeat(1, 4, 1, 1) if labels.shape[1] == 1 else labels
        else:
            labels_input = labels.clone()
        
        maskout_batch = (s3_days == t_days)
        
        for i, j in zip(range(b), i_tg):
            if maskout_batch[i]:
                imgs_input[i, :, :, :, :] = 0.
                labels_input[i, :, :, :] = 0
            if labels_input.shape[1] == n_sess:
                labels_input[i, j, :, :] = gt_label[i, 0, :, :]  # Use first channel
            imgs_input[i, j, :, :, :] = xt[i, :, :, :]
        
        # Reshape for model input: [B, S, C, H, W] -> [B, S*C, H, W]
        xt_reshaped = imgs_input.reshape(b, s * c, h, w).contiguous()
        
        # Forward pass
        t = t.view(gt_img.shape[0]).to(self.device)
        out = self.forward(
            xt_reshaped.to(torch.float32),
            t.to(torch.float32),
            intv_t=intvs,
            treat_code=treat_cond,
            i_tg=i_tg
        )
        
        # Compute losses
        img_pred = out[:, 4:7, :, :]  # Predicted image (3 channels)
        mask_pred = out[:, 0:4, :, :]  # Predicted mask (4 channels)
        
        # Image reconstruction loss with spatial weighting
        loss_weights = torch.sum(labels_input, dim=1, keepdim=True)
        loss_weights = loss_weights * torch.exp(-loss_weights)
        
        # Apply dilation for spatial weighting
        if hasattr(self, 'dilation_filters'):
            loss_weights = torch.nn.functional.conv2d(
                loss_weights,
                self.dilation_filters.to(loss_weights.device),
                padding='same'
            ) + 1.0
        else:
            loss_weights = loss_weights + 1.0
        
        loss1 = torch.mean(loss_weights * (img_pred - epsilon) ** 2)
        mse = self.loss_function(img_pred, epsilon)
        
        # Dice loss
        dice_loss = self.dice(mask_pred, labels_input).squeeze()
        
        # Weight dice loss by noise level
        w_tg = torch.from_numpy(w_tg).to(self.device)
        for i, j in zip(range(b), i_tg):
            if maskout_batch[i]:
                loss_ij = dice_loss[i, j] * torch.sqrt(w_tg[i])
                dice_loss[i, :] = 0.
                dice_loss[i, j] = loss_ij
            else:
                dice_loss[i, j] = dice_loss[i, j] * torch.sqrt(w_tg[i])
        
        loss = loss1 + torch.mean(dice_loss) * self.cfg.aux_loss_w
        
        # Calculate dice metric
        mask_pred_binary = torch.sigmoid(mask_pred)
        mask_pred_binary = (mask_pred_binary > 0.5) * 1
        self.dice_metric(mask_pred_binary, labels_input)
        dice_last = self.dice_metric.aggregate()
        self.dice_metric.reset()
        
        return loss, mse, dice_last
    
    def training_step(self, batch, batch_idx):
        """Training step using slice-wise processing."""
        loss, mse, dice = self.get_loss_slicewise(batch, mode='train')
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step using slice-wise processing."""
        loss, mse, dice = self.get_loss_slicewise(batch, mode='val')
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss


def load_patient_splits(splits_file: Path) -> Dict[str, List[str]]:
    """Load train/val/test splits from JSON file."""
    import json
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    return splits


def get_patient_files(patient_ids: List[str], data_dir: Path) -> List[Dict]:
    """Get file dictionaries for list of patient IDs."""
    file_list = []
    for patient_id in patient_ids:
        file_dict = {
            key: str(data_dir / f'{patient_id}_{key}.npy')
            for key in npz_keys
        }
        # Verify files exist
        if all(Path(file_dict[key]).exists() for key in npz_keys):
            file_list.append(file_dict)
        else:
            print(f"Warning: Missing files for patient {patient_id}")
    return file_list


def setup_callbacks(config):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logdir,
        filename=config.ckpt_filename,
        monitor=config.ckpt_monitor,
        mode=config.ckpt_mode,
        save_top_k=config.ckpt_save_top_k,
        save_last=config.ckpt_save_last,
        verbose=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config):
    """Setup WandB logger."""
    if config.logdir == 'wandb':
        logger = WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.exp_name,
            save_dir=config.logdir
        )
        return logger
    return None


def main():
    # Parse arguments
    config = load_args(default_config)
    
    print("\n" + "="*70)
    print("TaDiff Training - Slice-wise Processing")
    print("="*70)
    print(f"Data directory: {config.data_dir[config.data_pool[0]]}")
    print(f"Batch size: {config.batch_size}")
    print(f"GPU devices: {config.gpu_devices}")
    print(f"Max epochs: {config.max_epochs}")
    print("="*70 + "\n")
    
    # Load data splits
    splits_file = Path('./data/splits') / f'{config.data_pool[0]}_splits.json'
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}\n"
            f"Run: python scripts/prepare_data_splits.py first"
        )
    
    splits = load_patient_splits(splits_file)
    print(f"Loaded splits:")
    print(f"  Train: {len(splits['train'])} patients")
    print(f"  Val: {len(splits['val'])} patients")
    print(f"  Test: {len(splits['test'])} patients\n")
    
    # Get file lists
    data_dir = Path(config.data_dir[config.data_pool[0]])
    train_files = get_patient_files(splits['train'], data_dir)
    val_files = get_patient_files(splits['val'], data_dir)
    
    print(f"Valid files found:")
    print(f"  Train: {len(train_files)} patients")
    print(f"  Val: {len(val_files)} patients\n")
    
    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError("No valid training or validation files found!")
    
    # Create data module
    data_module = TaDiffDataModule(
        train_files=train_files,
        val_files=val_files,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        min_tumor_voxels=100
    )

    
    # Initialize model
    model = SliceWiseTaDiffModel(config)
    
# Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # Configure trainer strategy
    if ',' in str(config.gpu_devices):
        # Multi-GPU training
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
        devices = [int(x.strip()) for x in config.gpu_devices.split(',')]
    else:
        # Single GPU training
        strategy = 'auto'
        devices = [int(config.gpu_devices)]
    
    print("\n" + "="*60)
    print("Setting up Trainer...")
    print("="*60)
    print(f"  Devices: {devices}")
    print(f"  Strategy: {strategy}")
    print(f"  Precision: {config.precision}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Accumulate grad batches: {config.accumulate_grad_batches}")
    
    
    # Setup trainer
    trainer = pl.Trainer(
        # Hardware
        accelerator=config.gpu_accelerator,
        devices=[int(config.gpu_devices)],
        strategy='auto',
        
        # Training duration
        max_epochs=config.max_epochs if config.max_epochs > 0 else None,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        
        # Precision
        precision=config.precision,
        
        # Logging
        logger=logger,
        log_every_n_steps=config.log_interval,
        
        # Callbacks
        callbacks=callbacks,
        
        # Validation
        check_val_every_n_epoch=config.val_interval_epoch,
        
        # Optimization
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.grad_clip,
        
        # Checkpointing
        enable_checkpointing=True,
        
        # Reproducibility
        deterministic=False,  # Set to True for full reproducibility (slower)
        benchmark=True,  # cudnn benchmark for faster training
    )

  
    
    # Print training info
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    # Train
    if config.resume_from_ckpt and config.ckpt_best_or_last:
        print(f"Resuming from checkpoint: {config.ckpt_best_or_last}\n")
        trainer.fit(model, data_module, ckpt_path=config.ckpt_best_or_last)
    else:
        trainer.fit(model, data_module)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best model saved to: {config.logdir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()