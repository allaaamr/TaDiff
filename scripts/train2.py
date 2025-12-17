"""
TaDiff Training Script - Matches test.py structure

Simple training that:
1. Loads volumes patient by patient (like test.py)
2. Identifies top-k tumor slices per session
3. Calls model.get_loss() for each slice (already implemented!)
4. No complex dataset classes - just use DataLoader

Matches test.py structure but adds training loop.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from monai.data import CacheDataset, DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.cfg_tadiff_net import config as default_config
from config.arg_parse import load_args
from src.tadiff_model import Tadiff_model
from src.data.data_loader import val_transforms, npz_keys
from src.evaluation.metrics import calculate_tumor_volumes, get_slice_indices
from src.utils.image_processing import prepare_image_batch


def process_slice_train(
    slice_idx: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    days: torch.Tensor,
    treatments: torch.Tensor,
    model: Tadiff_model,
    optimizer: torch.optim.Optimizer,
    mode: str = 'train'
) -> Dict[str, float]:
    """
    Process a single 2D slice for training/validation.
    Simply extracts the slice and calls model.get_loss()!
    
    Args:
        slice_idx: Z-index of slice
        images: [1, S, C, H, W, D] - Full 3D volume
        labels: [1, S, H, W, D] - Full 3D labels (single channel per session)
        days: [1, 4] - Time points
        treatments: [1, 4] - Treatment codes
        model: TaDiff model
        optimizer: Optimizer
        mode: 'train' or 'val'
        
    Returns:
        Dict with loss, mse, dice
    """
    # Extract 2D slice
    imgs_slice = images[..., slice_idx]  # [1, S, C, H, W]
    labels_slice = labels[..., slice_idx]  # [1, S, H, W]
    
    # Convert from MONAI MetaTensor to regular torch.Tensor
    imgs_slice = torch.as_tensor(imgs_slice).clone()
    labels_slice = torch.as_tensor(labels_slice).clone()
    days = torch.as_tensor(days).clone()
    treatments = torch.as_tensor(treatments).clone()
    
    # Get number of sessions and timepoints
    n_sessions = imgs_slice.shape[1]
    n_timepoints = days.shape[1]
    
    print(f"\n[BEFORE SELECTION] slice_idx={slice_idx}")
    print(f"  imgs_slice: {imgs_slice.shape}, labels_slice: {labels_slice.shape}")
    print(f"  days: {days.shape}, treatments: {treatments.shape}")
    
    # Model expects exactly 4 sessions and 4 timepoints
    # Select last 4 sessions (most recent)
    if n_sessions > 4:
        imgs_slice = imgs_slice[:, -4:, :, :, :]  # [1, 4, C, H, W]
        labels_slice = labels_slice[:, -4:, :, :]  # [1, 4, H, W]
    elif n_sessions < 4:
        # Pad by repeating last session
        pad_sessions = 4 - n_sessions
        imgs_slice = torch.cat([
            imgs_slice,
            imgs_slice[:, -1:, :, :, :].repeat(1, pad_sessions, 1, 1, 1)
        ], dim=1)
        labels_slice = torch.cat([
            labels_slice,
            labels_slice[:, -1:, :, :].repeat(1, pad_sessions, 1, 1)
        ], dim=1)
    
    # Select corresponding timepoints (last 4)
    if n_timepoints > 4:
        days = days[:, -4:]  # [1, 4]
        treatments = treatments[:, -4:]  # [1, 4]
    elif n_timepoints < 4:
        # Pad by repeating last timepoint
        pad_timepoints = 4 - n_timepoints
        days = torch.cat([days, days[:, -1:].repeat(1, pad_timepoints)], dim=1)
        treatments = torch.cat([treatments, treatments[:, -1:].repeat(1, pad_timepoints)], dim=1)
    
    print(f"[AFTER SELECTION]")
    print(f"  imgs_slice: {imgs_slice.shape}, labels_slice: {labels_slice.shape}")
    print(f"  days: {days.shape}, treatments: {treatments.shape}")
    
    # Expand labels to 4 channels (model expects [B, S, 4, H, W])
    # labels_slice = labels_slice.unsqueeze(2).repeat(1, 1, 4, 1, 1)  # [1, 4, 4, H, W]
    
    # Create batch dict (what get_loss expects)
    batch = {
        'image': imgs_slice,
        'label': labels_slice,
        'days': days,
        'treatments': treatments
    }
    
    # Call the existing get_loss() - it does everything!
    if mode == 'train':
        model.train()
        loss, mse, dice = model.get_loss(batch, mode='train')
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if hasattr(model.cfg, 'grad_clip') and model.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.grad_clip)
        
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            loss, mse, dice = model.get_loss(batch, mode='val')
    
    return {
        'loss': loss.item(),
        'mse': mse.item(),
        'dice': dice.item()
    }


def process_session_train(
    batch: Dict[str, torch.Tensor],
    model: Tadiff_model,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    top_k: int = 3,
    mode: str = 'train'
) -> Dict[str, float]:
    """
    Process a session - matches test.py's process_session() but for training.
    
    Args:
        batch: Dictionary with 'image', 'label', 'days', 'treatment'
        model: TaDiff model
        optimizer: Optimizer
        device: Device
        top_k: Number of slices to process
        mode: 'train' or 'val'
        
    Returns:
        Averaged metrics across slices
    """
    # Extract tensors
    labels = batch['label'].to(device)  # [1, S, D, H, W] from MONAI
    images = batch['image'].to(device)  # [1, C*S, D, H, W] from MONAI
    days = batch['days'].to(device)
    treatments = batch['treatment'].to(device)
    
    # Reorder dimensions: [B, ..., D, H, W] -> [B, ..., H, W, D]
    images = images.permute(0, 1, 3, 4, 2)  # [1, C*S, H, W, D]
    labels = labels.permute(0, 1, 3, 4, 2)  # [1, S, H, W, D]
    
    # Calculate tumor volumes per slice
    n_sessions = labels.shape[1]
    z_mask_size = calculate_tumor_volumes(labels[0, :, :, :, :])  # [D]
    
    # Get top-k slices with most tumor
    top_k_indices = get_slice_indices(z_mask_size, top_k=top_k)
    
    # Prepare images: [1, C*S, H, W, D] -> [1, S, C, H, W, D]
    images = prepare_image_batch(images, n_sessions)
    
    # Process each slice
    slice_metrics = []
    for slice_idx in top_k_indices:
        metrics = process_slice_train(
            slice_idx=slice_idx.item(),
            images=images,
            labels=labels,
            days=days,
            treatments=treatments,
            model=model,
            optimizer=optimizer if mode == 'train' else None,
            mode=mode
        )
        slice_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        'loss': np.mean([m['loss'] for m in slice_metrics]),
        'mse': np.mean([m['mse'] for m in slice_metrics]),
        'dice': np.mean([m['dice'] for m in slice_metrics])
    }
    
    return avg_metrics


def train_epoch(
    model: Tadiff_model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_metrics = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        metrics = process_session_train(
            batch=batch,
            model=model,
            optimizer=optimizer,
            device=device,
            top_k=3,
            mode='train'
        )
        epoch_metrics.append(metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'dice': f"{metrics['dice']:.4f}"
        })
    
    # Average metrics
    avg_metrics = {
        'loss': np.mean([m['loss'] for m in epoch_metrics]),
        'mse': np.mean([m['mse'] for m in epoch_metrics]),
        'dice': np.mean([m['dice'] for m in epoch_metrics])
    }
    
    return avg_metrics


def validate_epoch(
    model: Tadiff_model,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    epoch_metrics = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, batch in enumerate(pbar):
            metrics = process_session_train(
                batch=batch,
                model=model,
                optimizer=None,
                device=device,
                top_k=3,
                mode='val'
            )
            epoch_metrics.append(metrics)
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })
    
    # Average metrics
    avg_metrics = {
        'loss': np.mean([m['loss'] for m in epoch_metrics]),
        'mse': np.mean([m['mse'] for m in epoch_metrics]),
        'dice': np.mean([m['dice'] for m in epoch_metrics])
    }
    
    return avg_metrics


def load_patient_splits(splits_file: Path) -> Dict[str, List[str]]:
    """Load train/val/test splits from JSON."""
    import json
    with open(splits_file, 'r') as f:
        return json.load(f)


def get_patient_files(patient_ids: List[str], data_dir: Path) -> List[Dict]:
    """Get file dictionaries for patients."""
    file_list = []
    for patient_id in patient_ids:
        file_dict = {
            key: str(data_dir / f'{patient_id}_{key}.npy')
            for key in npz_keys
        }
        if all(Path(file_dict[key]).exists() for key in npz_keys):
            file_list.append(file_dict)
        else:
            print(f"Warning: Missing files for {patient_id}")
    return file_list


def main():
    # Load config
    config = load_args(default_config)
    
    # Setup device
    device = torch.device(f'cuda:{config.gpu_devices}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\n" + "="*70)
    print("TaDiff Training - Simple Approach (uses get_loss())")
    print("="*70)
    print(f"Data directory: {config.data_dir[config.data_pool[0]]}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Learning rate: {config.lr}")
    print("="*70 + "\n")
    
    # Load splits
    splits_file = Path('./data/splits') / f'{config.data_pool[0]}_splits.json'
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    
    splits = load_patient_splits(splits_file)
    print(f"Loaded splits:")
    print(f"  Train: {len(splits['train'])} patients")
    print(f"  Val: {len(splits['val'])} patients\n")
    
    # Get file lists
    data_dir = Path(config.data_dir[config.data_pool[0]])
    train_files = get_patient_files(splits['train'], data_dir)
    val_files = get_patient_files(splits['val'], data_dir)
    
    print(f"Valid files:")
    print(f"  Train: {len(train_files)} patients")
    print(f"  Val: {len(val_files)} patients\n")
    
    # Create dataloaders (no caching - like test.py)
    train_dataset = CacheDataset(data=train_files, transform=val_transforms, cache_rate=0.0)
    val_dataset = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Initialize model
    model = Tadiff_model(config).to(device)
    
    print(f"\nModel initialized:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=getattr(config, 'weight_decay', 0.01)
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.lr * 0.01
    )
    
    # Training loop
    best_val_dice = 0.0
    os.makedirs(config.logdir, exist_ok=True)
    
    print("="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    for epoch in range(config.max_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"\nEpoch {epoch} [Train] - Loss: {train_metrics['loss']:.4f}, "
              f"MSE: {train_metrics['mse']:.4f}, Dice: {train_metrics['dice']:.4f}")
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch)
        print(f"Epoch {epoch} [Val]   - Loss: {val_metrics['loss']:.4f}, "
              f"MSE: {val_metrics['mse']:.4f}, Dice: {val_metrics['dice']:.4f}\n")
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
            }, os.path.join(config.logdir, 'best.ckpt'))
            print(f"✓ Saved best model (dice: {best_val_dice:.4f})")
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_metrics['dice'],
        }, os.path.join(config.logdir, 'last.ckpt'))
        
        # Step scheduler
        scheduler.step()
        
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Models saved to: {config.logdir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()