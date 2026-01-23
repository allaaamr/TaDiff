"""
TaDiff Training Script 
1. Loads volumes patient by patient 
2. Identifies top-k tumor slices per session
3. Calls model.get_loss() for each slice 

"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import wandb
from monai.data import CacheDataset, DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.cfg_tadiff_net import config as default_config
from config.arg_parse import load_args
from src.tadiff_model import Tadiff_model
from src.data.data_loader import val_transforms, non_load_val_transforms,  npz_keys
from src.evaluation.metrics import calculate_tumor_volumes, get_slice_indices
from src.utils.image_processing import prepare_image_batch
from src.data.datasampler import *

from torch.utils.data import Dataset

import os
import matplotlib.pyplot as plt

def save_gray(img2d, path):
    arr = img2d.detach().float().cpu().numpy()
    plt.imsave(path, arr, cmap="gray")

@torch.no_grad()
def reconstruct_x0_from_eps(x_t, eps_pred, alphabar_t, eps=1e-8):
    # x_t, eps_pred: (B,3,H,W)
    # alphabar_t: (B,1,1,1)
    return (x_t - torch.sqrt(1 - alphabar_t) * eps_pred) / (torch.sqrt(alphabar_t) + eps)

@torch.no_grad()
def visualize_one_val_sample(model, diffusion, batch, slice_idx, device, epoch, out_root="debug_vis", t_vis=500):
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    os.makedirs(out_dir, exist_ok=True)

    model.eval()

    images = batch["image"].to(device)   # expected (1,S,C,H,W,D)
    labels = batch["label"].to(device)
    days = batch["days"].to(device)
    treatments = (batch["treatment"] if "treatment" in batch else batch["treatments"]).to(device)
    geno = batch["geno"].to(device)

    # extract 2D slice
    imgs2d = images[..., slice_idx]  # (1,S,C,H,W)
    lbls2d = labels[..., slice_idx]  # (1,S,H,W)

    B, S, C, H, W = imgs2d.shape

    # target is last session (training behavior)
    i_tg = torch.full((B,), -1, dtype=torch.long, device=device)
    idx_b = torch.arange(B, device=device)
    x0 = imgs2d[idx_b, i_tg, ...].float()  # (1,C,H,W)
    y0 = lbls2d[idx_b, i_tg, ...].float()  # (1,H,W)

    # fixed timestep for visualization
    t_int = int(min(max(1, t_vis), diffusion.T))
    t = torch.full((B,), t_int, dtype=torch.long, device=device)

    # noise (make sure diffusion is on the same device)
    # your diffusion.sample uses alphabar indexing with t-1; if it expects CPU t, pass t.cpu()
    x_t, eps = diffusion.sample(x0, t.cpu())
    x_t = x_t.to(device)
    eps = eps.to(device)

    # build model input: replace target session with x_t, then flatten
    imgs_in = imgs2d.clone().float()
    imgs_in[idx_b, i_tg, ...] = x_t
    x_in = imgs_in.reshape(B, S*C, H, W).contiguous()

    # conditioning (same as your get_loss)
    s1_days, s2_days, s3_days, t_days = days[:, 0], days[:, 1], days[:, 2], days[:, 3]
    tr1, tr2, tr3, trt = treatments[:, 0], treatments[:, 1], treatments[:, 2], treatments[:, 3]
    intvs = [s1_days.float(), s2_days.float(), s3_days.float(), t_days.float()]
    treat_cond = [tr1.float(), tr2.float(), tr3.float(), trt.float()]

    pred = model(x_in, t.float(), intv_t=intvs, treat_code=treat_cond, geno=geno, i_tg=i_tg)
    eps_pred = pred[:, 4:7, :, :]  # (1,3,H,W)
    mask_pred = pred[:, 0:4, :, :] # (1,4,H,W)

    alphabar_t = diffusion.alphabar[t_int - 1].to(device).view(B, 1, 1, 1)
    x0_hat = reconstruct_x0_from_eps(x_t, eps_pred, alphabar_t)

    # Save the 3 modalities
    modal_names = ["t1", "t1c", "flair"] if C == 3 else [f"m{m}" for m in range(C)]
    for m, name in enumerate(modal_names[:C]):
        save_gray(x0[0, m],    os.path.join(out_dir, f"x0_gt_{name}.png"))
        save_gray(x_t[0, m],   os.path.join(out_dir, f"xt_noisy_t{t_int}_{name}.png"))
        save_gray(x0_hat[0, m],os.path.join(out_dir, f"x0hat_{name}.png"))

    # Optional: save mask prediction vs GT
    # pick one channel, or argmax if multi-class; here just save channel 3 as example
    save_gray(torch.sigmoid(mask_pred[0, -1]), os.path.join(out_dir, "mask_pred_ch3.png"))
    # GT might be not one-hot; adapt as needed:
    # if y0 is integer mask, just save it
    save_gray(y0[0], os.path.join(out_dir, "mask_gt.png"))


def grad_norm(p):
    return 0.0 if (p.grad is None) else p.grad.data.norm().item()

def compute_geno_stats(file_dicts, eps=1e-6):
    all_g = []
    for fd in file_dicts:
        g = np.load(fd["geno"]).astype(np.float32)
        all_g.append(g[None, :])
    G = np.concatenate(all_g, axis=0)  # [N_patients, G]
    mean = G.mean(axis=0)
    std = G.std(axis=0)
    std = np.maximum(std, eps)
    return mean, std

def process_slice_train(
    slice_idx: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    days: torch.Tensor,
    treatments: torch.Tensor,
    model: Tadiff_model,
    optimizer: torch.optim.Optimizer,
    geno: torch.Tensor,
    epoch: int,
    flag: bool,
    mode: str = 'train',
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
    geno = torch.as_tensor(geno).clone()
    
    # Get number of sessions and timepoints
    n_sessions = imgs_slice.shape[1]
    n_timepoints = days.shape[1]
    
    batch = {
        'image': imgs_slice,
        'label': labels_slice,
        'days': days,
        'treatments': treatments,
        'geno': geno
    }
    
    # Call the existing get_loss() - it does everything!
    if mode == 'train':
        model.train()
        loss, mse, dice = model.get_loss(batch, epoch, flag, mode='train')
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # for name, p in model.named_parameters():
        #     if "geno_embed" in name or "k_proj" in name or "v_proj" in name or "treats_embed" in name:
        #         print(name, grad_norm(p))
        
        # Gradient clipping
        if hasattr(model.cfg, 'grad_clip') and model.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.grad_clip)
        
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            loss, mse, dice = model.get_loss(batch, epoch,flag, mode='val')
    
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
    epoch: int,
    flag: bool,
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
    labels = batch['label'].to(device)  # [1, S,  H, W, D] from MONAI
    images = batch['image'].to(device)  # [1, C*S, H, W, D] from MONAI
    days = batch['days'].to(device)
    treatments = batch['treatment'].to(device)
    geno = batch['geno'].to(device)

    # print("images.shape in process sess ", images.shape)
    # print("labels.shape in process sess ", labels.shape)


    # # Reorder dimensions: [B, ..., D, H, W] -> [B, ..., H, W, D]
    # images = images.permute(0, 1, 3, 4, 2)  # [1, C*S, H, W, D]
    # labels = labels.permute(0, 1, 3, 4, 2)  # [1, S, H, W, D]
    # print("labels.shape ", labels.shape)

    # Calculate tumor volumes per slice 
    n_sessions = labels.shape[1]
    z_mask_size = calculate_tumor_volumes(labels[0])  # labels[0]: [S, H, W, D]
    
    # Get top-k slices with most tumor
    top_k_indices = get_slice_indices(z_mask_size, top_k=top_k)
    
    # Prepare images: [1, C*S, H, W, D] -> [1, S, C, H, W, D]
    # images = prepare_image_batch(images, n_sessions)
    
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
            mode=mode,
            geno=geno,
            epoch=epoch,
            flag=flag,
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
            epoch=epoch,
            flag= False,
            top_k=3,
            mode='train'
        )
        epoch_metrics.append(metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'loss': f"{metrics['mse']:.4f}",
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
    flag = True

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, batch in enumerate(pbar):
            metrics = process_session_train(
                batch=batch,
                model=model,
                optimizer=None,
                device=device,
                epoch=epoch,
                flag = flag,
                top_k=3,
                mode='val'
            )
            epoch_metrics.append(metrics)
            flag = False
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'mse': f"{metrics['mse']:.4f}",
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
    print("TaDiff Training")
    print("="*70)
    print(f"Data directory: {config.data_dir[config.data_pool[0]]}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Learning rate: {config.lr}")
    print("="*70 + "\n")
    
    # Load splits
    splits_file = Path(config.split_dir) 
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

    wandb.init(
    project="TaDiff",                    # change to your project
    name=f"run_{config.data_pool[0]}",   # or a descriptive name
    config={
        "lr": config.lr,
        "max_epochs": config.max_epochs,
        "data_pool": config.data_pool[0],
    }
    )
    
    # # Create dataloaders (no caching - like test.py)
    # train_dataset = CacheDataset(data=train_files, transform=val_transforms, cache_rate=0.0)
    # val_dataset = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)
    
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create datasets with sliding windows
    # train_dataset = SlidingWindowDataset(train_files, transform=non_load_val_transforms)
    # val_dataset = SlidingWindowDataset(val_files, transform=non_load_val_transforms)
    geno_mean, geno_std = compute_geno_stats(train_files)
    train_dataset = PatientSamplingDataset(train_files, transform=None, samples_per_patient=getattr(config, "samples_per_patient", 1),  geno_mean=geno_mean, geno_std=geno_std, rng_seed=getattr(config, "rng_seed", None))
    val_dataset = PatientSamplingDataset(val_files, transform=None, samples_per_patient=getattr(config, "val_samples_per_patient", 1),  geno_mean=geno_mean, geno_std=geno_std, rng_seed=getattr(config, "rng_seed", None))
    
    print(f"\n{'='*70}")
    print("Sliding Window Statistics:")
    print(f"{'='*70}")
    print(f"Train patients: {len(train_files)} → Training points: {len(train_dataset)}")
    print(f"Val patients: {len(val_files)} → Validation points: {len(val_dataset)}")
    print(f"Avg windows per train patient: {len(train_dataset)/len(train_files):.1f}")
    print(f"Avg windows per val patient: {len(val_dataset)/len(val_files):.1f}")
    print(f"{'='*70}\n")

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
        
        wandb.log({
            "train/loss": train_metrics["loss"],
            "train/mse": train_metrics["mse"],
            "train/dice": train_metrics["dice"],
            "epoch": epoch
        })

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch)
        print(f"Epoch {epoch} [Val]   - Loss: {val_metrics['loss']:.4f}, "
              f"MSE: {val_metrics['mse']:.4f}, Dice: {val_metrics['dice']:.4f}\n")

        wandb.log({
            "val/loss": val_metrics["loss"],
            "val/mse": val_metrics["mse"],
            "val/dice": val_metrics["dice"],
        })
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
        lr_now = scheduler.get_last_lr()[0]
        wandb.log({"lr": lr_now})
        
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Models saved to: {config.logdir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()