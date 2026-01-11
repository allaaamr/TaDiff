"""
SADM.py - Sequence-Aware Diffusion Model for 2D Longitudinal Medical Images

ORIGINAL ROLE (3D SADM):
- Main training and inference script for SADM
- Loads ACDC cardiac dataset (3D volumes)
- Trains ViViT + DDPM end-to-end

CHANGES FOR 2D ADAPTATION:
1. Data loading: Uses your PatientSamplingDataset for 2D slices
2. Input format: (B, S, C, H, W) instead of (B, L, C, H, W, D)
3. Added support for genomic features, treatments, days
4. Loss: MSE for noise + Dice for segmentation
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(str(Path(__file__).parent))
from sadm_net.DDPM.ddpm import ContextUNet2D, DDPM
from sadm_net.ViViT.vivit import SequenceAwareViT2D
from data.datasampler import PatientSamplingDataset
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

# ============================================================================
# SADM MODEL
# ============================================================================

class SADM(nn.Module):
    """
    Sequence-Aware Diffusion Model.
    
    Combines:
    - SequenceAwareViT2D: Processes longitudinal history into conditioning
    - ContextUNet2D + DDPM: Diffusion model for image generation
    
    Args:
        img_size: Input image size
        in_channels: Number of image channels (e.g., 3 for T1, T1c, FLAIR)
        out_channels: Output channels (image + segmentation)
        embed_dim: Transformer embedding dimension
        model_channels: UNet base channels
        ... (see individual components for more)
    """
    
    def __init__(
        self,
        img_size: int = 192,
        patch_size: int = 8,
        in_channels: int = 3,
        out_channels: int = 7,
        embed_dim: int = 256,
        model_channels: int = 64,
        channel_mult: Tuple = (1, 2, 4, 8),
        temporal_depth: int = 4,
        spatial_depth: int = 4,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        attention_resolutions: Tuple = (16, 8),
        max_seq_len: int = 4,
        n_T: int = 1000,
        use_geno: bool = True,
        geno_dim: int = 16,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.device = device
        
        # Sequence-Aware Transformer for conditioning
        self.sat = SequenceAwareViT2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            temporal_depth=temporal_depth,
            spatial_depth=spatial_depth,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_labels=True,
            use_geno=use_geno,
            geno_dim=geno_dim,
        )
        
        # UNet for diffusion
        self.unet = ContextUNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_heads=num_heads,
            cond_dim=embed_dim,
            image_size=img_size,
        )
        
        # DDPM wrapper
        self.ddpm = DDPM(
            nn_model=self.unet,
            betas=(1e-4, 0.02),
            n_T=n_T,
            device=device,
            drop_prob=0.1,
        )
        
        # Loss functions
        self.dice_loss = DiceLoss(sigmoid=True, reduction='none') if HAS_MONAI else None
        self.dice_metric = DiceMetric(include_background=True, reduction='mean') if HAS_MONAI else None
        
        # Alpha bar for loss weighting
        alphabar = np.cumprod(1 - np.linspace(1e-4, 0.02, n_T))
        self.register_buffer('alphabar', torch.tensor(alphabar, dtype=torch.float32))
        
        # Dilation filter for loss weighting
        self.register_buffer('dilation_filter', torch.ones(1, 1, 11, 11) / 121.)
    
    def get_conditioning(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get conditioning from SAT."""
        cond_spatial, cond_tokens = self.sat(
            images=batch['image'],
            labels=batch.get('label', None),
            days=batch.get('days', None),
            treatments=batch.get('treatment', batch.get('treatments', None)),
            geno=batch.get('geno', None),
        )
        cond_global = F.adaptive_avg_pool2d(cond_spatial, 1).flatten(1)
        return cond_global, cond_tokens
    
    def get_loss(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss.
        
        Returns:
            loss: Total loss
            mse: MSE loss (noise prediction)
            dice: Dice score
        """
        images = batch['image']
        labels = batch['label']
        print(images.shape)
        B, S, C, H, W = images.shape
        device = images.device
        
        # Target is last timepoint
        target_img = images[:, -1]  # (B, C, H, W)
        target_label = labels[:, -1]  # (B, H, W)
        
        # Get conditioning from history (all but last)
        hist_batch = {
            'image': images[:, :-1],
            'label': labels[:, :-1],
            'days': batch.get('days', torch.zeros(B, S))[:, :-1],
            'treatment': batch.get('treatment', torch.zeros(B, S).long())[:, :-1],
            'geno': batch.get('geno', None),
        }
        cond_global, cond_tokens = self.get_conditioning(hist_batch)
        
        # Sample timestep
        t = torch.randint(0, self.ddpm.n_T, (B,), device=device)
        
        # Add noise
        noise = torch.randn_like(target_img)
        x_t = self.ddpm.q_sample(target_img, t, noise)
        
        # Predict
        pred = self.unet(x_t, t, cond_global, cond_tokens)
        pred_noise = pred[:, :self.in_channels]
        pred_seg = pred[:, self.in_channels:]
        
        # MSE loss on noise
        label_weight = target_label.unsqueeze(1).float()
        loss_weight = label_weight * torch.exp(-label_weight)
        loss_weight = F.conv2d(loss_weight, self.dilation_filter.to(device), padding='same') + 1.
        
        mse_weighted = torch.mean(loss_weight * (pred_noise - noise) ** 2)
        mse = F.mse_loss(pred_noise, noise)
        
        # Dice loss on segmentation
        if target_label.dim() == 3:
            target_label_expanded = target_label.unsqueeze(1).expand(-1, pred_seg.shape[1], -1, -1)
        else:
            target_label_expanded = target_label
        
        if self.dice_loss is not None:
            dice_loss = self.dice_loss(pred_seg, target_label_expanded.float())
            w_t = self.alphabar[t].sqrt()
            dice_loss = dice_loss * w_t.view(B, 1, 1, 1)
        else:
            dice_loss = torch.tensor(0.0, device=device)
        
        loss = mse_weighted + dice_loss.mean()
        
        # Dice metric
        if self.dice_metric is not None:
            pred_binary = (torch.sigmoid(pred_seg) > 0.5).float()
            self.dice_metric(pred_binary, target_label_expanded.float())
            dice_score = self.dice_metric.aggregate()
            self.dice_metric.reset()
        else:
            dice_score = torch.tensor(0.0, device=device)
        
        return loss, mse, dice_score
    
    @torch.no_grad()
    def sample(self, batch: dict, num_steps: int = 100) -> torch.Tensor:
        """Generate samples given conditioning."""
        images = batch['image']
        B, S, C, H, W = images.shape
        
        cond_global, cond_tokens = self.get_conditioning(batch)
        
        self.ddpm.n_T = num_steps
        samples = self.ddpm.sample(
            (B, self.in_channels, H, W),
            cond_global,
            cond_tokens,
        )
        
        return samples


# # ============================================================================
# # TRAINING
# # ============================================================================

# def train_epoch(model, dataloader, optimizer, device, epoch):
#     """Train for one epoch."""
#     model.train()
#     metrics = []
    
#     pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
#     for batch in pbar:
#         batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
#         optimizer.zero_grad()
#         loss, mse, dice = model.get_loss(batch)
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
        
#         metrics.append({'loss': loss.item(), 'mse': mse.item(), 'dice': dice.item()})
#         pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{mse.item():.4f}", dice=f"{dice.item():.4f}")
    
#     return {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}


# def validate_epoch(model, dataloader, device, epoch):
#     """Validate for one epoch."""
#     model.eval()
#     metrics = []
    
#     with torch.no_grad():
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
#         for batch in pbar:
#             batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             loss, mse, dice = model.get_loss(batch)
#             metrics.append({'loss': loss.item(), 'mse': mse.item(), 'dice': dice.item()})
#             pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")
    
#     return {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}


# def main():
#     parser = argparse.ArgumentParser(description='Train SADM for 2D medical images')
#     parser.add_argument('--data_dir', type=str, default='/home/alaa.mohamed/TaDiff/data/miu', help='Data directory')
#     parser.add_argument('--splits_file', type=str, default='/home/alaa.mohamed/TaDiff/data/splits/miu/miu_splits.json', help='Train/val splits')
#     parser.add_argument('--logdir', type=str, default='./logs/sadm', help='Log directory')
#     parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
#     parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
#     parser.add_argument('--img_size', type=int, default=198, help='Image size')
#     parser.add_argument('--in_channels', type=int, default=3, help='Input channels')
#     parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
#     parser.add_argument('--model_channels', type=int, default=32, help='UNet base channels')
#     parser.add_argument('--n_T', type=int, default=1000, help='Diffusion timesteps')
#     parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
#     parser.add_argument('--device', type=str, default='cuda:0', help='Device')
#     args = parser.parse_args()
    
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     os.makedirs(args.logdir, exist_ok=True)
    
#     # Load splits
#     if os.path.exists(args.splits_file):
#         with open(args.splits_file) as f:
#             splits = json.load(f)
#         print(f"Loaded splits: {len(splits['train'])} train, {len(splits['val'])} val")
#     else:
#         print("No splits file found. Using dummy data for testing.")
#         splits = {'train': [], 'val': []}
    

    
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {n_params:,}")
#     print(f"SAT parameters: {sum(p.numel() for p in model.sat.parameters()):,}")
#     print(f"UNet parameters: {sum(p.numel() for p in model.unet.parameters()):,}")
    
#     # Optimizer
#     optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
#     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
#     # Wandb
#     if args.use_wandb and HAS_WANDB:
#         wandb.init(project="SADM-2D", config=vars(args))
    
#     # Training loop
#     best_val_dice = 0.0
    
#     print("\n" + "="*70)
#     print("Starting Training")
#     print("="*70)
    
#     # For testing without data:
#     if len(splits['train']) == 0:
#         print("\nNo training data. Running model test...")
#         B, S, C, H, W = 2, 4, args.in_channels, args.img_size, args.img_size
#         batch = {
#             'image': torch.randn(B, S, C, H, W).to(device),
#             'label': torch.randint(0, 2, (B, S, H, W)).float().to(device),
#             'days': torch.tensor([[0, 30, 60, 90], [0, 45, 90, 135]]).float().to(device),
#             'treatment': torch.randint(0, 5, (B, S)).to(device),
#             'geno': torch.randn(B, 16).to(device),
#         }
        
#         loss, mse, dice = model.get_loss(batch)
#         print(f"Test loss: {loss.item():.4f}, MSE: {mse.item():.4f}, Dice: {dice.item():.4f}")
        
#         print("\nModel test passed! Ready for training with real data.")
#         return
    
#     # With real data:
#     npz_keys = ['image', 'label', 'days', 'treatment', 'geno']
#     train_files = [
#         {k: os.path.join(args.data_dir, f"{pid}_{k}.npy") for k in npz_keys}
#         for pid in splits['train']
#     ]
#     val_files = [
#         {k: os.path.join(args.data_dir, f"{pid}_{k}.npy") for k in npz_keys}
#         for pid in splits['val']
#     ]
    
#     train_dataset = PatientSamplingDataset(train_files, samples_per_patient=50)
#     val_dataset = PatientSamplingDataset(val_files, samples_per_patient=10)
    
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
#     for epoch in range(args.epochs):
#         train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
#         val_metrics = validate_epoch(model, val_loader, device, epoch)
        
#         scheduler.step()
        
#         print(f"\nEpoch {epoch}:")
#         print(f"  Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, Dice: {train_metrics['dice']:.4f}")
#         print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, Dice: {val_metrics['dice']:.4f}")
        
#         if args.use_wandb and HAS_WANDB:
#             wandb.log({
#                 'train/loss': train_metrics['loss'],
#                 'train/mse': train_metrics['mse'],
#                 'train/dice': train_metrics['dice'],
#                 'val/loss': val_metrics['loss'],
#                 'val/mse': val_metrics['mse'],
#                 'val/dice': val_metrics['dice'],
#                 'lr': scheduler.get_last_lr()[0],
#                 'epoch': epoch,
#             })
        
#         if val_metrics['dice'] > best_val_dice:
#             best_val_dice = val_metrics['dice']
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_dice': best_val_dice,
#             }, os.path.join(args.logdir, 'best.ckpt'))
#             print(f"  ✓ Saved best model (dice: {best_val_dice:.4f})")
        
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'val_dice': val_metrics['dice'],
#         }, os.path.join(args.logdir, 'last.ckpt'))
    
#     print("\n" + "="*70)
#     print(f"Training Complete! Best val dice: {best_val_dice:.4f}")
#     print("="*70)


# if __name__ == '__main__':
#     main()