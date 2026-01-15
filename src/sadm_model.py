"""
SADM.py - Sequence-Aware Diffusion Model for 2D Longitudinal Medical Images

Fixed version that handles:
- imgs_slice: (B, S, C, H, W) = (1, 4, 3, 240, 240)
- labels_slice: (B, S, H, W) = (1, 4, 240, 240)  # Single channel labels
- days: (B, S) = (1, 4)
- treatments: (B, S) = (1, 4)
- geno: (B, G) = (1, 13)
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sadm_net.DDPM.ddpm import ContextUNet2D
from src.sadm_net.ViViT.vivit import SequenceAwareViT2D

try:
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    print("Warning: MONAI not installed.")


class GaussianDiffusion:
    """Gaussian Diffusion process."""
    def __init__(self, T: int = 1000, schedule: str = 'linear'):
        self.T = T
        if schedule == 'linear':
            self.betas = np.linspace(1e-4, 2e-2, T)
        else:
            self.betas = np.linspace(1e-4, 2e-2, T)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        self.betas = torch.tensor(self.betas, dtype=torch.float32)
        self.alphas = torch.tensor(self.alphas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(self.alphas_cumprod, dtype=torch.float32)
    
    def sample(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(x_t | x_0)"""
        device = x0.device
        t_idx = (t - 1).long().cpu()
        
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t_idx]).to(device).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_cumprod[t_idx]).to(device).view(-1, 1, 1, 1)
        
        epsilon = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
        
        return xt, epsilon


class SADM(nn.Module):
    """
    Sequence-Aware Diffusion Model.
    
    Expected input format:
    - batch['image']: (B, S, C, H, W) - S sessions, C channels per session
    - batch['label']: (B, S, H, W) or (B, S, num_classes, H, W)
    - batch['days']: (B, S)
    - batch['treatments'] or batch['treatment']: (B, S)
    - batch['geno']: (B, G)
    """
    
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 20,
        in_channels: int = 3,  # Channels PER SESSION (T1, T1c, FLAIR)
        num_seg_classes: int = 1,  # Number of segmentation classes (1 mask per session)
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
        ddpm_schedule: str = 'linear',
        use_geno: bool = True,
        geno_dim: int = 13,
        dropout: float = 0.1,
        aux_loss_w: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.in_channels = in_channels  # Per-session channels (3)
        self.num_seg_classes = num_seg_classes  # 1 mask per session
        self.out_channels = num_seg_classes + in_channels  # 1 + 3 = 4 (seg first, then noise)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.n_T = n_T
        self.aux_loss_w = aux_loss_w
        self.img_size = img_size
        self.patch_size = patch_size
        
        print(f"\n{'='*60}")
        print(f"SADM Model Configuration:")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}")
        print(f"  Channels per session: {in_channels}")
        print(f"  Segmentation classes: {num_seg_classes}")
        print(f"  Output channels: {self.out_channels} (seg: {num_seg_classes}, noise: {in_channels})")
        print(f"  Max sessions: {max_seq_len}")
        print(f"  Genomic dim: {geno_dim}")
        print(f"{'='*60}\n")
        
        # Config-like attributes for compatibility
        self.cfg = type('Config', (), {
            'aux_loss_w': aux_loss_w,
            'max_T': n_T,
            'ddpm_schedule': ddpm_schedule,
        })()
        
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
            out_channels=self.out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_heads=num_heads,
            cond_dim=embed_dim,
            image_size=img_size,
        )
        
        # Diffusion process
        self.diffusion = GaussianDiffusion(T=n_T, schedule=ddpm_schedule)
        
        # Alpha bar for loss weighting
        alphabar_np = np.cumprod(1 - np.linspace(1e-4, 2e-2, n_T))
        self.register_buffer('alphabar', torch.tensor(alphabar_np, dtype=torch.float32))
        
        # Dilation filter for loss weighting
        self.register_buffer('dilation_filters', torch.ones(1, 1, 11, 11) / 10.)
        
        # Loss functions
        if HAS_MONAI:
            self.dice = DiceLoss(
                smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
                to_onehot_y=False, sigmoid=True, reduction="none"
            )
            self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        else:
            self.dice = None
            self.dice_metric = None
        
        self.loss_function = F.mse_loss
    
    def get_conditioning(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        days: torch.Tensor,
        treatments: torch.Tensor,
        geno: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get conditioning from SAT using history sessions.
        
        Args:
            images: (B, S, C, H, W) - already with target replaced by noisy version
            labels: (B, S, H, W) or (B, S, num_classes, H, W)
            days: (B, S)
            treatments: (B, S)
            geno: (B, G)
        """
        B, S, C, H, W = images.shape
        device = images.device
        
        # Use all but last session for history conditioning
        if S > 1:
            hist_images = images[:, :-1]  # (B, S-1, C, H, W)
            hist_labels = labels[:, :-1]
            hist_days = days[:, :-1]
            hist_treatments = treatments[:, :-1]
        else:
            # Only one session - use it
            hist_images = images
            hist_labels = labels
            hist_days = days
            hist_treatments = treatments
        
        # Handle label dimensions - need (B, S, H, W) for SAT
        if hist_labels.dim() == 5:  # (B, S, num_classes, H, W)
            hist_labels_single = hist_labels.max(dim=2)[0]  # (B, S, H, W)
        elif hist_labels.dim() == 4:  # (B, S, H, W)
            hist_labels_single = hist_labels
        else:
            raise ValueError(f"Unexpected label shape: {hist_labels.shape}")
        
        # Get conditioning from SAT
        cond_spatial, cond_tokens = self.sat(
            images=hist_images,
            labels=hist_labels_single,
            days=hist_days,
            treatments=hist_treatments,
            geno=geno,
        )
        
        # Global conditioning via spatial pooling
        cond_global = F.adaptive_avg_pool2d(cond_spatial, 1).flatten(1)
        
        return cond_global, cond_tokens
    
    def get_loss(self, batch: dict, mode: str = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss.
        
        Expected batch format:
        - 'image': (B, S, C, H, W) - e.g., (1, 4, 3, 240, 240) - 4 sessions, 3 modalities
        - 'label': (B, S, H, W) - e.g., (1, 4, 240, 240) - 1 mask per session
        - 'days': (B, S)
        - 'treatments' or 'treatment': (B, S)
        - 'geno': (B, G)
        """
        # Extract batch data
        imgs = batch['image']
        label = batch['label']  # (B, S, H, W) - single channel per session
        days = batch['days']
        treatments = batch.get('treatments', batch.get('treatment'))
        geno = batch.get('geno', None)
        
        B, S, C, H, W = imgs.shape
        device = imgs.device
        
        # Validate input
        assert C == self.in_channels, \
            f"Input has {C} channels but model expects {self.in_channels}"
        
        # Target is always last session
        i_tg = -1
        
        # Extract target image and label
        gt_img = imgs[:, i_tg, ...].to(torch.float32)  # (B, C, H, W)
        gt_label = label[:, i_tg, ...].to(torch.float32)  # (B, H, W) - single channel
        
        # For loss weighting, use the single-channel label
        # (B, H, W) -> (B, 1, H, W)
        gt_label_for_weight = gt_label.unsqueeze(1)
        
        # Sample diffusion timestep
        t = torch.randint(1, self.diffusion.T + 1, [B], device=device)
        w_tg = self.alphabar[t - 1].to(device)
        
        # Forward diffusion: add noise to target image
        xt, epsilon = self.diffusion.sample(gt_img, t)
        
        # Prepare conditioning images (replace target with noisy version)
        imgs_cond = imgs.clone()
        label_cond = label.clone()
        
        # Check for maskout case (last history == target day)
        s3_days = days[:, -2] if S > 1 else days[:, 0]
        t_days = days[:, -1]
        maskout_batch = (s3_days == t_days)
        
        for i in range(B):
            if maskout_batch[i]:
                imgs_cond[i, :, :, :, :] = 0.
                label_cond[i, ...] = 0
            imgs_cond[i, -1, :, :, :] = xt[i, :, :, :]
        
        # Get SAT conditioning
        cond_global, cond_tokens = self.get_conditioning(
            images=imgs_cond,
            labels=label_cond,
            days=days,
            treatments=treatments,
            geno=geno,
        )
        
        # Forward through UNet
        t_float = t.to(torch.float32)
        out = self.unet(xt, t_float, cond_global, cond_tokens)
        
        # Split output: first 1 = segmentation (single channel), next 3 = noise prediction
        # Changed from 4 classes to 1 class since we have single-channel labels
        mask_pred = out[:, 0:1, :, :]  # (B, 1, H, W) - single channel segmentation
        img_pred = out[:, 1:, :, :]    # (B, C, H, W) - noise prediction
        
        # Compute loss weights based on label
        loss_weights = gt_label_for_weight.float()  # (B, 1, H, W)
        loss_weights = loss_weights * torch.exp(-loss_weights)
        loss_weights = F.conv2d(
            loss_weights, self.dilation_filters.to(device), padding='same'
        ) + 1.
        
        # Weighted MSE loss on noise prediction
        loss1 = torch.mean(loss_weights * (img_pred - epsilon) ** 2)
        mse = self.loss_function(img_pred, epsilon)
        
        # Dice loss on segmentation (single channel)
        if self.dice is not None:
            # gt_label: (B, H, W) -> (B, 1, H, W) for dice loss
            gt_label_expanded = gt_label.unsqueeze(1)
            dice_loss = self.dice(mask_pred, gt_label_expanded)
            
            for i in range(B):
                dice_loss[i, ...] = dice_loss[i, ...] * torch.sqrt(w_tg[i])
            
            dice_loss_mean = torch.mean(dice_loss)
        else:
            dice_loss_mean = torch.tensor(0.0, device=device)
        
        loss = loss1 + dice_loss_mean * self.aux_loss_w
        
        # Dice metric
        if self.dice_metric is not None:
            mask_pred_binary = (torch.sigmoid(mask_pred) > 0.5).float()
            gt_label_expanded = gt_label.unsqueeze(1)
            self.dice_metric(mask_pred_binary, gt_label_expanded)
            dice_score = self.dice_metric.aggregate()
            self.dice_metric.reset()
        else:
            dice_score = torch.tensor(0.0, device=device)
        
        return loss, mse, dice_score
    
    @torch.no_grad()
    def sample(self, batch: dict, num_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples given conditioning."""
        imgs = batch['image']
        label = batch['label']
        days = batch['days']
        treatments = batch.get('treatments', batch.get('treatment'))
        geno = batch.get('geno', None)
        
        B, S, C, H, W = imgs.shape
        device = imgs.device
        
        # Get conditioning
        cond_global, cond_tokens = self.get_conditioning(
            images=imgs, labels=label, days=days,
            treatments=treatments, geno=geno,
        )
        
        # Start from noise
        x_t = torch.randn(B, C, H, W, device=device)
        T = num_steps or self.n_T
        
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            out = self.unet(x_t, t_tensor, cond_global, cond_tokens)
            # Segmentation is first channel, noise prediction is remaining channels
            pred_noise = out[:, self.num_seg_classes:, :, :]  # (B, C, H, W)
            
            alpha_t = self.diffusion.alphas[t - 1].to(device)
            alpha_bar_t = self.diffusion.alphas_cumprod[t - 1].to(device)
            beta_t = self.diffusion.betas[t - 1].to(device)
            
            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            )
            
            if t > 1:
                noise = torch.randn_like(x_t)
                alpha_bar_prev = self.diffusion.alphas_cumprod[t - 2].to(device)
                variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                x_t = mean + torch.sqrt(variance) * noise
            else:
                x_t = mean
        
        # Get segmentation from final output
        t_final = torch.ones(B, device=device)
        out_final = self.unet(x_t, t_final, cond_global, cond_tokens)
        seg_pred = torch.sigmoid(out_final[:, :self.num_seg_classes, :, :])  # (B, 1, H, W)
        
        return x_t, seg_pred