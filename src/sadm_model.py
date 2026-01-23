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
import matplotlib.pyplot as plt
from src.visualization.visualizer import (
    # plot_uncertainty_figure,
    # save_visualization_results,
    create_directory,
    Visualizer
)

from src.evaluation.metrics import (
    # setup_metrics,
    # calculate_metrics,
    calculate_tumor_volumes,
    get_slice_indices,
    MetricsCalculator
)
try:
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    print("Warning: MONAI not installed.")


def evaluate_predictions(
    predictions: Dict[str, torch.Tensor],
    metrics: Dict,
    session_idx: int,
    slice_idx: int,
    session_path: str
) -> Dict[str, Dict]:

    """
    Evaluate model predictions and generate visualizations.
    
    Performs:
    1. Ensemble prediction averaging
    2. Metric calculation for individual and ensemble predictions
    3. Visualization of predictions vs ground truth
    4. Result saving
    
    Args:
        predictions: Dictionary containing:
            - 'images': Predicted scans [num_samples, C, H, W]
            - 'masks': Predicted segmentations [num_samples, 4, H, W]
            - 'ground_truth': Target scans [num_samples, C, H, W]
            - 'target_masks': Target segmentations [num_samples, 4, H, W]
        metrics: Dictionary of metric functions
        session_idx: Index of the session being processed
        slice_idx: Z-index of the slice being processed
        session_path: Directory for saving results
        
    Returns:
        Dict[str, Dict]: Dictionary of metric scores for each sample and ensemble
        
    Outputs:
        - PNG visualizations saved to {session_path}/
        - Console output of evaluation metrics
    """
    scores = {}

    # Calculate average predictions
    # predictions['images'] = predictions['images'].squeeze(0)
    # predictions['masks'] = predictions['masks'].squeeze(0)
    print("predictions['images'] ", predictions['images'].shape)
    print("predictions['masks']" , predictions['masks'].shape)
    print("predictions['ground_truth']" ,predictions['ground_truth'].shape)
    print("predictions['target_masks']" , predictions['target_masks'].shape)

    avg_img = torch.mean(predictions['images'], 0)  # (3, H, W)
    avg_mask_pred = torch.sigmoid(predictions['masks'])
    avg_mask_pred = torch.mean(avg_mask_pred, 0)    # (H, W)
    
    # Calculate uncertainty maps
    img_std = torch.std(predictions['images'], 0)  # (3, H, W) - t1,t1c,flair 
    seg_seq_std = torch.std(predictions['masks'], 0)  # ( H, W) - uncertainty in sequence
    
    # Prepare visualization data
    images = {
        'prediction': predictions['images'][0].cpu().numpy(),  # Use first sample for visualization
        'ground_truth': predictions['ground_truth'][0].cpu().numpy()
    }
    masks = {
        'prediction': avg_mask_pred.cpu().numpy().astype(np.float32),  # Convert to float32
        'ground_truth': predictions['target_masks'][0].cpu().numpy().astype(np.float32),  # Convert to float32
        # 'uncertainty': img_std.cpu().numpy().astype(np.float32),  # Add uncertainty map
        # 'sequence_uncertainty': seg_seq_std.cpu().numpy().astype(np.float32)  # Add sequence uncertainty
    }

    if masks['prediction'].shape[1] == 1: 
        masks['prediction'] = masks['prediction'].unsqueeze(0)
        masks['ground_truth'] = masks['ground_truth'].unsqueeze(0)

    print("masks['prediction']" ,masks['prediction'].shape)
    print("masks['ground_truth']" , masks['ground_truth'].shape)
    
    # Create visualizer with default colors
    visualizer = Visualizer({
        0: (0, 0, 0),       # background
        1: (255, 0, 0),     # red
        2: (0, 255, 0),     # green  
        3: (0, 0, 255),     # blue
        4: (255, 255, 0)    # yellow for ensemble
    })

    modal_names = ['t1', 't1c', 'flair']
    
    try:
        # Ensure directory exists
        create_directory(session_path)
        
        # Create file prefix
        file_prefix = f'ses-{session_idx:02d}_slice-{slice_idx:03d}'
        
        # Convert masks to PIL images
        pred_mask_pil = visualizer.to_pil(masks['prediction'][  :, :])
        gt_mask_pil = visualizer.to_pil(masks['ground_truth'][  :, :])
        
        # Save masks
        pred_mask_pil.save(os.path.join(session_path, f"{file_prefix}-pred-mask.png"))
        gt_mask_pil.save(os.path.join(session_path, f"{file_prefix}-gt-mask.png"))
        
        # # Save uncertainty maps
        # visualizer.plot_uncertainty(masks['uncertainty'][0, :, :], 
        #                             os.path.join(session_path, f"{file_prefix}-uncertainty_t1.png"), 
        #                             overlay=avg_img[0, :, :].cpu().numpy())
        # visualizer.plot_uncertainty(masks['uncertainty'][1, :, :], 
        #                             os.path.join(session_path, f"{file_prefix}-uncertainty_t1c.png"),
        #                             overlay=avg_img[1, :, :].cpu().numpy())
        # visualizer.plot_uncertainty(masks['uncertainty'][2, :, :], 
        #                             os.path.join(session_path, f"{file_prefix}-uncertainty_flair.png"),
        #                             overlay=avg_img[2, :, :].cpu().numpy())
        
        # visualizer.plot_uncertainty(masks['sequence_uncertainty'][0, :, :], 
        #                             os.path.join(session_path, f"{file_prefix}-uncertainty_mask.png"),
        #                             overlay=avg_img[2, :, :].cpu().numpy())
        # uncertainty_pil = visualizer.to_pil(masks['uncertainty'][-1, :, :])
        # seq_uncertainty_pil = visualizer.to_pil(masks['sequence_uncertainty'][-1, :, :])
        # uncertainty_pil.save(os.path.join(session_path, f"{file_prefix}-uncertainty.png"))
        # seq_uncertainty_pil.save(os.path.join(session_path, f"{file_prefix}-sequence-uncertainty.png"))
        
        # Save images with overlays and contours for each modality
        for j in range(3):  # For each modality
            pred_img = visualizer.to_pil(images['prediction'][j])
            gt_img = visualizer.to_pil(images['ground_truth'][j])

            # pred_img = images['prediction'][j]
            # gt_img = images['ground_truth'][j]          
            # Save original images
            pred_img.save(os.path.join(session_path, f"{file_prefix}-pred-{modal_names[j]}.png"))
            gt_img.save(os.path.join(session_path, f"{file_prefix}-gt-{modal_names[j]}.png"))
            
            # Save overlays
            # pred_overlay = visualizer.overlay_maps(pred_img, pred_mask_pil, gt_mask_pil)
            # pred_overlay.save(os.path.join(session_path, f"{file_prefix}-pred-{modal_names[j]}_overlay.png"))
            
            # Save contours
            pred_contour = visualizer.draw_contour(pred_img, pred_mask_pil)
            pred_contour.save(os.path.join(session_path, f"{file_prefix}-pred-{modal_names[j]}_contour.png"))
            
            
    except Exception as e:
        print(f"Error saving visualization results: {e}")
        raise
    
    # print("predictions['target_masks'] ", predictions['target_masks'].shape)
    pm = predictions['masks']    # (N, S, H, W) float (prob)
    gm = predictions['target_masks']  # (N, S, H, W)

    # print("preds dtype/max/min:", pm.dtype, pm.min().item(), pm.max().item())
    # print("gt   dtype/max/min:", gm.dtype, gm.min().item(), gm.max().item())
    # print("gt unique (sample 0, session 0):", torch.unique(gm[0,0])[:20])
    # print("gt sum per session (sample 0):", [gm[0,s].sum().item() for s in range(gm.shape[1])])

    gt_masks = predictions['target_masks']
    # bring to CPU if needed
    gt_masks = gt_masks.clone()
    # If dtype not binary, threshold and convert to long
    if gt_masks.max() > 1.0:
        gt_masks_bin = (gt_masks > 0).long()
    else:
        gt_masks_bin = gt_masks.round().long()
    predictions['target_masks'] = gt_masks_bin
    
    # Calculate metrics for each sample
    for i in range(len(predictions['images'])):
        sample_metrics = metrics.calculate_metrics(
            pred_img=predictions['images'][i].unsqueeze(0),
            gt_img=predictions['ground_truth'][i].unsqueeze(0),
            pred_mask=predictions['masks'][i].unsqueeze(0),
            gt_mask=predictions['target_masks'][i].unsqueeze(0)
        )
        scores[f'sample_{i}'] = sample_metrics
    
    # Calculate metrics for ensemble prediction
    ensemble_metrics = metrics.calculate_metrics(
        pred_img=avg_img.unsqueeze(0),
        gt_img=predictions['ground_truth'][0].unsqueeze(0),
        pred_mask=avg_mask_pred.unsqueeze(0),
        gt_mask=predictions['target_masks'][0].unsqueeze(0)
    )
    scores['ensemble'] = ensemble_metrics
    
    print(f"Session {session_idx}, Slice {slice_idx} evaluation complete")
    return scores


class GaussianDiffusion:
    """Gaussian Diffusion process."""
    def __init__(self, T: int = 1000, schedule: str = 'linear',  device='cpu'):
        self.T = T
        self.device = device

        if schedule == 'linear':
            self.betas = torch.linspace(1e-4, 2e-2, T, device=self.device)
        else:
            self.betas = torch.linspace(1e-4, 2e-2, T, device=self.device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,  dim=0)
        
        # self.betas = torch.tensor(self.betas, dtype=torch.float32)
        # self.alphas = torch.tensor(self.alphas, dtype=torch.float32)
        # self.alphas_cumprod = torch.tensor(self.alphas_cumprod, dtype=torch.float32)
        self.alphabar = self.alphas_cumprod

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
        device: str = "cuda:0",
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
            num_res_blocks= num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_heads=num_heads,
            cond_dim=embed_dim,
            image_size=img_size,
        )
        
        # Diffusion process
        self.diffusion = GaussianDiffusion(T=n_T, schedule=ddpm_schedule, device=device)
        
        # Alpha bar for loss weighting
        alphabar_np = np.cumprod(1 - np.linspace(1e-4, 2e-2, n_T))
        self.register_buffer('alphabar', torch.tensor(alphabar_np, dtype=torch.float32))
        
        # Dilation filter for loss weighting
        self.register_buffer('dilation_filters', torch.ones(1, 1, 11, 11) / 10.)
        
        # Loss functions
        # if HAS_MONAI:
        self.dice = DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
            to_onehot_y=False, sigmoid=True, reduction="none"
        )
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        # else:
        #     self.dice = None
        #     self.dice_metric = None
        
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
    
    def get_loss(self, batch: dict, epoch: int, flag: bool, mode: str = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        i_tg = -torch.ones((B,), dtype=torch.int64, device=device)

        
        idx_b = torch.arange(B, device=imgs.device)          # [0, 1, ..., B-1]
        idx_s = i_tg.to(imgs.device).long()                  # target session index per batch
        # imgs: [B, S, C, H, W] -> gt_img: [B, C, H, W]
        gt_img = imgs[idx_b, idx_s, ...].to(torch.float32)
        # label: [B, S, H, W] -> gt_label: [B, H, W]
        gt_label = label[idx_b, idx_s, ...].to(torch.float32)
        gt_label = (gt_label > 0).float()

        # print("gt_img," , gt_img.shape ) # [1, 3, 240, 240]) 
        # print("gt_label," ,gt_label.shape ) # [1, 240, 240])
        gt_label_for_weight = gt_label


        # Sample diffusion timestep
        t = torch.randint(1, self.diffusion.T + 1, [B], device=device)
        w_tg = self.alphabar[t - 1].to(device)
        
        # Forward diffusion: add noise to target image
        xt, epsilon = self.diffusion.sample(gt_img, t)
        
        imgs_cond = imgs.clone()
        label_cond = label.clone()
        
        # Check for maskout case (last history == target day)
        s3_days = days[:, -2] if S > 1 else days[:, 0]
        t_days = days[:, -1]
        maskout_batch = (s3_days == t_days)
        if maskout_batch: 
            print("maskout_batch " , maskout_batch)
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
        # xt = xt.squeeze(0)
        out = self.unet(xt, t_float, cond_global, cond_tokens)
        
        # Split output: first 1 = segmentation (single channel), next 3 = noise prediction
        # Changed from 4 classes to 1 class since we have single-channel labels
        mask_pred = out[:, 0:1, :, :]  # (B, 1, H, W) - single channel segmentation
        img_pred = out[:, 1:, :, :]    # (B, C, H, W) - noise prediction
        
        # prob = torch.sigmoid(mask_pred)
        # print("[PRED] prob min/max/mean:", prob.min().item(), prob.max().item(), prob.mean().item())
        # print("[PRED] bin sum:", (prob > 0.5).float().sum().item())

        # Compute loss weights based on label
        loss_weights = gt_label_for_weight.float()  # (B, 1, H, W)
        loss_weights = loss_weights * torch.exp(-loss_weights)
        loss_weights = F.conv2d(loss_weights, self.dilation_filters.to(device), padding='same') + 1.
        
        # Weighted MSE loss on noise prediction
        loss1 = torch.mean(loss_weights * (img_pred - epsilon) ** 2)
        mse = self.loss_function(img_pred, epsilon)
        
        # Dice loss on segmentation (single channel)
        # if self.dice is not None:
            # gt_label: (B, H, W) -> (B, 1, H, W) for dice loss
        gt_label_expanded = gt_label.unsqueeze(1)
        dice_loss = self.dice(mask_pred, gt_label_expanded)
        
        for i in range(B):
            dice_loss[i, ...] = dice_loss[i, ...] * torch.sqrt(w_tg[i])
        
        dice_loss_mean = torch.mean(dice_loss)
        # else:
        #     dice_loss_mean = torch.tensor(0.0, device=device)
        
        loss = loss1 + dice_loss_mean * self.aux_loss_w
        
        # Dice metric
        # if self.dice_metric is not None:
        mask_pred_binary = (torch.sigmoid(mask_pred) > 0.1).float()
        gt_label_expanded = gt_label.unsqueeze(1)
        self.dice_metric(mask_pred_binary, gt_label_expanded)
        dice_score = self.dice_metric.aggregate()
        self.dice_metric.reset()
        # else:
        #     dice_score = torch.tensor(0.0, device=device)
                
        if mode == 'val' and flag and epoch%10 == 0:

            # choose a fixed visualization timestep for val
            fixed_t = int(getattr(self.cfg, "debug_vis_fixed_t", 200))
            fixed_t = max(1, min(fixed_t, self.diffusion.T))  # clamp

            # --- 1) build fixed timesteps tensor ---
            t_vis = torch.full((B,), fixed_t, device=gt_img.device, dtype=torch.long)
            w_tg = self.alphabar[t_vis - 1].to(device)
            
            # --- 2) create xt at this fixed timestep from the *clean* gt_img ---
            xt_vis, _ = self.diffusion.sample(gt_img, t_vis)
            # xt_vis = xt_vis.to(gt_img.device)  # (B, 3, H, W)

            # --- 3) pack model input exactly like training: replace target session with xt_vis ---
            imgs_vis = imgs.clone()  # imgs currently contains random-xt inserted; that's fine, we'll overwrite target

            for i, j in zip(range(B), i_tg):
                imgs_vis[i, j, :, :, :] = xt_vis[i]


            # --- 4) run model at fixed t to get eps prediction ---
            
            # # Forward through UNet
            # print("xt ", xt.shape)
            # xt_vis = xt_vis.squeeze(0)
            out_vis = self.unet(xt_vis, t_vis, cond_global, cond_tokens)

            eps_pred_vis = out_vis[:, 1:, :, :]   # (B, 3, H, W)
            mask_pred1 = out_vis[:, 0:1, :, :]  #  (B, C, H, W)
            # --- 5) convert eps_pred -> x0_hat using alphabar at fixed_t ---
            alphabar_t = self.diffusion.alphabar[fixed_t - 1].to(gt_img.device).view(1, 1, 1, 1)
            x0_hat_vis = (xt_vis - torch.sqrt(1.0 - alphabar_t) * eps_pred_vis) / (torch.sqrt(alphabar_t) + 1e-8)

            # --- (optional but recommended) clamp for stable visualization ---
            # If  data is normalized to [-1, 1], keep this. If not, remove it.
            # x0_hat_vis = torch.clamp(x0_hat_vis, -1.0, 1.0)

            mask_pred1 = mask_pred1.squeeze(1)
            predictions = {
                'images': x0_hat_vis,     # <-- denoised estimate (x0_hat), NOT eps
                'masks': mask_pred1,      # probs
                'ground_truth': gt_img,   # x0
                'target_masks': gt_label
            }
            device = "cuda:0"
            metrics = MetricsCalculator(device)

            # Calculate metrics and save visualizations
            slice_scores = evaluate_predictions(
                predictions=predictions,
                metrics=metrics,
                session_idx=3,
                slice_idx=2,
                session_path= f"results/SADM_Sampling_Full_T1k_single_inverse/{epoch}"
            )

            
            self.full_inverse_eval_once(
                batch=batch,
                epoch=epoch,
                out_dir="results/SADM_Sampling_Full_T1k_full_inverse",
                num_samples=1
            )

        return loss, mse, dice_score
    
    @torch.no_grad()
    def full_inverse_eval_once(self, batch, epoch, out_dir, num_samples=1):
        self.unet.eval()

                # Extract batch data
        imgs0 = batch['image']
        label0 = batch['label']  # (B, S, H, W) - single channel per session
        days = batch['days']
        treatments = batch.get('treatments', batch.get('treatment'))
        geno = batch.get('geno', None)
        
        B, S, C, H, W = imgs0.shape
        device = imgs0.device
        
        imgs = imgs0.clone()     # working tensor you can corrupt safely
        label = label0.clone()

        i_tg = -torch.ones((B,), dtype=torch.int64, device=device)
        idx_b = torch.arange(B, device=imgs.device) # [0, 1, ..., B-1]
        idx_s = i_tg.to(imgs.device).long()
        # target session index per batch # imgs: [B, S, C, H, W] -> gt_img: [B, C, H, W] 
        gt_img_0 = imgs[idx_b, idx_s, ...].to(torch.float32) 
        # label: [B, S, H, W] -> gt_label: [B, H, W] 
        gt_label_0 = label[idx_b, idx_s, ...].to(torch.float32) 
        gt_label_0 = (gt_label_0 > 0).float()

        # print("gt_img_0 ", gt_img_0.shape)
        # print("gt_label_0 ", gt_label_0.shape)

        # start from PURE NOISE
        T = int(self.diffusion.T)
        t = torch.randint(1, T + 1, [B], device=device)
        w_tg = self.alphabar[t - 1].to(device)
        
        # Forward diffusion: add noise to target image
        x_t, epsilon = self.diffusion.sample(gt_img_0, t)

        # Get conditioning
        cond_global, cond_tokens = self.get_conditioning(
            images=imgs, labels=label, days=days,
            treatments=treatments, geno=geno,
        )
        
        for t in reversed(range(1, T + 1)):
            t_idx = torch.full((B,), t, device=device)
            t_float = t_idx.float()
            out = self.unet(x_t, t_float, cond_global, cond_tokens)
            # Segmentation is first channel, noise prediction is remaining channels
            pred_noise = out[:, 1:, :, :]  # (B, C, H, W)
            
            alpha_t     = self.diffusion.alphas[t_idx - 1].to(device).view(B,1,1,1)
            alpha_bar_t = self.diffusion.alphas_cumprod[t_idx - 1].to(device).view(B,1,1,1)
            beta_t      = self.diffusion.betas[t_idx - 1].to(device).view(B,1,1,1)
            
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
        
        recon_img = x_t
        # Get segmentation from final output
        t_final = torch.ones(B, device=device)
        out_final = self.unet(x_t, t_final, cond_global, cond_tokens)
        seg_pred = torch.sigmoid(out_final[:, 0:1, :, :])  # (B, 1, H, W)
        
        
        predictions = {
            "images": recon_img,
            "masks": seg_pred.squeeze(0),
            "ground_truth": gt_img_0,
            "target_masks": gt_label_0
        }


        metrics = MetricsCalculator(str(device))
        evaluate_predictions(
            predictions=predictions,
            metrics=metrics,
            session_idx=3,
            slice_idx=2,
            session_path=os.path.join(out_dir, f"epoch_{epoch}")
        )


    # @torch.no_grad()
    # def sample(self, batch: dict, num_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # """Generate samples given conditioning."""
        # imgs = batch['image']
        # label = batch['label']
        # days = batch['days']
        # treatments = batch.get('treatments', batch.get('treatment'))
        # geno = batch.get('geno', None)
        
        # B, S, C, H, W = imgs.shape
        # device = imgs.device
        
        # # Get conditioning
        # cond_global, cond_tokens = self.get_conditioning(
        #     images=imgs, labels=label, days=days,
        #     treatments=treatments, geno=geno,
        # )
        
        # # Start from noise
        # x_t = torch.randn(B, C, H, W, device=device)
        # T = self.n_T
        
        # for t in reversed(range(1, T + 1)):
        #     t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
        #     out = self.unet(x_t, t_tensor, cond_global, cond_tokens)
        #     # Segmentation is first channel, noise prediction is remaining channels
        #     pred_noise = out[:, self.num_seg_classes:, :, :]  # (B, C, H, W)
            
        #     alpha_t = self.diffusion.alphas[t - 1].to(device)
        #     alpha_bar_t = self.diffusion.alphas_cumprod[t - 1].to(device)
        #     beta_t = self.diffusion.betas[t - 1].to(device)
            
        #     mean = (1 / torch.sqrt(alpha_t)) * (
        #         x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        #     )
            
        #     if t > 1:
        #         noise = torch.randn_like(x_t)
        #         alpha_bar_prev = self.diffusion.alphas_cumprod[t - 2].to(device)
        #         variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        #         x_t = mean + torch.sqrt(variance) * noise
        #     else:
        #         x_t = mean
        
        # # Get segmentation from final output
        # t_final = torch.ones(B, device=device)
        # out_final = self.unet(x_t, t_final, cond_global, cond_tokens)
        # pred = out_final[:, :self.in_channels, :, :]
        # seg_pred = torch.sigmoid(out_final[:, :self.num_seg_classes, :, :])  # (B, 1, H, W)
        
        # predictions = {
        #     "images": pred,
        #     "masks": seg_pred,
        #     "ground_truth": x_0,
        #     "target_masks": masks[:, :, :, :] 
        # }

        # metrics = MetricsCalculator(str(self.device))
        # evaluate_predictions(
        #     predictions=predictions,
        #     metrics=metrics,
        #     session_idx=3,
        #     slice_idx=2,
        #     session_path=os.path.join(out_dir, f"epoch_{epoch}")
        # )

        # # # print ranges to debug normalization mismatches
        # # print("[FULL INV] GT  min/max/mean/std:",
        # #     x_0.min().item(), x_0.max().item(), x_0.mean().item(), x_0.std().item())
        # # print("[FULL INV] GEN min/max/mean/std:",
        # #     pred_img.min().item(), pred_img.max().item(), pred_img.mean().item(), pred_img.std().item())


        # return x_t, seg_pred