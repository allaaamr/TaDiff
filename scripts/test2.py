"""
SADM Model Testing and Evaluation Script

Adapted from TaDiff test.py for the Sequence-Aware Diffusion Model.

Key differences from TaDiff:
- Uses SAT (Sequence-Aware Transformer) for conditioning instead of image concatenation
- UNet receives only noisy target + SAT conditioning
- Diffusion inverse is done through the SADM model directly
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

# Import your existing modules
from config.cfg_tadiff_net import config as train_config
from config.test_config import TestConfig
from src.visualization.visualizer import create_directory, Visualizer
from src.evaluation.metrics import calculate_tumor_volumes, get_slice_indices, MetricsCalculator
from torch.utils.data import Dataset, DataLoader

# Import SADM
from src.sadm_model import SADM

npz_keys = ['image', 'label', 'days', 'treatment', 'geno']


class TestLoader(Dataset):
    """Test dataset loader - same as your original."""
    
    def __init__(self, file_dicts: List[Dict], transform=None):
        self.file_dicts = file_dicts[:]
        self.transform = transform

    def __len__(self):
        return max(1, len(self.file_dicts))

    def __getitem__(self, patient_idx):
        file_dict = self.file_dicts[patient_idx]

        # Load arrays
        data = {k: np.load(file_dict[k]) for k in npz_keys}
        img_full = data['image']
        lbl_full = data['label']
        days_full = data['days']
        treat_full = data['treatment']
        geno_full = data['geno']

        S_all = int(lbl_full.shape[0])
        C_times_S, D, H, W = img_full.shape
        C_full = C_times_S // S_all

        S_treat = treat_full.shape[0]
        if S_all != S_treat:
            treat_full = treat_full[:S_all - 1]
            days_full = days_full[:S_all - 1]

        # Reshape
        S, D, H, W = lbl_full.shape
        lbl_full = lbl_full.reshape(S, H, W, D)

        img_full = img_full.reshape(C_full, S_all, H, W, D)
        img_full = np.moveaxis(img_full, 0, 1)  # -> (S, C, H, W, D)

        return {
            'image': img_full,
            'label': lbl_full,
            'days': days_full,
            'treatment': treat_full,
            'geno': geno_full
        }


class SADMInference:
    """
    SADM inference wrapper that provides similar interface to GaussianDiffusion.TaDiff_inverse()
    """
    
    def __init__(self, model: SADM, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        days: torch.Tensor,
        treatments: torch.Tensor,
        geno: torch.Tensor,
        num_steps: int = 100,
        num_samples: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run SADM prediction.
        
        Args:
            images: (B, S, C, H, W) - history images (S sessions, C channels)
            labels: (B, S, H, W) - history labels
            days: (B, S) - days for each session
            treatments: (B, S) - treatment codes
            geno: (B, G) - genomic features
            num_steps: Number of diffusion steps for sampling
            num_samples: Number of samples to generate
            
        Returns:
            pred_images: (num_samples, C, H, W) predicted images
            pred_masks: (num_samples, 1, H, W) predicted segmentation
        """
        B, S, C, H, W = images.shape
        
        # Repeat for multiple samples
        if num_samples > 1:
            images = images.repeat(num_samples, 1, 1, 1, 1)
            labels = labels.repeat(num_samples, 1, 1, 1)
            days = days.repeat(num_samples, 1)
            treatments = treatments.repeat(num_samples, 1)
            geno = geno.repeat(num_samples, 1)
        
        # Create batch for SADM
        batch = {
            'image': images,
            'label': labels,
            'days': days,
            'treatments': treatments,
            'geno': geno,
        }
        
        # Run sampling
        pred_images, pred_masks = self.model.sample(batch, num_steps=num_steps)
        
        return pred_images, pred_masks
    
    @torch.no_grad()
    def predict_with_history(
        self,
        history_images: torch.Tensor,
        history_labels: torch.Tensor,
        history_days: torch.Tensor,
        history_treatments: torch.Tensor,
        target_day: torch.Tensor,
        target_treatment: torch.Tensor,
        geno: torch.Tensor,
        num_steps: int = 100,
        num_samples: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future image given history.
        
        This is more similar to how TaDiff works - you provide history
        and predict the future.
        
        Args:
            history_images: (B, S_hist, C, H, W) - history images
            history_labels: (B, S_hist, H, W) - history labels
            history_days: (B, S_hist) - history days
            history_treatments: (B, S_hist) - history treatments
            target_day: (B,) - target day to predict
            target_treatment: (B,) - target treatment
            geno: (B, G) - genomic features
            num_steps: Diffusion steps
            num_samples: Number of samples
            
        Returns:
            pred_images: (num_samples, C, H, W)
            pred_masks: (num_samples, 1, H, W)
        """
        B, S_hist, C, H, W = history_images.shape
        device = history_images.device
        
        # Append placeholder for target session
        # Create dummy target image (will be replaced by noise in sampling)
        dummy_target = torch.zeros(B, 1, C, H, W, device=device)
        images = torch.cat([history_images, dummy_target], dim=1)  # (B, S_hist+1, C, H, W)
        
        # Append dummy target label
        dummy_label = torch.zeros(B, 1, H, W, device=device)
        labels = torch.cat([history_labels, dummy_label], dim=1)  # (B, S_hist+1, H, W)
        
        # Append target day/treatment
        days = torch.cat([history_days, target_day.unsqueeze(1)], dim=1)  # (B, S_hist+1)
        treatments = torch.cat([history_treatments, target_treatment.unsqueeze(1)], dim=1)
        
        # Repeat for samples
        if num_samples > 1:
            images = images.repeat(num_samples, 1, 1, 1, 1)
            labels = labels.repeat(num_samples, 1, 1, 1)
            days = days.repeat(num_samples, 1)
            treatments = treatments.repeat(num_samples, 1)
            geno = geno.repeat(num_samples, 1)
        
        # Get conditioning from history only
        cond_global, cond_tokens = self.model.get_conditioning(
            images=images[:, :-1],  # Exclude target
            labels=labels[:, :-1],
            days=days[:, :-1],
            treatments=treatments[:, :-1],
            geno=geno,
        )
        
        # Start from noise
        x_t = torch.randn(num_samples * B, C, H, W, device=device)
        
        # Reverse diffusion
        T = num_steps
        for t in tqdm(reversed(range(1, T + 1)), desc="Sampling", leave=False):
            t_tensor = torch.full((num_samples * B,), t, device=device, dtype=torch.float32)
            
            out = self.model.unet(x_t, t_tensor, cond_global, cond_tokens)
            pred_noise = out[:, self.model.num_seg_classes:, :, :]
            
            alpha_t = self.model.diffusion.alphas[t - 1].to(device)
            alpha_bar_t = self.model.diffusion.alphas_cumprod[t - 1].to(device)
            beta_t = self.model.diffusion.betas[t - 1].to(device)
            
            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            )
            
            if t > 1:
                noise = torch.randn_like(x_t)
                alpha_bar_prev = self.model.diffusion.alphas_cumprod[t - 2].to(device)
                variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                x_t = mean + torch.sqrt(variance) * noise
            else:
                x_t = mean
        
        # Get final segmentation
        t_final = torch.ones(num_samples * B, device=device)
        out_final = self.model.unet(x_t, t_final, cond_global, cond_tokens)
        pred_masks = torch.sigmoid(out_final[:, :self.model.num_seg_classes, :, :])
        
        return x_t, pred_masks


def process_slice_sadm(
    slice_idx: int,
    session_idx: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    days: torch.Tensor,
    treatments: torch.Tensor,
    geno: torch.Tensor,
    model: SADM,
    device: torch.device,
    metrics: MetricsCalculator,
    session_path: str,
    diffusion_steps: int,
    num_samples: int,
    target_idx: int
) -> Dict[str, Dict]:
    """
    Process a single 2D slice through SADM prediction pipeline.
    
    Adapted from process_slice() for SADM architecture.
    """
    # Move to device
    images = images.to(device)
    labels = labels.to(device)
    days = days.to(device)
    treatments = treatments.to(device)
    geno = geno.to(device)
    
    # Get dimensions
    b, s, c, h, w, z = images.shape
    
    # Remove T2 if present (keep T1, T1c, FLAIR)
    if c == 4:
        images = images[:, :, :3, :, :, :]
        c = 3
    
    # Get session indices for history
    session_indices = np.array([
        max(0, session_idx - 3),
        max(0, session_idx - 2),
        max(0, session_idx - 1),
        session_idx
    ])
    session_indices = list(session_indices)
    print(f"Session indices: {session_indices}")
    
    # Extract 2D slice
    # images: (B, S, C, H, W, D) -> slice -> (B, S, C, H, W)
    imgs_slice = images[0, session_indices, :, :, :, slice_idx]  # (4, C, H, W)
    labels_slice = labels[0, session_indices, :, :, slice_idx]   # (4, H, W)
    days_slice = days[0, session_indices]                         # (4,)
    treatments_slice = treatments[0, session_indices]             # (4,)
    
    # Add batch dimension and repeat for num_samples
    imgs_slice = imgs_slice.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)  # (num_samples, 4, C, H, W)
    labels_slice = labels_slice.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # (num_samples, 4, H, W)
    days_slice = days_slice.unsqueeze(0).repeat(num_samples, 1)            # (num_samples, 4)
    treatments_slice = treatments_slice.unsqueeze(0).repeat(num_samples, 1)  # (num_samples, 4)
    geno_repeated = geno.repeat(num_samples, 1)  # (num_samples, G)
    
    # Store ground truth before modification
    gt_img = imgs_slice[:, target_idx, :, :, :].clone()  # (num_samples, C, H, W)
    gt_mask = labels_slice[:, target_idx, :, :].clone()  # (num_samples, H, W)
    
    # Create batch for SADM
    batch = {
        'image': imgs_slice,
        'label': labels_slice,
        'days': days_slice,
        'treatments': treatments_slice,
        'geno': geno_repeated,
    }
    
    # Run SADM sampling
    print(f"Running SADM sampling with {diffusion_steps} steps...")
    pred_img, pred_mask = model.sample(batch, num_steps=diffusion_steps)
    
    # pred_img: (num_samples, C, H, W)
    # pred_mask: (num_samples, 1, H, W)
    
    print(f"Prediction shapes: img={pred_img.shape}, mask={pred_mask.shape}")
    
    # Prepare predictions dict (compatible with evaluate_predictions)
    predictions = {
        'images': pred_img,           # (num_samples, C, H, W)
        'masks': pred_mask,           # (num_samples, 1, H, W)
        'ground_truth': gt_img,       # (num_samples, C, H, W)
        'target_masks': gt_mask.unsqueeze(1)  # (num_samples, 1, H, W)
    }
    
    # Evaluate and visualize
    slice_scores = evaluate_predictions_sadm(
        predictions=predictions,
        metrics=metrics,
        session_idx=session_idx,
        slice_idx=slice_idx,
        session_path=session_path
    )
    
    return slice_scores


def evaluate_predictions_sadm(
    predictions: Dict[str, torch.Tensor],
    metrics: MetricsCalculator,
    session_idx: int,
    slice_idx: int,
    session_path: str
) -> Dict[str, Dict]:
    """
    Evaluate SADM predictions.
    
    Adapted for single-channel segmentation output.
    """
    scores = {}
    
    # Calculate average predictions
    avg_img = torch.mean(predictions['images'], dim=0)  # (C, H, W)
    avg_mask = torch.mean(predictions['masks'], dim=0)  # (1, H, W)
    
    # Calculate uncertainty
    img_std = torch.std(predictions['images'], dim=0)   # (C, H, W)
    mask_std = torch.std(predictions['masks'], dim=0)   # (1, H, W)
    
    # Create visualizer
    visualizer = Visualizer({
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    })
    
    modal_names = ['t1', 't1c', 'flair']
    
    try:
        create_directory(session_path)
        file_prefix = f'ses-{session_idx:02d}_slice-{slice_idx:03d}'
        
        # Save masks
        pred_mask_np = (avg_mask[0] > 0.5).float().cpu().numpy()
        gt_mask_np = predictions['target_masks'][0, 0].cpu().numpy()
        
        pred_mask_pil = visualizer.to_pil(pred_mask_np)
        gt_mask_pil = visualizer.to_pil(gt_mask_np)
        
        pred_mask_pil.save(os.path.join(session_path, f"{file_prefix}-pred-mask.png"))
        gt_mask_pil.save(os.path.join(session_path, f"{file_prefix}-gt-mask.png"))
        
        # Save uncertainty
        visualizer.plot_uncertainty(
            mask_std[0].cpu().numpy(),
            os.path.join(session_path, f"{file_prefix}-mask-uncertainty.png"),
            overlay=avg_img[0].cpu().numpy()
        )
        
        # Save images for each modality
        for j, modal_name in enumerate(modal_names):
            if j < predictions['images'].shape[1]:
                pred_img_pil = visualizer.to_pil(predictions['images'][0, j].cpu().numpy())
                gt_img_pil = visualizer.to_pil(predictions['ground_truth'][0, j].cpu().numpy())
                
                pred_img_pil.save(os.path.join(session_path, f"{file_prefix}-pred-{modal_name}.png"))
                gt_img_pil.save(os.path.join(session_path, f"{file_prefix}-gt-{modal_name}.png"))
                
                # Save with contour
                pred_contour = visualizer.draw_contour(pred_img_pil, pred_mask_pil)
                pred_contour.save(os.path.join(session_path, f"{file_prefix}-pred-{modal_name}_contour.png"))
                
                # Save uncertainty per modality
                visualizer.plot_uncertainty(
                    img_std[j].cpu().numpy(),
                    os.path.join(session_path, f"{file_prefix}-uncertainty-{modal_name}.png"),
                    overlay=avg_img[j].cpu().numpy()
                )
                
    except Exception as e:
        print(f"Error saving visualizations: {e}")
    
    # Prepare masks for metrics (threshold predictions)
    pred_masks_binary = (predictions['masks'] > 0.5).int()
    gt_masks = predictions['target_masks'].int()
    
    # Calculate metrics for each sample
    for i in range(len(predictions['images'])):
        sample_metrics = metrics.calculate_metrics(
            pred_img=predictions['images'][i].unsqueeze(0),
            gt_img=predictions['ground_truth'][i].unsqueeze(0),
            pred_mask=pred_masks_binary[i].unsqueeze(0),
            gt_mask=gt_masks[i].unsqueeze(0)
        )
        scores[f'sample_{i}'] = sample_metrics
    
    # Ensemble metrics
    ensemble_mask = (avg_mask > 0.5).float().unsqueeze(0)
    ensemble_metrics = metrics.calculate_metrics(
        pred_img=avg_img.unsqueeze(0),
        gt_img=predictions['ground_truth'][0].unsqueeze(0),
        pred_mask=ensemble_mask,
        gt_mask=gt_masks[0].unsqueeze(0)
    )
    scores['ensemble'] = ensemble_metrics
    
    print(f"Session {session_idx}, Slice {slice_idx} evaluation complete")
    return scores


def get_test_files(config: TestConfig):
    """Get list of test files for each patient."""
    test_files = []
    for patient_id in config.patient_ids:
        file_dict = {
            key: os.path.join(config.data_root, f'{patient_id}_{key}.npy')
            for key in npz_keys
        }
        test_files.append(file_dict)
    return test_files


def setup_sadm_model(config: TestConfig, device: str) -> SADM:
    """Initialize and load SADM model."""
    
    # Create SADM model
    model = SADM(
        img_size=config.image_size if hasattr(config, 'image_size') else 240,
        patch_size=20,
        in_channels=3,  # T1, T1c, FLAIR
        num_seg_classes=1,  # Single mask per session
        embed_dim=256,
        model_channels=config.model_channels if hasattr(config, 'model_channels') else 64,
        geno_dim=13,
        n_T=1000,
        device=device,
    ).to(device)
    
    # Load checkpoint if available
    if hasattr(config, 'model_checkpoint') and os.path.exists(config.model_checkpoint):
        print(f"Loading SADM checkpoint from {config.model_checkpoint}")
        checkpoint = torch.load(config.model_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")
    else:
        print("No checkpoint found, using randomly initialized model")
    
    model.eval()
    return model


def main():
    """Main testing function for SADM."""
    
    # Load configuration
    config = TestConfig()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize SADM model
    model = setup_sadm_model(config, str(device))
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(device, config.dice_thresholds)
    
    # Load test data
    test_files = get_test_files(config)
    test_dataset = TestLoader(file_dicts=test_files)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create output directory
    os.makedirs(config.save_path, exist_ok=True)
    
    # Load or create CSV
    csv_path = os.path.join(config.save_path, 'sadm_test_scores.csv')
    if os.path.exists(csv_path):
        all_scores = pd.read_csv(csv_path, index_col=0).to_dict('index')
    else:
        all_scores = {}
    
    # Process each patient
    for i, batch in enumerate(dataloader):
        patient_id = config.patient_ids[i]
        print(f'\n{"="*60}')
        print(f'Processing patient {patient_id}')
        print(f'{"="*60}')
        
        # Get number of sessions
        num_sessions = batch['label'].shape[1]
        print(f"Patient has {num_sessions} sessions")
        
        # Process each session
        for session_idx in range(num_sessions):
            print(f"\nSession {session_idx}")
            
            # Find best slice (most tumor)
            z_mask_size = calculate_tumor_volumes(batch['label'][0])
            slice_idx = int(get_slice_indices(z_mask_size, top_k=1))
            print(f"Selected slice: {slice_idx}")
            
            # Process slice
            slice_scores = process_slice_sadm(
                slice_idx=slice_idx,
                session_idx=session_idx,
                images=batch['image'],
                labels=batch['label'],
                days=batch['days'],
                treatments=batch['treatment'],
                geno=batch['geno'],
                model=model,
                device=device,
                metrics=metrics_calculator,
                session_path=os.path.join(config.save_path, f'p-{patient_id}', f'ses-{session_idx:02d}'),
                diffusion_steps=config.diffusion_steps,
                num_samples=config.num_samples,
                target_idx=-1  # Always predict last session
            )
            
            # Flatten scores
            flattened_scores = {}
            for sample_key, sample_metrics in slice_scores.items():
                for metric_name, metric_value in sample_metrics.items():
                    flattened_scores[f"{sample_key}_{metric_name}"] = metric_value
            
            flattened_scores['patient_id'] = patient_id
            flattened_scores['session_idx'] = session_idx
            flattened_scores['slice_idx'] = slice_idx
            
            # Save
            score_key = f'{patient_id}_ses{session_idx}_slice{slice_idx:03d}'
            all_scores[score_key] = flattened_scores
            
            pd.DataFrame.from_dict(all_scores, orient='index').to_csv(csv_path)
            print(f"Saved scores for {score_key}")
    
    print(f"\n{'='*60}")
    print(f"Testing complete! Results saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()