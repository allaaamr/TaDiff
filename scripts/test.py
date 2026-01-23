"""
TaDiff Model Testing and Evaluation Script

This script provides comprehensive testing and evaluation of the TaDiff model for medical image analysis.
Key features include:
- Automated evaluation of model predictions against ground truth
- Support for multiple diffusion sampling methods (DDIM, DPM-Solver++)
- Quantitative metric calculation (Dice, MSE, etc.)
- Visualization of predictions and uncertainty
- Batch processing of patient sessions
- Ensemble prediction generation

The script processes 3D medical volumes by:
1. Identifying tumor-containing slices
2. Running diffusion-based predictions
3. Calculating evaluation metrics
4. Generating visualizations

Typical workflow:
1. Load pre-trained model checkpoint
2. Configure evaluation parameters
3. Process patient data
4. Save results and visualizations

Example usage:
    python test.py --patient_ids 17 42 --diffusion_steps 50 --num_samples 4
"""
from typing import List, Dict, Optional

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

from config.cfg_tadiff_net import config as train_config
from config.test_config import TestConfig
from src.tadiff_model import Tadiff_model
from src.tadiff_net.diffusion import GaussianDiffusion
from src.data.data_loader import load_data, val_transforms
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
from src.utils.image_processing import (
    # normalize_to_255,
    to_pil_image,
    prepare_image_batch,
    create_noise_tensor,
    extract_slice
)
from monai.data import CacheDataset, DataLoader
from torch.utils.data import Dataset

import inspect
import src.tadiff_net.diffusion as diffmod
from src.data.datasampler import normalize_mri_per_modality
print("DIFFUSION IMPORTED FROM:", diffmod.__file__)
print("TaDiff_inverse signature:", inspect.signature(GaussianDiffusion.TaDiff_inverse))


npz_keys= ['image', 'label', 'days', 'treatment', 'geno']

class TestLoader(Dataset):
    """
      - Choose a target session among all sessions (biased to produce past/middle/future ratios)
      - Choose k in {1,2,3} input sessions from remaining sessions WITH replacement
      - Pad inputs to exactly 3 and put target last → [in1,in2,in3,target]
      - Collapse session+modalities into channels, apply non_load_val_transforms (MONAI),
        then reshape back to (S=4, C_use, H, W, D)
    Returns a dict with tensors:
      'image': (S=4, C, H, W, D)
      'label': (S=4, H, W, D)
      'days': (S=4,)
      'treatment': (S=4,)
      'sample_info': metadata for debugging (dict)
    Args:
      file_dicts: list of dicts with keys npz_keys (paths to .npy)
      transform: non_load_val_transforms (val_transforms with LoadImaged removed)
      samples_per_patient: how many samples to draw per patient per epoch (len = n_patients * samples_per_patient)
      rng_seed: optional seed for reproducibility
    """
    def __init__(self, file_dicts: List[Dict], transform=None):
        self.file_dicts = file_dicts[:]
        self.transform = transform

    def __len__(self):
        return max(1, len(self.file_dicts)) 

    def _choose_target_session(self, S_all: int) -> int:
        """
        Choose a target session index in 0..S_all-1 with bucketed probabilities.
        We divide the session indices into 3 buckets (past/middle/future):
           past = first ceil(S_all/3) indices
           middle = next ceil(S_all/3)
           future = remaining indices
        We assign each bucket the total probability P_TARGET_BUCKET[*] and distribute uniformly
        inside the bucket (i.e., per-session weight = bucket_prob / bucket_size).
        This yields the overall mass near the requested bucket probs.
        """


    def __getitem__(self, patient_idx):
        file_dict = self.file_dicts[patient_idx]

        # Load arrays
        data = {k: np.load(file_dict[k]) for k in npz_keys}
        img_full = data['image']    # (C_times_S, D, H, W)
        lbl_full = data['label']    # (S_all, D, H, W)
        days_full = data['days']    # (S_all,)
        treat_full = data['treatment']  # (S_all,)
        geno_full = data['geno']  # (S_all,)

        S_all = int(lbl_full.shape[0])
        C_times_S, D, H, W = img_full.shape
        C_full = C_times_S // S_all
        assert C_full * S_all == C_times_S, f"image channels ({C_times_S}) not divisible by sessions ({S_all})"

        S_treat = treat_full.shape[0]
        if S_all != S_treat:
            treat_full = treat_full[: S_all - 1]
            days_full = days_full[: S_all - 1]


        print("Patient : ", patient_idx)
        print("img_full : ", img_full.shape)
        print("lbl_full : ", lbl_full.shape)
        print("days_full : ", days_full.shape)
        print("treat_full : ", treat_full.shape)
        print("days_full : ", days_full)
        print("treat_full : ", treat_full)
        # Reshape images to  S_all, C_full, H, W, D)
        # img_full = img_full.reshape( S_all, C_full D, H, W)
        # img_full = np.transpose(img_full, (1, 0, 3, 4, 2))  # (S, C, H, W, D)

        S, D, H, W = lbl_full.shape
        # Undo the bad reshape: go back to (S, H, W, D)
        lbl_full = lbl_full.reshape(S, H, W, D)
        # Now do the CORRECT axis move to get (S, D, H, W)
        # lbl_full = np.transpose(lbl_full, (0, 3, 1, 2)).astype(np.int16)


        img_full = img_full.reshape(C_full, S_all, H, W, D)  # (C, S, H, W, D)
        img_full = np.moveaxis(img_full, 0, 1)  # -> (S, C, H, W, D)
        img_full = normalize_mri_per_modality(img_full)

        print(img_full.shape)
        print(lbl_full.shape)


        assert img_full.shape[0] == lbl_full.shape[0]  # same number of sessions
        S, C, H, W, D = img_full.shape
        print("S, C, H, W, D:", S, C, H, W, D)
        print("img range:", img_full.min(), img_full.max())
        print("lbl unique:", np.unique(lbl_full))

        # Visual sanity: save a few slices straight from disk (no reordering) and after reshape
        import matplotlib.pyplot as plt
        # from disk: load the same file twice to show original memory ordering
        raw = np.load(file_dict['image'])  # original (C_times_S, D, H, W)
        # pick channel idx 0 of first modality before reshape:
        plt.imsave("debug_raw_chan0_slice100.png", raw[0, 100, :, :], cmap='gray')

        # after reshape:
        plt.imsave("debug_reshaped_s0_c0_slice100.png", img_full[0, 0, :, :, 100], cmap='gray')
        plt.imsave("debug_labelreshaped_s0_c0_slice100.png", lbl_full[0,  :, :, 50], cmap='gray')

        # Sel

        return {
            'image': img_full,      # (S, C, H, W, D)
            'label': lbl_full,      # (S, H, W, D)
            'days': days_full,        # (S,)
            'treatment': treat_full,  # (S,)
            'geno': geno_full
        }


def process_slice(
    slice_idx: int,
    session_idx: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    days: torch.Tensor,
    treatments: torch.Tensor,
    geno: torch.Tensor,
    model: Tadiff_model,
    device: torch.device,
    metrics: Dict,
    session_path: str,
    diffusion_steps: int,
    num_samples: int,
    target_idx: int
) -> Dict[str, Dict]:
    """
    Process a single 2D slice through the TaDiff prediction pipeline.
    
    The workflow includes:
    1. Data preparation and noise initialization
    2. Diffusion-based inverse process (DDIM or DPM-Solver++)
    3. Prediction post-processing
    4. Metric calculation
    
    Args:
        slice_idx: Z-index of the slice being processed
        session_idx: Index of the session being processed
        images: Input scans [1, num_sessions, C, H, W, D]
        labels: Segmentation masks [1, num_sessions, H, W, D]
        days: Treatment day values [1, num_sessions]
        treatments: Treatment codes [1, num_sessions]
        model: Loaded Tadiff_model instance
        device: Target device
        metrics: Dictionary of metric functions
        session_path: Directory for saving results
        diffusion_steps: Number of diffusion steps
        num_samples: Number of predictions to generate
        target_idx: Index of target session for prediction
        
    Returns:
        Dict[str, Dict]: Dictionary containing:
            - Keys: Sample identifiers (f"sample_{n}" or "ensemble")
            - Values: Dictionary of metric scores
    """
    # Move input tensors to device
    images = images.to(device)
    labels = labels.to(device)
    days = days.to(device)
    treatments = treatments.to(device)
    geno = geno.to(device)
    geno = geno.repeat(num_samples, 1)
    # Prepare data
    print("In process slice function, loaded image",images.shape)
    b, s, c, h, w, z = images.shape
    # Reshape images to separate modalities and sessions
    # images = images.view(b, 4, -1, h, w, z)  # t1, t1c, flair, t2
    # images = images.permute(0, 2, 1, 3, 4, 5)  # b, s, c, h, w, z
    if c ==4: 
        images = images[:, :, :-1, :, :, :]  # remove T2 modal, b, s, c-1, h, w, z
        print("removed T2 Modal in process slice",images.shape)

    # Get target session indices
    session_indices = np.array([
        session_idx - 3,
        session_idx - 2,
        session_idx - 1,
        session_idx
    ])
    session_indices[session_indices < 0] = 0
    session_indices = list(session_indices)
    print("session_indices ",session_indices )
    # Extract relevant slices
    print("Extract relevant slices")
    print("labels ", labels.shape)

    masks = labels[0, session_indices, :, :, :]
    print("masks ", masks.shape)
    masks = masks[:, :, :, [slice_idx]*num_samples].permute(3, 0, 1, 2)
    print("masks ", masks.shape)

    seq_imgs = images[0, session_indices, :, :, :, :]
    seq_imgs = seq_imgs[:, :, :, :, [slice_idx]*num_samples].permute(4, 0, 1, 2, 3)
    print("seq_imgs ", seq_imgs.shape)

    # Create noise and prepare target images
    noise = torch.randn((num_samples, 3, h, w), device=device)
    x_t = seq_imgs.clone()
    x_0 = []
    
    # Set up target indices
    i_tg = target_idx * torch.ones((num_samples,), dtype=torch.int8, device=device)
    
    # Prepare input tensors
    print("seq_imgs ", seq_imgs.shape)
    for i, j in zip(range(num_samples), i_tg):
        x_0.append(seq_imgs[[i], j, :, :, :])
        x_t[i, j, :, :, :] = noise[i, :, :, :]
    x_0 = torch.cat(x_0, dim=0)
    x_t = x_t.reshape(num_samples, len(session_indices) * 3, h, w)
    
    # Prepare condition tensors
    daysq = days[0, session_indices].repeat(num_samples, 1).to(device)
    treatments_q = treatments[0, session_indices].repeat(num_samples, 1).to(device)
    
    # Run diffusion
    diffusion = GaussianDiffusion(T=diffusion_steps, 
                                schedule="linear",
                                device=device
                                )
    pred_img, seg_seq = diffusion.TaDiff_inverse(
        net=model,
        start_t=diffusion_steps,
        steps=diffusion_steps,
        x=x_t,
        intv=[daysq[:, i].to(torch.float32) for i in range(4)],
        treat_cond=[treatments_q[:, i].to(torch.float32) for i in range(4)],
        i_tg=i_tg,
        geno=geno,
        device=device
    )
    
    # Process predictions
    seg_seq = torch.sigmoid(seg_seq)
    print("pred_image ", pred_img.shape)
    print("ground_truth ", x_0.shape)

    predictions = {
        'images': pred_img,  # (samples, C, H, W)  # C = 3 modality 
        'masks': seg_seq,  # (samples, 4, H, W)
        'ground_truth': x_0,  # (samples, C, H, W)
        'target_masks': masks  # (samples, 4, H, W)
    }
    
    # Calculate metrics and save visualizations
    slice_scores = evaluate_predictions(
        predictions=predictions,
        metrics=metrics,
        session_idx=session_idx,
        slice_idx=slice_idx,
        session_path=session_path
    )
    
    return slice_scores

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
    avg_img = torch.mean(predictions['images'], 0)  # (3, H, W)
    avg_mask_pred = torch.sigmoid(predictions['masks'])
    avg_mask_pred = torch.mean(avg_mask_pred, 0)    # (4, H, W)
    
    # Calculate uncertainty maps
    img_std = torch.std(predictions['images'], 0)  # (3, H, W) - t1,t1c,flair 
    seg_seq_std = torch.std(predictions['masks'], 0)  # (4, H, W) - uncertainty in sequence
    
    # Prepare visualization data
    images = {
        'prediction': predictions['images'][0].cpu().numpy(),  # Use first sample for visualization
        'ground_truth': predictions['ground_truth'][0].cpu().numpy()
    }
    masks = {
        'prediction': avg_mask_pred.cpu().numpy().astype(np.float32),  # Convert to float32
        'ground_truth': predictions['target_masks'][0].cpu().numpy().astype(np.float32),  # Convert to float32
        'uncertainty': img_std.cpu().numpy().astype(np.float32),  # Add uncertainty map
        'sequence_uncertainty': seg_seq_std.cpu().numpy().astype(np.float32)  # Add sequence uncertainty
    }
    
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
        pred_mask_pil = visualizer.to_pil(masks['prediction'][-1, :, :])
        gt_mask_pil = visualizer.to_pil(masks['ground_truth'][-1, :, :])
        
        # Save masks
        pred_mask_pil.save(os.path.join(session_path, f"{file_prefix}-pred-mask.png"))
        gt_mask_pil.save(os.path.join(session_path, f"{file_prefix}-gt-mask.png"))
        
        # Save uncertainty maps
        visualizer.plot_uncertainty(masks['uncertainty'][0, :, :], 
                                    os.path.join(session_path, f"{file_prefix}-uncertainty_t1.png"), 
                                    overlay=avg_img[0, :, :].cpu().numpy())
        visualizer.plot_uncertainty(masks['uncertainty'][1, :, :], 
                                    os.path.join(session_path, f"{file_prefix}-uncertainty_t1c.png"),
                                    overlay=avg_img[1, :, :].cpu().numpy())
        visualizer.plot_uncertainty(masks['uncertainty'][2, :, :], 
                                    os.path.join(session_path, f"{file_prefix}-uncertainty_flair.png"),
                                    overlay=avg_img[2, :, :].cpu().numpy())
        
        visualizer.plot_uncertainty(masks['sequence_uncertainty'][0, :, :], 
                                    os.path.join(session_path, f"{file_prefix}-uncertainty_mask.png"),
                                    overlay=avg_img[2, :, :].cpu().numpy())
        # uncertainty_pil = visualizer.to_pil(masks['uncertainty'][-1, :, :])
        # seq_uncertainty_pil = visualizer.to_pil(masks['sequence_uncertainty'][-1, :, :])
        # uncertainty_pil.save(os.path.join(session_path, f"{file_prefix}-uncertainty.png"))
        # seq_uncertainty_pil.save(os.path.join(session_path, f"{file_prefix}-sequence-uncertainty.png"))
        
        # Save images with overlays and contours for each modality
        for j in range(3):  # For each modality
            pred_img = visualizer.to_pil(images['prediction'][j])
            gt_img = visualizer.to_pil(images['ground_truth'][j])
            
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
    
    print("predictions['target_masks'] ", predictions['target_masks'].shape)
    pm = predictions['masks']    # (N, S, H, W) float (prob)
    gm = predictions['target_masks']  # (N, S, H, W)

    print("preds dtype/max/min:", pm.dtype, pm.min().item(), pm.max().item())
    print("gt   dtype/max/min:", gm.dtype, gm.min().item(), gm.max().item())
    print("gt unique (sample 0, session 0):", torch.unique(gm[0,0])[:20])
    print("gt sum per session (sample 0):", [gm[0,s].sum().item() for s in range(gm.shape[1])])

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

def get_test_files(config: TestConfig):
    """Get list of test files for each patient"""
    test_files = []
    for patient_id in config.patient_ids:
        file_dict = {
            key: os.path.join(config.data_root, f'{patient_id}_{key}.npy')
            for key in npz_keys
        }
        test_files.append(file_dict)
    return test_files

def load_data(test_files, config: TestConfig):
    """Load and prepare test data"""
    test_dataset = TestLoader(file_dicts=test_files, transform=val_transforms)
    return DataLoader(test_dataset, batch_size=1, shuffle=False)

def setup_model(config: TestConfig, train_config, device: str):
    """Initialize and load model"""

    ckpt = torch.load(config.model_checkpoint, map_location="cpu")
    print(ckpt.keys())
    print("hparams:", ckpt.get("hyper_parameters", {}).keys())
    print("saved config:", ckpt.get("hyper_parameters", {}).get("config", None))


    model = Tadiff_model.load_from_checkpoint(
        config.model_checkpoint,
        model_channels=config.model_channels,
        num_heads=config.num_heads,
        num_res_blocks=config.num_res_blocks,
        strict=True,
        config=train_config,
        map_location="cuda:0"
    )
    model.to(device)
    model.eval()
    return model

def main():
    # Load configuration
    config = TestConfig()
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Initialize components
    model = setup_model(config, train_config, device)
    metrics_calculator = MetricsCalculator(device, config.dice_thresholds)
    visualizer = Visualizer(config.colors)
    
    # Load data
    test_files = get_test_files(config)
    dataloader = load_data(test_files, config)
    
    # Create output directory
    os.makedirs(config.save_path, exist_ok=True)
    
    # Create or load CSV file
    csv_path = os.path.join(config.save_path, 'test_scores.csv')
    if os.path.exists(csv_path):
        all_scores = pd.read_csv(csv_path, index_col=0).to_dict('index')
    else:
        all_scores = {}
    
    # Process each patient

    for i, batch in enumerate(dataloader):
        patient_id = config.patient_ids[i]
        print(f'Processing patient {patient_id}')
        # Process each session
        print("patient has sessions : ", batch['label'].shape[1])
        for session_idx in range(batch['label'].shape[1]):
            # Process each slice
            print("slices ", batch['label'].shape)
            z_mask_size = calculate_tumor_volumes(batch['label'][0])  # labels[0]: [S, H, W, D]

            slice_idx = int(get_slice_indices(z_mask_size, top_k=1))
            print(slice_idx)
            # Get predictions using the first process_slice()
            slice_scores = process_slice(
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
                session_path=os.path.join(config.save_path, f'p-{patient_id}', f'slice-{slice_idx:03d}'),
                diffusion_steps=config.diffusion_steps,
                num_samples=config.num_samples,
                target_idx=config.target_session_idx
            )
            
            # Flatten the nested dictionary structure
            flattened_scores = {}
            for sample_key, sample_metrics in slice_scores.items():
                for metric_name, metric_value in sample_metrics.items():
                    flattened_scores[f"{sample_key}_{metric_name}"] = metric_value
            
            # Add metadata
            flattened_scores['patient_id'] = patient_id
            flattened_scores['session_idx'] = session_idx
            flattened_scores['slice_idx'] = slice_idx
            
            # Save metrics immediately
            score_key = f'{patient_id}_slice_{slice_idx:03d}'
            all_scores[score_key] = flattened_scores
            
            # Save to CSV after each slice
            pd.DataFrame.from_dict(all_scores, orient='index').to_csv(csv_path)
            print(f"Saved scores for {score_key}")
            # for slice_idx in range(batch['label'].shape[-1]):
            #     # Skip slices with small tumor size
            #     if torch.sum(batch['label'][0, :, :, :, slice_idx]) < config.min_tumor_size:
            #         continue
                
            #     # Get predictions using the first process_slice()
            #     slice_scores = process_slice(
            #         slice_idx=slice_idx,
            #         session_idx=session_idx,
            #         images=batch['image'],
            #         labels=batch['label'],
            #         days=batch['days'],
            #         treatments=batch['treatment'],
            #         model=model,
            #         device=device,
            #         metrics=metrics_calculator,
            #         session_path=os.path.join(config.save_path, f'p-{patient_id}', f'slice-{slice_idx:03d}'),
            #         diffusion_steps=config.diffusion_steps,
            #         num_samples=config.num_samples,
            #         target_idx=config.target_session_idx
            #     )
                
            #     # Flatten the nested dictionary structure
            #     flattened_scores = {}
            #     for sample_key, sample_metrics in slice_scores.items():
            #         for metric_name, metric_value in sample_metrics.items():
            #             flattened_scores[f"{sample_key}_{metric_name}"] = metric_value
                
            #     # Add metadata
            #     flattened_scores['patient_id'] = patient_id
            #     flattened_scores['session_idx'] = session_idx
            #     flattened_scores['slice_idx'] = slice_idx
                
            #     # Save metrics immediately
            #     score_key = f'{patient_id}_slice_{slice_idx:03d}'
            #     all_scores[score_key] = flattened_scores
                
            #     # Save to CSV after each slice
            #     pd.DataFrame.from_dict(all_scores, orient='index').to_csv(csv_path)
            #     print(f"Saved scores for {score_key}")
    
    print("All processing complete. Final scores saved to:", csv_path)

if __name__ == '__main__':
    main()
