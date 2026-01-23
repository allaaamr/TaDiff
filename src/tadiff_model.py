import numpy as np
import torch
from src.tadiff_net.tadiff_unet_arch import TaDiff_Net
# import wandb # logging metrics
import os
from pytorch_lightning import LightningModule, Callback
from torch.optim import AdamW, SGD
from src.tadiff_net.ssim import SSIM
import matplotlib.pyplot as plt
from src.visualization.visualizer import (
    # plot_uncertainty_figure,
    # save_visualization_results,
    create_directory,
    Visualizer
)
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from src.tadiff_net.diffusion import GaussianDiffusion
from src.evaluation.metrics import (
    # setup_metrics,
    # calculate_metrics,
    calculate_tumor_volumes,
    get_slice_indices,
    MetricsCalculator
)
import torch.nn.functional as F
from typing import List, Dict, Optional

from monai.losses.dice import DiceLoss, GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric


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
    # predictions['ground_truth'] = predictions['ground_truth'].squeeze(0)
    # predictions['target_masks'] = predictions['target_masks'].squeeze(0)

    # print("predictions['images'] ", predictions['images'].shape)
    # print("predictions['masks']" , predictions['masks'].shape)
    # print("predictions['ground_truth']" ,predictions['ground_truth'].shape)
    # print("predictions['target_masks']" , predictions['target_masks'].shape)

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

class Tadiff_model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = config
        self._model = TaDiff_Net(
            image_size=self.cfg.image_size, 
            in_channels=self.cfg.in_channels-1, 
            out_channels=self.cfg.out_channels,
            # num_intv_time=self.cfg.num_intv_time,
            model_channels=self.cfg.model_channels, 
            num_res_blocks=self.cfg.num_res_blocks, 
            channel_mult=self.cfg.channel_mult,
            attention_resolutions=self.cfg.attention_resolutions, 
            num_heads=self.cfg.num_heads,
            geno=self.cfg.geno,
            )
        self.visualize = True
        # if self.cfg.precision=='16':
        #     self._model.convert_to_fp16()
        
        # Sets up the diffusion process (how noise is added / removed).
        self.diffusion = GaussianDiffusion(T=self.cfg.max_T, schedule=self.cfg.ddpm_schedule)#'linear')
        alphabar_np = np.cumprod(1-np.linspace(1e-4, 2e-2, self.cfg.max_T))
        self.alphabar = torch.tensor(alphabar_np, dtype=torch.float32)
        # self.diffusion = LinearDiffusion(T=self.cfg.max_T)#'linear')

        # Dice loss and metric for the segmentation outputs (auxiliary task).
        self.best_val_loss = None
        self.best_val_epoch = 0
        self.val_step_outputs = [] # for callback
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        # self.dilation3 = torch.ones(1,1,3,3)
        self.dilation_filters = torch.ones(1,1,11,11) / 10.
        # self.dice = DiceLoss(include_background=False, sigmoid=True)
        self.dice = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
                             to_onehot_y=False, sigmoid=True, reduction="none")
        # self.dice = GeneralizedDiceFocalLoss( to_onehot_y=False, sigmoid=True, reduction="none")
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        # self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch") # per class dice
        self.loss_function = F.mse_loss

    def forward(self, x, timesteps, intv_t, treat_code, geno, i_tg=None):
        return self._model(x, timesteps, intv_t,  treat_code, i_tg, geno)
    
    def load_model(self, path=None, device='cuda:0'):
        # for loading old trained model without using pytorch-lightning
        if path is not None:
            self._model.load_state_dict(torch.load(path, map_location=device), strict=False)
        # self._model.load_state_dict(torch.load(path, map_location=device), strict=False)
        self._model.eval().to(device)
        print('Model Created!')

    def _save_gray(self, img2d, path):
        arr = img2d.detach().float().cpu().numpy()
        plt.imsave(path, arr, cmap="gray")

    @torch.no_grad()
    def _debug_save_denoise_triplet(self, gt_img, xt, eps_pred, t_int, out_dir, prefix=""):
        """
        gt_img, xt, eps_pred: (B,3,H,W) tensors
        t_int: int timestep used for xt
        """
        os.makedirs(out_dir, exist_ok=True)

        # alphabar[t-1] is scalar
        alphabar_t = self.diffusion.alphabar[t_int - 1].to(gt_img.device).view(1, 1, 1, 1)
        x0_hat = (xt - torch.sqrt(1 - alphabar_t) * eps_pred) / (torch.sqrt(alphabar_t) + 1e-8)

        modal_names = ["t1", "t1c", "flair"]
        for m, name in enumerate(modal_names):
            self._save_gray(gt_img[0, m],  os.path.join(out_dir, f"{prefix}x0_gt_{name}.png"))
            self._save_gray(xt[0, m],      os.path.join(out_dir, f"{prefix}xt_t{t_int}_{name}.png"))
            self._save_gray(x0_hat[0, m],  os.path.join(out_dir, f"{prefix}x0hat_{name}.png"))

    @torch.no_grad()
    def full_inverse_eval_once(self, batch, epoch, out_dir, num_samples=1):
        self._model.eval()

        # imgs = batch["image"].to(self.device)         # (B,S,C,H,W)
        # label = batch["label"].to(self.device)        # (B,S, H,W) 
        imgs0 = batch["image"].to(self.device).clone()     # clean copy
        label0 = batch["label"].to(self.device).clone()
        days = batch["days"].to(self.device)          # (B,S)
        treatments = batch["treatments"].to(self.device)
        geno = batch.get("geno", None)
        if geno is not None:
            geno = geno.to(self.device)

        imgs = imgs0.clone()     # working tensor you can corrupt safely
        label = label0.clone()

        B, S, C, H, W = imgs.shape
        assert S == 4 and C == 3, f"Expected (B,4,3,H,W) but got {imgs.shape}"

        # use the first subject in batch, repeat num_samples times
        seq_imgs = imgs[0:1].repeat(num_samples, 1, 1, 1, 1)     # (N,4,3,H,W)
        masks = label[0:1].repeat(num_samples, 1, 1, 1)       # (N,4,H,W) if that's your label format
        daysq = days[0:1].repeat(num_samples, 1)                 # (N,4)
        treatments_q = treatments[0:1].repeat(num_samples, 1)    # (N,4)

        # target session index (last session)
        i_tg = torch.full((num_samples,), 3, dtype=torch.long, device=self.device)

        # ground-truth target image
        x_0 = seq_imgs[:, 3, :, :, :]                            # (N,3,H,W)

        # start from PURE NOISE in target slot, keep history sessions intact
        x_t = seq_imgs.clone()
        x_t[:, 3, :, :, :] = torch.randn((num_samples, 3, H, W), device=self.device)

        # flatten to TaDiff input format (N, 12, H, W)
        x_in = x_t.reshape(num_samples, S * C, H, W).contiguous()

        # IMPORTANT: use the SAME diffusion object as training (schedule & T)
        # and start_t = T, steps = T for full inverse
        T = int(self.diffusion.T)

        pred_img, seg_seq = self.diffusion.TaDiff_inverse(
            net=self,  # LightningModule has forward() defined -> ok
            start_t=T,
            steps=T,
            x=x_in,
            intv=[daysq[:, i].float() for i in range(4)],
            treat_cond=[treatments_q[:, i].float() for i in range(4)],
            i_tg=i_tg,
            geno=geno,
            device=self.device
        )

        # pred_img should be (N,3,H,W), seg_seq (N,4,H,W) or similar depending on impl
        seg_seq = torch.sigmoid(seg_seq)
        print("seq_seq " ,seg_seq.shape)
        print("masks " ,masks.shape)

        predictions = {
            "images": pred_img,
            "masks": seg_seq,
            "ground_truth": x_0,
            "target_masks": masks[:, :, :, :] 
        }

        metrics = MetricsCalculator(str(self.device))
        evaluate_predictions(
            predictions=predictions,
            metrics=metrics,
            session_idx=3,
            slice_idx=2,
            session_path=os.path.join(out_dir, f"epoch_{epoch}")
        )

        # print ranges to debug normalization mismatches
        print("[FULL INV] GT  min/max/mean/std:",
            x_0.min().item(), x_0.max().item(), x_0.mean().item(), x_0.std().item())
        print("[FULL INV] GEN min/max/mean/std:",
            pred_img.min().item(), pred_img.max().item(), pred_img.mean().item(), pred_img.std().item())


    def configure_optimizers(self):
        if self.cfg.opt == 'adamw':
            optimizer = AdamW(self.trainer.model.parameters(), 
                            lr=float(self.cfg.lr), 
                            weight_decay=self.cfg.weight_decay
                            )
        else:
            optimizer = SGD(self.trainer.model.parameters(), 
                            lr=float(self.cfg.lr), 
                            momentum = 0.9, 
                            nesterov = True,
                            weight_decay=self.cfg.weight_decay
                            )
            
        
        # self.ssim_score = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
        # self.trainer.train_dataloader  # now accessible :)
        num_devices = (
            torch.cuda.device_count()
            if self.trainer.num_devices == -1
            else int(self.trainer.num_devices)
        )
        
        # self.trainer.reset_train_dataloaders(self)
        # if self.cfg.max_epochs > 0:
        #     total_steps = (
        #         (1 + len(self.trainer.datamodule.train_dataloader())
        #         // self.cfg.accumulate_grad_batches
        #         // num_devices)
        #         * self.cfg.max_epochs 
        #     )
        # else:
        #     total_steps = self.cfg.max_steps
        if getattr(self.cfg, "max_steps", 0) and self.cfg.max_steps > 0:
            total_steps = int(self.cfg.max_steps)
        else:
            # Fallback: approximate with max_epochs
            # (you can refine this later if you want steps_per_epoch)
            total_steps = int(self.cfg.max_epochs)

        # Avoid zero steps
        if total_steps <= 0:
            total_steps = 1


        
        scheduler = {
            "scheduler": WarmupCosineSchedule(
                optimizer, warmup_steps=self.cfg.warmup_steps, t_total=total_steps),
            "interval": "step",  # runs per batch rather than per epoch
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]
        # return optimizer


    def get_loss(self, batch, epoch, flag, mode='train'):
        days, treatments, geno = batch["days"], batch["treatments"], batch["geno"]
        
        imgs0 = batch["image"].to(self.device).clone()     # clean copy
        label0 = batch["label"].to(self.device).clone()
        n_sess = label0.shape[1]

        imgs = imgs0.clone()     # working tensor you can corrupt safely
        label = label0.clone()

        b, s, c, h, w = imgs.shape
        s1_days, s2_days, s3_days, t_days = days[:, 0], days[:,1], days[:, 2], days[:, 3]
        # Always use the future / target exam as the prediction target
        # Here we assume the future scan is the last session (index -1).
        i_tg = -torch.ones((b,), dtype=torch.int64, device=self.device)

        # Build conditioning vectors
        treat1, treat2, treat3, treat_t = treatments[:,0], treatments[:,1], treatments[:,2], treatments[:,3]
        intvs = [s1_days.to(torch.float32), s2_days.to(torch.float32), s3_days.to(torch.float32), t_days.to(torch.float32)]
        treat_cond = [treat1.to(torch.float32), treat2.to(torch.float32), treat3.to(torch.float32), treat_t.to(torch.float32)]
        
        # Extract the target image & mask

        idx_b = torch.arange(b, device=imgs.device)          # [0, 1, ..., B-1]
        idx_s = i_tg.to(imgs.device).long()                  # target session index per batch
        # imgs: [B, S, C, H, W] -> gt_img: [B, C, H, W]
        gt_img = imgs0[idx_b, idx_s, ...].to(torch.float32)

        # label: [B, S, 4, H, W] -> gt_label: [B, C, H, W]
        gt_label = label0[idx_b, idx_s, ...].to(torch.float32)
        t = torch.randint(1, self.diffusion.T + 1, [gt_img.shape[0]])
        w_tg = self.alphabar[t - 1]
        
        # Sample a diffusion timestep and add noise
        # xt: (B, C, H, W) – noised version of gt_img at timestep t.
        # epsilon: (B, C, H, W) – the true noise that was added (the diffusion target).
        xt, epsilon = self.diffusion.sample(gt_img.to(torch.float32), t)
        xt_img = xt 
        # First they mark sequences where the last historical session coincides with the target day:
        maskout_batch = (s3_days == t_days) 
        # Then for each batch element:
        for i, j in zip(range(b), i_tg): # Pick the target session j = i_tg[i]
            if maskout_batch[i]:
                imgs[i, :, :, :, :] = 0. # remove all original images
                label[i, :, :, :] = 0  # remove all original masks
            label[i, j, :, :] = gt_label[i, :, :]  # correct target mask
            imgs[i, j, :, :, :] = xt[i, :, :, :]   # noised target image
        
        #reshape to give, one giant image per batch, not a time sequence.
        xt = imgs.reshape(b, s*c, h, w).contiguous() 
        t = t.view(gt_img.shape[0]).to(self.device)
        out = self.forward(xt.to(torch.float32), t.to(torch.float32), 
                           intv_t=intvs, treat_code=treat_cond, geno=geno, i_tg=i_tg)

        # Summary: Because the network is trained to:
            # see all sessions as conditioning input,
            # but only predict the diffusion noise & segmentation for one target session,
            # where the target session’s image is replaced with a noisy version.

        # Compute loss and backprop
        r"""
        The loss weights create a spatial importance map telling the model:
        “These pixels matter more for the noise-prediction loss.”
        Why?
        Because in medical imaging most of the image is background, and the tumor or lesion is usually:
        very small  /  spatially sparse / clinically important
        Without loss weights, the MSE between predicted noise and true noise would be dominated by background pixels, making the model:
        learn background very well but learn tumor regions poorly
        """
        label = label.float()
        loss_weigths = torch.sum(label, dim=1, keepdim=True) # range 0 -4
        loss_weigths = loss_weigths * torch.exp(-loss_weigths)
        # This spreads the weights to neighboring pixels.
        loss_weigths = F.conv2d(loss_weigths, self.dilation_filters.to(loss_weigths.device), padding='same') + 1.
       
        img_pred, mask_pred = out[:, 4:7, :, :], out[:, 0:4, :, :]
        # print("img_pred ", img_pred.shape)
        loss1 = torch.mean(loss_weigths * (img_pred - epsilon)**2)
        mse = self.loss_function(img_pred, epsilon) # without weights on tumor
        
        dice_loss = self.dice(mask_pred, label) # all segementaed masks b, 4, 1, 1
    
        # dice_loss = dice_loss * w_tg.view(b, 1)  # weighted the loss one more time, w_tg ** 3 for target image, but for refence image only appply w_tg
        # weighted future tumor loss based on nosized level
        for i, j in zip(range(b), i_tg):
            # dice_loss[i, j] = dice_loss[i, j] * torch.sqrt(w_tg[i])
            if maskout_batch[i]:
                loss_ij = dice_loss[i, j] * torch.sqrt(w_tg[i])
                dice_loss[i, :] = 0.
                dice_loss[i, j] = loss_ij  # weight target image loss, w_tg ** 3
            else:
                dice_loss[i, j] = dice_loss[i, j] * torch.sqrt(w_tg[i]) # w_tg[i]**2  # weight target image loss, w_tg ** 3
            
        # w_dims = (b,) + tuple((1 for _ in dice_loss.shape[1:])) 
        # dice_loss = dice_loss * w_tg.view(b, 1)  # weighted the loss one more time, w_tg ** 3 for target image, but for refence image only appply w_tg
        
        loss = loss1 + torch.mean(dice_loss) * self.cfg.aux_loss_w
        
        # mask_pred = F.sigmoid(mask_pred)
        mask_pred1 = torch.sigmoid(mask_pred)
        mask_pred = (mask_pred1 > 0.5) * 1  # fix threshold for segment mask 0.5
        self.dice_metric(mask_pred, label)
        dice_last =  self.dice_metric.aggregate() # only mean 4 mask dices
        self.dice_metric.reset()
        # if mode == 'train':
        #     self.dice_metric(mask_pred, label)
        #     dice_last =  self.dice_metric.aggregate() # only mean 4 mask dices
        #     self.dice_metric.reset()
        # else: 
        #     self.dice_metric(mask_pred[:, 3:4,:, :], label[:, 3:4, :, :])
        #     dice_last = self.dice_metric.aggregate()#.item() # only last masks 
        #     self.dice_metric.reset()

        if mode == 'val' and flag and epoch%10 == 0:

            # choose a fixed visualization timestep for val
            fixed_t = int(getattr(self.cfg, "debug_vis_fixed_t", 50))
            fixed_t = max(1, min(fixed_t, self.diffusion.T))  # clamp

            # --- 1) build fixed timesteps tensor ---
            t_vis = torch.full((b,), fixed_t, device=gt_img.device, dtype=torch.long)

            # --- 2) create xt at this fixed timestep from the *clean* gt_img ---
            # diffusion.sample expects t on CPU in your implementation
            xt_vis, _ = self.diffusion.sample(gt_img.to(torch.float32), t_vis.cpu())
            xt_vis = xt_vis.to(gt_img.device)  # (B, 3, H, W)

            # --- 3) pack model input exactly like training: replace target session with xt_vis ---
            imgs_vis = imgs.clone()  # imgs currently contains random-xt inserted; that's fine, we'll overwrite target

            for i, j in zip(range(b), i_tg):
                imgs_vis[i, j, :, :, :] = xt_vis[i]

            x_in_vis = imgs_vis.reshape(b, s * c, h, w).contiguous()

            # --- 4) run model at fixed t to get eps prediction ---
            out_vis = self.forward(
                x_in_vis.to(torch.float32),
                t_vis.to(torch.float32),
                intv_t=intvs,
                treat_code=treat_cond,
                geno=geno,
                i_tg=i_tg
            )
            eps_pred_vis = out_vis[:, 4:7, :, :]   # (B, 3, H, W)

            # --- 5) convert eps_pred -> x0_hat using alphabar at fixed_t ---
            alphabar_t = self.diffusion.alphabar[fixed_t - 1].to(gt_img.device).view(1, 1, 1, 1)
            x0_hat_vis = (xt_vis - torch.sqrt(1.0 - alphabar_t) * eps_pred_vis) / (torch.sqrt(alphabar_t) + 1e-8)

            # --- (optional but recommended) clamp for stable visualization ---
            # If your data is normalized to [-1, 1], keep this. If not, remove it.
            # x0_hat_vis = torch.clamp(x0_hat_vis, -1.0, 1.0)


            predictions = {
                'images': x0_hat_vis,     # <-- denoised estimate (x0_hat), NOT eps
                'masks': mask_pred1,      # probs
                'ground_truth': gt_img,   # x0
                'target_masks': label
            }
            device = "cuda:0"
            metrics = MetricsCalculator(device)

            # Calculate metrics and save visualizations
            slice_scores = evaluate_predictions(
                predictions=predictions,
                metrics=metrics,
                session_idx=3,
                slice_idx=2,
                session_path= f"results/tadiff_1PCos_single_inverse/{epoch}"
            )

            
            self.full_inverse_eval_once(
                batch=batch,
                epoch=epoch,
                out_dir="results/tadiff_1PCos_full_inverse",
                num_samples=1
            )

                
        return loss, mse, dice_last

    
    
    def training_step(self, batch):
        loss, mse, dice_seg = self.get_loss(batch, mode='train')
        self.log("train_loss", loss,  sync_dist=True, on_epoch=True, prog_bar=True) # on_epoch=False default
        self.log("train_mse", mse,  sync_dist=True, on_epoch=False, prog_bar=False) # on_epoch=False default
        self.log("train_dice", dice_seg,  sync_dist=True, on_epoch=False, prog_bar=False) # on_epoch=False default
        return {"loss": loss, "mse": mse, "dice_seg": dice_seg}

    def validation_step(self, batch):
        loss, mse, dice = self.get_loss(batch, mode='val')
        self.val_step_outputs.append({"val_loss": loss})
        self.log("val_loss", loss.item(), sync_dist=True, prog_bar=False) # on_epoch=True default
        self.log("val_mse", mse.item(), sync_dist=True, prog_bar=False) # on_epoch=True default
        self.log("val_dice", dice.item(), sync_dist=True, prog_bar=False) # on_epoch=True default
        return {"val_loss": loss, "val_mse": mse, "val_dice": dice}

# class MyCallback(Callback):
#     def __init__(self, batch, config):
#         super().__init__()
#         self.batch = batch
#         self.cfg = config
#         # self.img = self.batch["image"]
#         # b, s, c, h, w
#         img_label = torch.cat([self.batch["image"], self.batch["label"].unsqueeze(2)], dim=2)
#         days = self.batch["days"]
#         treatments = self.batch["treatments"]
#         # n_sess = img_label.shape[1]
#         # self.val_labels = img_label[:, :, n_sess-1, :, :]
#         self.val_labels = img_label[:, :, -1, :, :] # 4sess, h, w
#         self.img_cond = img_label[:, :-1, :-1, :, :] # c-modal, 3sess, h, w
#         self.img_for_noise = img_label[:, -1, :-1, :, :]  # c-modal, 1sess,  h, w
#         b, s, c, h, w = self.img_cond.shape
#         self.img_cond = self.img_cond.reshape(b, s*c, h, w).contiguous()
#         self.gt_preimg = img_label#[:, :, :-1, :, :]
        
#         s1_days, s2_days, s3_days, t_days = days[:, 0], days[:,1], days[:, 2], days[:, 3]
#         # intvs = [s1_days.to(device), s2_days.to(device), t_days.to(device)]
#         # print(f'treat_cond: {treat_cond[0]}')
#         self.intvs = [s1_days.to(torch.float32), s2_days.to(torch.float32), 
#                       s3_days.to(torch.float32), t_days.to(torch.float32)]
        
        
#         treat1, treat2, treat3, treat_t = treatments[:,0], treatments[:,1], treatments[:,2], treatments[:,3]
#         self.treat_cond = [treat1.to(torch.float32), treat2.to(torch.float32),  
#                       treat3.to(torch.float32), treat_t.to(torch.float32)]
        
#         # zero_mask = torch.zeros_like(self.val_labels).unsqueeze(2)
#         noise = torch.randn((self.img_for_noise.shape))
#         self.val_imgs = torch.cat([self.img_cond, noise], dim=1)

#         # self.diffusion = LinearDiffusion(T=1000)
#         # self.diffusion = GaussianDiffusion(T=1000, schedule='linear')
#         self.diffusion = GaussianDiffusion(T=int(self.cfg.max_T), schedule=self.cfg.ddpm_schedule)
#         # self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
    
    
#     def on_validation_epoch_end(self, trainer, pl_module):
#         # clean up artifacts cache
#         # c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
#         # c.cleanup(wandb.util.from_human_size("0GB"))
        
#         mean_val_loss = torch.stack([torch.tensor(x["val_loss"].clone().detach()) for x in pl_module.val_step_outputs]).mean()
#         # pl_module.log("val_avg_dice", mean_val_dice, sync_dist=True )
#         # pl_module.log("val_avg_loss", mean_val_loss, sync_dist=True)
#         if pl_module.best_val_loss is None:
#             pl_module.best_val_loss = mean_val_loss
#             pl_module.best_val_epoch = pl_module.current_epoch
#         elif mean_val_loss < pl_module.best_val_loss:
#             pl_module.best_val_loss = mean_val_loss
#             pl_module.best_val_epoch = pl_module.current_epoch
            
#         if pl_module.global_rank == 0:
#             print("on_validation_epoch_end...")
#             print(
#                 f"current epoch: {pl_module.current_epoch} "
#                 f"current mean loss: {mean_val_loss:.4f}"
#                 f"\nbest mean loss: {pl_module.best_val_loss:.4f} "
#                 f"at epoch: {pl_module.best_val_epoch}"
#             )
#             # self.log("best mean loss:",pl_module.best_val_loss)
#             # self.log("at best epoch:", pl_module.best_val_epoch)
#         val_imgs = self.val_imgs.to(device=pl_module.device) # img[:, 0:9, :, :].unsqueeze(1)
#         # val_labels = self.val_labels.to(device=pl_module.device) # img[:, 9:12, :, :]
#         # timesteps = [t.to(device=pl_module.device) for t in self.timesteps]
#         # Get model prediction
#         intvs = [intv.to(device=pl_module.device) for intv in self.intvs]
#         treat_cond = [treat.to(device=pl_module.device) for treat in self.treat_cond]
#         preds, aux_out = self.diffusion.TaDiff_inverse2(pl_module, 
#                                         start_t=self.cfg.max_T//1.5, #600, 
#                                         steps=self.cfg.max_T//1.5, #600,
#                                         x=val_imgs, 
#                                         intv=intvs, 
#                                         treat_cond=treat_cond,
#                                         # days=self.days.to(device=pl_module.device), 
#                                         # treat=self.treatments.to(device=pl_module.device),
#                                         device=pl_module.device)
#         # Log the images as wandb Image
#         aux_out = torch.sigmoid(aux_out)
        
#         columns = ['days/tr-1', 'days/r-2', 'days/tr-3', 'tg-days/tr']
#         my_data = [[f'{d1}-{tr1}', f'{d2}-{tr2}', f'{d3}-{tr3}', f'{td}-{ttr}'] for d1, tr1, d2, tr2, d3, tr3, td, ttr in 
#                    list(zip(intvs[0], treat_cond[0], 
#                             intvs[1], treat_cond[1], 
#                             intvs[2], treat_cond[2], 
#                             intvs[3], treat_cond[3]))]
#         # data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
         
#         trainer.logger.log_table(key='test_samples', columns = columns, data = my_data)
        
#         trainer.logger.log_image(key="label", 
#                                     images=[self.gt_preimg[0, -2, 3, :, :].cpu().detach().numpy(), 
#                                             self.gt_preimg[0, -1, 3, :, :].cpu().detach().numpy(),
#                                             aux_out[0, 3, :, :].cpu().detach().numpy(),
#                                             aux_out[0, 2, :, :].cpu().detach().numpy(),
#                                             ],
#                                     caption=[f'input:day{intvs[2][0]}-tr{treat_cond[2][0]}', f'target:day{intvs[3][0]}-tr{treat_cond[3][0]}', "pred-mask-tg", "pred-mask-s3"]) # f'Ground Truth: {y_i}
        
#         trainer.logger.log_image(key="Flair", 
#                                     images=[self.gt_preimg[3, -2, 2, :, :].cpu().detach().numpy(), 
#                                             self.gt_preimg[3, -1, 2, :, :].cpu().detach().numpy(), 
#                                             preds[3, 2, :, :].cpu().detach().numpy(),
#                                             aux_out[3, 3, :, :].cpu().detach().numpy()
#                                             ],
#                                     caption=[f'input:day{intvs[2][3]}-tr{treat_cond[2][3]}', f'target:day{intvs[3][3]}-tr{treat_cond[3][3]}', "pred-img", "pred-mask-tg"])
        
#         trainer.logger.log_image(key="T1c", 
#                                     images=[self.gt_preimg[1, -2, 1,  :, :].cpu().detach().numpy(), 
#                                             self.gt_preimg[1, -1, 1,  :, :].cpu().detach().numpy(),
#                                             preds[1, 1, :, :].cpu().detach().numpy(),
#                                             aux_out[1, 3, :, :].cpu().detach().numpy(),
#                                             ],
#                                     caption=[f'input:day{intvs[2][1]}-tr{treat_cond[2][1]}', f'target:day{intvs[3][1]}-tr{treat_cond[3][1]}', "pred-img", "pred-mask-tg"])
        
#         trainer.logger.log_image(key="T1", 
#                                     images=[self.gt_preimg[2, -2, 0,  :, :].cpu().detach().numpy(), 
#                                             self.gt_preimg[2, -1, 0,  :, :].cpu().detach().numpy(),
#                                             preds[2, 0, :, :].cpu().detach().numpy(),
#                                             aux_out[2, 3, :, :].cpu().detach().numpy(),
#                                             ],
#                                     caption=[f'input:day{intvs[2][2]}-tr{treat_cond[2][2]}', f'target:day{intvs[3][2]}-tr{treat_cond[3][2]}', "pred-img", "pred-mask-tg"])

#         pl_module.val_step_outputs.clear()
class MyCallback(Callback):
    def __init__(self, batch, config):
        super().__init__()
        self.batch = batch
        self.cfg = config
        
        # Debug: Print shapes to understand the data structure
        # print(f"\nMyCallback Debug - Input shapes:")
        # print(f"  image shape: {batch['image'].shape}")
        # print(f"  label shape: {batch['label'].shape}")
        # print(f"  days shape: {batch['days'].shape}")
        # print(f"  treatment shape: {batch['treatment'].shape}")
        
        # Get batch data
        images = batch['image']
        labels = batch['label']
        days = batch['days']
        treatments = batch['treatment']
        
        # Handle different possible shapes
        if images.dim() == 6:
            b, n_sessions, c, h, w, d = images.shape
        elif images.dim() == 5:
            b, cs, h, w, d = images.shape
            c = 4  # T1, T1c, FLAIR, T2 (you changed this to 4)
            n_sessions = cs // c
            images = images.view(b, n_sessions, c, h, w, d)
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
        
        # print(f"  Detected: {n_sessions} sessions with {c} modalities")
        
        # Handle label shapes
        if labels.dim() == 5:
            pass
        elif labels.dim() == 4:
            labels = labels.unsqueeze(1)
        elif labels.dim() == 6:
            labels = labels.squeeze(2)
        else:
            raise ValueError(f"Unexpected label shape: {labels.shape}")
        
        # Concatenate
        labels_with_channel = labels.unsqueeze(2)
        img_label = torch.cat([images, labels_with_channel], dim=2)
        
        # print(f"\nMyCallback - After processing:")
        # print(f"  img_label shape: {img_label.shape}")
        
        # Store processed data
        self.val_labels = img_label[:, :, -1, :, :, :]
        
        # Handle variable number of sessions
        if n_sessions >= 2:
            # Use all but last session for conditioning
            self.img_cond = img_label[:, :-1, :-1, :, :, :]
            self.img_for_noise = img_label[:, -1, :-1, :, :, :]
        else:
            # Only one session - use it for both
            self.img_cond = img_label[:, :, :-1, :, :, :]
            self.img_for_noise = img_label[:, 0, :-1, :, :, :]
        
        # Reshape img_cond for model input
        b, s, c_mod, h, w, d = self.img_cond.shape
        self.img_cond = self.img_cond.reshape(b, s*c_mod, h, w, d).contiguous()
        
        self.gt_preimg = img_label
        
        # Process days and treatments - HANDLE VARIABLE LENGTH
        if days.dim() == 1:
            days = days.unsqueeze(0)
        if treatments.dim() == 1:
            treatments = treatments.unsqueeze(0)
        
        # print(f"  days shape after expansion: {days.shape}")
        # print(f"  treatments shape after expansion: {treatments.shape}")
        
        # Get actual number of timepoints available
        n_timepoints = days.shape[1]
        # print(f"  Number of timepoints: {n_timepoints}")
        
        # Model expects 4 timepoints, so we need to pad/repeat if we have less
        if n_timepoints >= 4:
            # We have enough, just extract first 4
            self.intvs = [days[:, i].to(torch.float32) for i in range(4)]
            self.treat_cond = [treatments[:, i].to(torch.float32) for i in range(4)]
        else:
            # We have less than 4, need to pad by repeating last timepoint
            self.intvs = [days[:, i].to(torch.float32) for i in range(n_timepoints)]
            self.treat_cond = [treatments[:, i].to(torch.float32) for i in range(n_timepoints)]
            
            # Pad to 4 by repeating the last timepoint
            last_day = days[:, -1].to(torch.float32)
            last_treatment = treatments[:, -1].to(torch.float32)
            
            while len(self.intvs) < 4:
                # For future timepoints, extrapolate by adding a time delta
                time_delta = last_day - days[:, -2].to(torch.float32) if n_timepoints >= 2 else torch.tensor(30.0)
                next_day = self.intvs[-1] + time_delta
                self.intvs.append(next_day)
                self.treat_cond.append(last_treatment.clone())
        
        # print(f"  Final number of timepoints: {len(self.intvs)}")
        # print(f"  Days: {[d.item() if d.numel() == 1 else d[0].item() for d in self.intvs]}")
        # print(f"  Treatments: {[t.item() if t.numel() == 1 else t[0].item() for t in self.treat_cond]}")
        
        # Create noise
        noise = torch.randn((self.img_for_noise.shape))
        self.val_imgs = torch.cat([self.img_cond, noise], dim=1)
        
        # print(f"  val_imgs shape: {self.val_imgs.shape}")
        
        # Initialize diffusion
        self.diffusion = GaussianDiffusion(
            T=int(self.cfg.max_T), 
            schedule=self.cfg.ddpm_schedule
        )
        
        # print(f"MyCallback initialization complete!")    
# # trainer = Trainer(callbacks=[MyPrintingCallback()])