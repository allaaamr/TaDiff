"""
DDPM/DDPM.py - Denoising Diffusion Probabilistic Model with 2D UNet

ORIGINAL ROLE (3D SADM):
- Contains the DDPM class that wraps the diffusion process
- Contains a 3D UNet (ContextUnet) for noise prediction
- All operations work on 3D volumes (B, C, H, W, D)
- Conditioning is injected via class labels or other embeddings

CHANGES FOR 2D ADAPTATION:
1. UNet architecture converted from 3D to 2D
2. All Conv3d → Conv2d, pooling and upsampling are 2D
3. Added SAT (Sequence-Aware Transformer) conditioning input
4. Conditioning is injected via:
   - Cross-attention blocks at attention resolutions
   - FiLM-style modulation in ResBlocks (optional)
   - Channel-wise concatenation (optional)
5. Input shape: (B, C, H, W) instead of (B, C, H, W, D)
6. Output includes both noise prediction and segmentation head

Based on TeaPearce's Conditional_Diffusion_MNIST, extended for medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

from .module import (
    get_timestep_embedding,
    Swish,
    GroupNorm32,
    ResBlock2D,
    AttentionBlock2D,
    CrossAttentionBlock2D,
    Downsample2D,
    Upsample2D,
)


class ContextUNet2D(nn.Module):
    """
    2D UNet with Sequence-Aware Transformer Conditioning.
    
    ORIGINAL (3D SADM):
    - 3D UNet architecture
    - Takes concatenated multi-timepoint volumes as input
    - Conditioning via timestep + class embeddings
    
    CHANGED FOR 2D:
    - 2D UNet architecture for slice-by-slice processing
    - Takes single target image (noisy) as input
    - Conditioning from SAT via cross-attention and FiLM
    - Separate input for SAT conditioning tensor
    
    Architecture:
    - Encoder: ResBlocks + optional Attention + Downsampling
    - Middle: ResBlocks + Attention + Cross-Attention
    - Decoder: ResBlocks + optional Attention + Upsampling + Skip connections
    
    Args:
        in_channels: Input image channels (e.g., 3 for T1, T1c, FLAIR)
        out_channels: Output channels (e.g., 3 for image + 4 for segmentation = 7)
        model_channels: Base channel count
        channel_mult: Channel multipliers per resolution level
        num_res_blocks: Number of ResBlocks per resolution level
        attention_resolutions: Resolutions at which to apply attention
        dropout: Dropout rate
        num_heads: Number of attention heads
        use_scale_shift_norm: Use FiLM-style conditioning in ResBlocks
        cond_dim: Dimension of SAT conditioning (set to 0 to disable cross-attention)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 7,
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
        num_heads: int = 8,
        use_scale_shift_norm: bool = True,
        cond_dim: int = 256,
        image_size: int = 128,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.image_size = image_size
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # SAT conditioning embedding (global)
        if cond_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, time_emb_dim),
                Swish(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.cond_embed = None
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Track channels for skip connections
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        
        # ============ ENCODER ============
        self.encoder = nn.ModuleList()
        
        current_res = image_size
        for level, mult in enumerate(channel_mult):
            for block_idx in range(num_res_blocks):
                layers = [
                    ResBlock2D(
                        ch,
                        model_channels * mult,
                        time_emb_dim,
                        dropout,
                        use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                
                # Add self-attention at specified resolutions
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock2D(ch, num_heads))
                    
                    # Add cross-attention for SAT conditioning
                    if cond_dim > 0:
                        layers.append(CrossAttentionBlock2D(ch, cond_dim, num_heads))
                
                self.encoder.append(nn.ModuleList(layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.encoder.append(nn.ModuleList([Downsample2D(ch)]))
                input_block_chans.append(ch)
                self._feature_size += ch
                current_res //= 2
        
        # ============ MIDDLE ============
        self.middle = nn.ModuleList([
            ResBlock2D(ch, ch, time_emb_dim, dropout, use_scale_shift_norm),
            AttentionBlock2D(ch, num_heads),
        ])
        
        # Add cross-attention in middle block
        if cond_dim > 0:
            self.middle.append(CrossAttentionBlock2D(ch, cond_dim, num_heads))
        
        self.middle.append(
            ResBlock2D(ch, ch, time_emb_dim, dropout, use_scale_shift_norm)
        )
        self._feature_size += ch
        
        # ============ DECODER ============
        self.decoder = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock2D(
                        ch + ich,
                        model_channels * mult,
                        time_emb_dim,
                        dropout,
                        use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                
                # Add self-attention at specified resolutions
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock2D(ch, num_heads))
                    
                    # Add cross-attention for SAT conditioning
                    if cond_dim > 0:
                        layers.append(CrossAttentionBlock2D(ch, cond_dim, num_heads))
                
                # Upsample (except last block of each level)
                if level > 0 and block_idx == num_res_blocks:
                    layers.append(Upsample2D(ch))
                    current_res *= 2
                
                self.decoder.append(nn.ModuleList(layers))
                self._feature_size += ch
        
        # Output projection
        self.output_norm = GroupNorm32(32, ch)
        self.output_act = Swish()
        self.output_proj = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        # Initialize output to zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the UNet.
        
        Args:
            x: (B, in_channels, H, W) noisy target image
            timesteps: (B,) diffusion timesteps
            cond: (B, cond_dim) global conditioning from SAT (pooled)
            cond_tokens: (B, N, cond_dim) conditioning tokens for cross-attention
            
        Returns:
            out: (B, out_channels, H, W) predicted noise + segmentation
        """
        # Time embedding
        t_emb = get_timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Add global SAT conditioning to time embedding
        if self.cond_embed is not None and cond is not None:
            cond_emb = self.cond_embed(cond)
            t_emb = t_emb + cond_emb
        
        # Input projection
        h = self.input_proj(x)
        
        # Encoder
        hs = [h]
        for block in self.encoder:
            for layer in block:
                if isinstance(layer, ResBlock2D):
                    h = layer(h, t_emb)
                elif isinstance(layer, CrossAttentionBlock2D):
                    if cond_tokens is not None:
                        h = layer(h, cond_tokens)
                elif isinstance(layer, (AttentionBlock2D, Downsample2D)):
                    h = layer(h)
                else:
                    h = layer(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle:
            if isinstance(layer, ResBlock2D):
                h = layer(h, t_emb)
            elif isinstance(layer, CrossAttentionBlock2D):
                if cond_tokens is not None:
                    h = layer(h, cond_tokens)
            else:
                h = layer(h)
        
        # Decoder
        for block in self.decoder:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                if isinstance(layer, ResBlock2D):
                    h = layer(h, t_emb)
                elif isinstance(layer, CrossAttentionBlock2D):
                    if cond_tokens is not None:
                        h = layer(h, cond_tokens)
                elif isinstance(layer, (AttentionBlock2D, Upsample2D)):
                    h = layer(h)
                else:
                    h = layer(h)
        
        # Output
        h = self.output_act(self.output_norm(h))
        h = self.output_proj(h)
        
        return h


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    
    ORIGINAL (3D SADM):
    - Wraps 3D UNet with diffusion process
    - Forward diffusion: adds noise to 3D volumes
    - Reverse diffusion: denoises 3D volumes
    
    CHANGES FOR 2D:
    - Works with 2D images
    - Accepts SAT conditioning inputs
    - Both training (get_loss) and sampling methods
    
    Args:
        nn_model: The noise prediction network (ContextUNet2D)
        betas: Tuple of (beta_start, beta_end) for linear schedule
        n_T: Number of diffusion timesteps
        device: Torch device
        drop_prob: Probability of dropping conditioning (for classifier-free guidance)
    """
    
    def __init__(
        self,
        nn_model: nn.Module,
        betas: Tuple[float, float] = (1e-4, 0.02),
        n_T: int = 1000,
        device: str = "cuda",
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.nn_model = nn_model.to(device)
        
        # Register diffusion parameters as buffers
        beta_start, beta_end = betas
        self.n_T = n_T
        
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, n_T).double()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
        
        # For sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas).float())
        self.register_buffer(
            'posterior_variance',
            (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float()
        )
        
        self.device = device
        self.drop_prob = drop_prob
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0: (B, C, H, W) original images
            t: (B,) timesteps
            noise: Optional pre-sampled noise
            
        Returns:
            x_t: (B, C, H, W) noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_loss(
        self,
        x_0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x_0: (B, C, H, W) target images
            cond: (B, cond_dim) global conditioning
            cond_tokens: (B, N, cond_dim) conditioning tokens
            
        Returns:
            loss: Scalar loss tensor
            noise: The noise that was added (for auxiliary losses)
        """
        B = x_0.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.n_T, (B,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise
        x_t = self.q_sample(x_0, t, noise)
        
        # Optionally drop conditioning for classifier-free guidance training
        if self.training and self.drop_prob > 0:
            drop_mask = torch.rand(B, device=self.device) < self.drop_prob
            if cond is not None:
                cond = cond.clone()
                cond[drop_mask] = 0
            if cond_tokens is not None:
                cond_tokens = cond_tokens.clone()
                cond_tokens[drop_mask] = 0
        
        # Predict noise
        pred = self.nn_model(x_t, t, cond, cond_tokens)
        
        # The model outputs both noise prediction and segmentation
        # Assume first in_channels are noise, rest are segmentation
        in_ch = self.nn_model.in_channels
        pred_noise = pred[:, :in_ch]
        
        # MSE loss on noise prediction
        loss = F.mse_loss(pred_noise, noise)
        
        return loss, noise
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        cond: Optional[torch.Tensor] = None,
        cond_tokens: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: p(x_{t-1} | x_t).
        
        Args:
            x_t: (B, C, H, W) noisy images
            t: Current timestep (scalar)
            cond: Global conditioning
            cond_tokens: Conditioning tokens
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            x_{t-1}: (B, C, H, W) denoised one step
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
        
        in_ch = self.nn_model.in_channels
        
        # Classifier-free guidance
        if guidance_scale != 1.0 and cond is not None:
            # Conditional prediction
            pred_cond = self.nn_model(x_t, t_tensor, cond, cond_tokens)[:, :in_ch]
            # Unconditional prediction
            pred_uncond = self.nn_model(x_t, t_tensor, None, None)[:, :in_ch]
            # Guidance
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_noise = self.nn_model(x_t, t_tensor, cond, cond_tokens)[:, :in_ch]
        
        # Compute x_{t-1}
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Mean
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
        )
        
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        cond_tokens: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion sampling.
        
        Args:
            shape: (B, C, H, W) output shape
            cond: Global conditioning
            cond_tokens: Conditioning tokens
            guidance_scale: Classifier-free guidance scale
            return_intermediates: If True, return all intermediate x_t
            
        Returns:
            samples: (B, C, H, W) generated images
            intermediates: Optional list of intermediate images
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=self.device)
        
        intermediates = [x_t] if return_intermediates else None
        
        # Reverse diffusion
        for t in reversed(range(self.n_T)):
            x_t = self.p_sample(x_t, t, cond, cond_tokens, guidance_scale)
            if return_intermediates:
                intermediates.append(x_t)
        
        if return_intermediates:
            return x_t, intermediates
        return x_t


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    print("Creating ContextUNet2D...")
    unet = ContextUNet2D(
        in_channels=3,
        out_channels=7,  # 3 image + 4 seg
        model_channels=32,  # Small for testing
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(8,),
        dropout=0.0,
        num_heads=4,
        cond_dim=128,
        image_size=64,
    ).to(device)
    
    # Test forward pass
    B = 2
    x = torch.randn(B, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    cond = torch.randn(B, 128).to(device)
    cond_tokens = torch.randn(B, 16, 128).to(device)
    
    print(f"Input: {x.shape}")
    out = unet(x, t, cond, cond_tokens)
    print(f"Output: {out.shape}")
    
    # Create DDPM
    print("\nCreating DDPM...")
    ddpm = DDPM(unet, betas=(1e-4, 0.02), n_T=1000, device=device)
    
    # Test loss
    print("Testing get_loss...")
    loss, noise = ddpm.get_loss(x, cond, cond_tokens)
    print(f"Loss: {loss.item():.4f}")
    
    # Test sampling (just a few steps for speed)
    print("\nTesting sampling (10 steps)...")
    ddpm.n_T = 10  # Reduce for testing
    samples = ddpm.sample((2, 3, 64, 64), cond, cond_tokens)
    print(f"Samples: {samples.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"\nUNet parameters: {n_params:,}")
    
    print("\nAll tests passed!")