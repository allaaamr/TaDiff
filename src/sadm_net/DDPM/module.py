"""
DDPM/module.py - UNet Building Blocks for Diffusion Model

ORIGINAL ROLE (3D SADM):
- Contains the building blocks for a 3D UNet used in the diffusion process
- Includes ResBlocks, Attention blocks, Up/Downsampling with 3D convolutions
- All spatial operations are 3D (Conv3d, BatchNorm3d, etc.)
- Handles volumes of shape (B, C, H, W, D)

CHANGES FOR 2D ADAPTATION:
1. Conv3d → Conv2d throughout
2. BatchNorm3d → GroupNorm (more stable for small batches in medical imaging)
3. 3D pooling/upsampling → 2D pooling/upsampling
4. Removed depth dimension handling
5. Input shape: (B, C, H, W) instead of (B, C, H, W, D)
6. Added conditioning projection layers for SAT features

This module is based on TeaPearce's DDPM implementation, adapted for 2D medical images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: (B,) tensor of timestep indices
        embedding_dim: Dimension of the embedding
        
    Returns:
        embeddings: (B, embedding_dim) sinusoidal embeddings
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # zero pad if odd
        emb = F.pad(emb, (0, 1), mode='constant')
    
    return emb


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 computation for stability."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock2D(nn.Module):
    """
    2D Residual Block with time embedding conditioning.
    
    ORIGINAL (3D): Used Conv3d, processed (B, C, H, W, D) volumes
    CHANGED TO (2D): Uses Conv2d, processes (B, C, H, W) images
    
    The block consists of:
    1. GroupNorm → Swish → Conv2d
    2. Time embedding projection (added to features)
    3. GroupNorm → Swish → Dropout → Conv2d
    4. Skip connection (with optional channel projection)
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_emb_dim: Dimension of time embedding
        dropout: Dropout rate
        use_scale_shift_norm: If True, use FiLM-style conditioning
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # First conv block
        self.norm1 = GroupNorm32(32, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            Swish(),
            nn.Linear(
                time_emb_dim,
                out_channels * 2 if use_scale_shift_norm else out_channels
            ),
        )
        
        # Second conv block
        self.norm2 = GroupNorm32(32, out_channels)
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
            
        # Initialize last conv to zero for residual learning
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) input features
            time_emb: (B, time_emb_dim) time embedding
            
        Returns:
            out: (B, C_out, H, W) output features
        """
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Add time embedding
        time_emb = self.time_emb_proj(time_emb)
        
        if self.use_scale_shift_norm:
            scale, shift = time_emb.chunk(2, dim=1)
            h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = self.act2(h)
        else:
            h = h + time_emb[:, :, None, None]
            h = self.act2(self.norm2(h))
        
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


class AttentionBlock2D(nn.Module):
    """
    2D Self-Attention Block.
    
    ORIGINAL (3D): Attention over flattened 3D spatial dimensions
    CHANGED TO (2D): Attention over flattened 2D spatial dimensions (H*W)
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
        num_head_channels: Channels per head (if specified, overrides num_heads)
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
    ):
        super().__init__()
        self.channels = channels
        
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        
        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        
        # Initialize output projection to zero
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            out: (B, C, H, W) attention-refined features
        """
        B, C, H, W = x.shape
        
        # Normalize and reshape for attention
        h = self.norm(x)
        h = h.reshape(B, C, -1)  # (B, C, H*W)
        
        # Compute Q, K, V
        qkv = self.qkv(h)  # (B, 3*C, H*W)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, -1)
        k = k.reshape(B, self.num_heads, head_dim, -1)
        v = v.reshape(B, self.num_heads, head_dim, -1)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, -1)
        
        # Project output
        out = self.proj_out(out)
        out = out.reshape(B, C, H, W)
        
        return x + out


class CrossAttentionBlock2D(nn.Module):
    """
    2D Cross-Attention Block for conditioning on SAT features.
    
    NEW FOR SADM: This block allows the UNet to attend to the
    conditioning tensor from the Sequence-Aware Transformer.
    
    Args:
        channels: Number of query channels (UNet features)
        context_dim: Dimension of context features (SAT output)
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm_q = GroupNorm32(32, channels)
        self.norm_ctx = nn.LayerNorm(context_dim)
        
        self.to_q = nn.Conv1d(channels, channels, 1)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        
        # Initialize output to zero
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) UNet features
            context: (B, N_ctx, context_dim) SAT context tokens
            
        Returns:
            out: (B, C, H, W) cross-attention refined features
        """
        B, C, H, W = x.shape
        
        # Normalize and reshape query
        h = self.norm_q(x)
        h = h.reshape(B, C, -1)  # (B, C, H*W)
        q = self.to_q(h)  # (B, C, H*W)
        
        # Normalize context and compute K, V
        context = self.norm_ctx(context)  # (B, N_ctx, context_dim)
        k = self.to_k(context)  # (B, N_ctx, C)
        v = self.to_v(context)  # (B, N_ctx, C)
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, -1)  # (B, heads, head_dim, H*W)
        k = k.transpose(1, 2).reshape(B, self.num_heads, self.head_dim, -1)  # (B, heads, head_dim, N_ctx)
        v = v.transpose(1, 2).reshape(B, self.num_heads, self.head_dim, -1)  # (B, heads, head_dim, N_ctx)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale  # (B, heads, H*W, N_ctx)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)  # (B, heads, head_dim, H*W)
        out = out.reshape(B, C, -1)
        
        # Project output
        out = self.proj_out(out)
        out = out.reshape(B, C, H, W)
        
        return x + out


class Downsample2D(nn.Module):
    """
    2D Downsampling layer.
    
    ORIGINAL (3D): 3D strided convolution or pooling
    CHANGED TO (2D): 2D strided convolution
    """
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample2D(nn.Module):
    """
    2D Upsampling layer.
    
    ORIGINAL (3D): 3D interpolation + convolution
    CHANGED TO (2D): 2D interpolation + convolution
    """
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class ConditioningProjection(nn.Module):
    """
    Project SAT conditioning to match UNet channel dimensions.
    
    NEW FOR SADM: Projects the SAT output (B, cond_dim, H, W) to
    match the channel dimension at each UNet resolution level.
    
    Args:
        cond_dim: Dimension of SAT conditioning
        out_channels: Target channel dimension for this UNet level
    """
    
    def __init__(self, cond_dim: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cond_dim, out_channels, 1),
            GroupNorm32(32, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, cond: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Args:
            cond: (B, cond_dim, H_cond, W_cond) SAT conditioning
            target_size: (H_target, W_target) target spatial size
            
        Returns:
            proj_cond: (B, out_channels, H_target, W_target)
        """
        # Resize to target spatial dimensions
        if cond.shape[-2:] != target_size:
            cond = F.interpolate(cond, size=target_size, mode='bilinear', align_corners=False)
        
        return self.proj(cond)


if __name__ == "__main__":
    # Test modules
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 2, 64, 32, 32
    time_emb_dim = 256
    
    # Test ResBlock2D
    print("Testing ResBlock2D...")
    resblock = ResBlock2D(C, C * 2, time_emb_dim).to(device)
    x = torch.randn(B, C, H, W).to(device)
    t_emb = torch.randn(B, time_emb_dim).to(device)
    out = resblock(x, t_emb)
    print(f"  Input: {x.shape} → Output: {out.shape}")
    
    # Test AttentionBlock2D
    print("Testing AttentionBlock2D...")
    attn = AttentionBlock2D(C * 2).to(device)
    out_attn = attn(out)
    print(f"  Input: {out.shape} → Output: {out_attn.shape}")
    
    # Test CrossAttentionBlock2D
    print("Testing CrossAttentionBlock2D...")
    context_dim = 256
    cross_attn = CrossAttentionBlock2D(C * 2, context_dim).to(device)
    context = torch.randn(B, 64, context_dim).to(device)  # 64 context tokens
    out_cross = cross_attn(out, context)
    print(f"  Query: {out.shape}, Context: {context.shape} → Output: {out_cross.shape}")
    
    # Test Downsample2D
    print("Testing Downsample2D...")
    down = Downsample2D(C * 2).to(device)
    out_down = down(out)
    print(f"  Input: {out.shape} → Output: {out_down.shape}")
    
    # Test Upsample2D
    print("Testing Upsample2D...")
    up = Upsample2D(C * 2).to(device)
    out_up = up(out_down)
    print(f"  Input: {out_down.shape} → Output: {out_up.shape}")
    
    print("\nAll module tests passed!")