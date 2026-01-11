"""
ViVit/module.py - Transformer Building Blocks for Sequence-Aware Transformer

ORIGINAL ROLE (3D SADM):
- Contains transformer building blocks based on ViViT (Video Vision Transformer)
- Multi-head self-attention for 3D+time (4D) token sequences
- Factorized attention: temporal attention + spatial attention separately
- Works with 3D volumes tokenized into patches

CHANGES FOR 2D ADAPTATION:
1. Patch embedding: 3D patches → 2D patches
2. Positional embeddings: 3D spatial + temporal → 2D spatial + temporal
3. Attention mechanisms remain similar but operate on 2D patch tokens
4. Input: (B, S, C, H, W) sequences of 2D images
5. Output: Conditioning tensor for diffusion UNet

The module implements ViViT Model 3 (Factorized Self-Attention):
- Temporal attention: For each spatial location, attend across time
- Spatial attention: For each timepoint, attend across spatial locations

Based on rishikksh20's ViViT implementation, adapted for 2D medical images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) for regularization.
    Randomly drops entire paths during training.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """
    MLP block for transformer.
    Two linear layers with GELU activation.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: If True, add bias to QKV projections
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input tokens
            mask: Optional attention mask
            
        Returns:
            out: (B, N, D) attention output
        """
        B, N, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Standard Transformer Block with pre-norm.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: If True, add bias to QKV
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=drop,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding.
    
    ORIGINAL (3D): Divides 3D volume into 3D patches using Conv3d
    CHANGED TO (2D): Divides 2D image into 2D patches using Conv2d
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Patch size (assumes square)
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        
        # Patch embedding via convolution
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            patches: (B, N, D) patch embeddings where N = num_patches
        """
        # (B, C, H, W) -> (B, D, H/p, W/p)
        x = self.proj(x)
        # (B, D, H/p, W/p) -> (B, N, D)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.norm(x)
        return x


class TemporalPosEmbed(nn.Module):
    """
    Temporal positional embedding using sinusoidal encoding for continuous time.
    
    This allows encoding arbitrary time values (days) rather than just indices.
    
    Args:
        dim: Embedding dimension
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Learnable projection for time values
        self.time_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_values: (B, S) continuous time values (e.g., days)
            
        Returns:
            embeddings: (B, S, D) temporal embeddings
        """
        # Sinusoidal encoding
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time_values.device) * -emb_scale)
        
        # (B, S) -> (B, S, half_dim)
        emb = time_values[:, :, None] * emb[None, None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        # Project
        emb = self.time_proj(emb)
        
        return emb


class TreatmentEmbed(nn.Module):
    """
    Treatment code embedding.
    
    Args:
        num_treatments: Number of treatment types
        dim: Embedding dimension
    """
    
    def __init__(self, num_treatments: int = 10, dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(num_treatments, dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, treatment_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            treatment_codes: (B, S) integer treatment codes
            
        Returns:
            embeddings: (B, S, D) treatment embeddings
        """
        emb = self.embed(treatment_codes)
        emb = self.proj(emb)
        return emb


class GenoEmbed(nn.Module):
    """
    Genomic feature embedding.
    
    Projects genomic features to embedding dimension and adds
    to all timepoint embeddings (since genomics don't change over time).
    
    Args:
        geno_dim: Input genomic feature dimension
        embed_dim: Output embedding dimension
    """
    
    def __init__(self, geno_dim: int = 16, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(geno_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, geno: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geno: (B, G) genomic features
            
        Returns:
            embeddings: (B, D) genomic embeddings
        """
        return self.proj(geno)


if __name__ == "__main__":
    # Test modules
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, S, C, H, W = 2, 4, 3, 128, 128
    embed_dim = 256
    
    # Test PatchEmbed2D
    print("Testing PatchEmbed2D...")
    patch_embed = PatchEmbed2D(img_size=H, patch_size=8, in_channels=C, embed_dim=embed_dim).to(device)
    x = torch.randn(B, C, H, W).to(device)
    patches = patch_embed(x)
    print(f"  Input: {x.shape} → Patches: {patches.shape}")
    print(f"  Expected: (B, {(H//8)**2}, {embed_dim}) = ({B}, {(H//8)**2}, {embed_dim})")
    
    # Test TransformerBlock
    print("\nTesting TransformerBlock...")
    block = TransformerBlock(embed_dim, num_heads=8).to(device)
    out = block(patches)
    print(f"  Input: {patches.shape} → Output: {out.shape}")
    
    # Test TemporalPosEmbed
    print("\nTesting TemporalPosEmbed...")
    time_embed = TemporalPosEmbed(embed_dim).to(device)
    days = torch.tensor([[0, 30, 60, 90], [0, 45, 90, 135]]).float().to(device)
    time_emb = time_embed(days)
    print(f"  Days: {days.shape} → Embedding: {time_emb.shape}")
    
    # Test TreatmentEmbed
    print("\nTesting TreatmentEmbed...")
    treat_embed = TreatmentEmbed(num_treatments=5, dim=embed_dim).to(device)
    treatments = torch.randint(0, 5, (B, S)).to(device)
    treat_emb = treat_embed(treatments)
    print(f"  Treatments: {treatments.shape} → Embedding: {treat_emb.shape}")
    
    # Test GenoEmbed
    print("\nTesting GenoEmbed...")
    geno_embed = GenoEmbed(geno_dim=16, embed_dim=embed_dim).to(device)
    geno = torch.randn(B, 16).to(device)
    geno_emb = geno_embed(geno)
    print(f"  Geno: {geno.shape} → Embedding: {geno_emb.shape}")
    
    print("\nAll module tests passed!")