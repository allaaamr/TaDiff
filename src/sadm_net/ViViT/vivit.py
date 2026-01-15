"""
ViVit/vivit.py - Sequence-Aware Transformer for Longitudinal 2D Medical Images

ORIGINAL ROLE (3D SADM):
- ViViT-based transformer for processing 3D medical image sequences
- Input: Longitudinal 3D volumes (B, L, C, H, W, D) with L timepoints
- Tokenizes each 3D volume into patches
- Temporal encoder: Self-attention across timepoints for each spatial location
- Spatial decoder: Self-attention across spatial locations for each timepoint
- Output: 3D conditioning tensor for the diffusion UNet

CHANGES FOR 2D ADAPTATION:
1. Input: 2D image sequences (B, S, C, H, W) with S sessions/timepoints
2. Patch embedding: 2D patches instead of 3D patches
3. Spatial operations: 2D (H × W) instead of 3D (H × W × D)
4. Output: 2D conditioning tensor (B, cond_dim, H, W)
5. Added support for:
   - Continuous time values (days) via sinusoidal embeddings
   - Treatment codes via learned embeddings
   - Genomic features via projection layers
   - Segmentation labels concatenated with images

Architecture (ViViT Model 3 - Factorized Self-Attention):
1. Patch Embedding: Each 2D image → patch tokens
2. Positional Encoding: Spatial + temporal positional embeddings
3. Temporal Encoder: For each spatial position, attend across all timepoints
4. Spatial Decoder: For each timepoint, attend across all spatial positions
5. Output Head: Aggregate and project to conditioning tensor

Based on rishikksh20's ViViT, modified for SADM's conditioning role.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from einops import rearrange, repeat

from .module import (
    TransformerBlock,
    PatchEmbed2D,
    TemporalPosEmbed,
    TreatmentEmbed,
    GenoEmbed,
    DropPath,
)


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Transformer Encoder.
    
    For each spatial location, performs self-attention across all timepoints.
    This captures how each spatial region evolves over time.
    
    ORIGINAL (3D): Attention across L timepoints for each of (H/p × W/p × D/p) locations
    CHANGED TO (2D): Attention across S timepoints for each of (H/p × W/p) locations
    
    Args:
        dim: Token dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    
    def __init__(
        self,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, N, D) tokens - S timepoints, N spatial patches
            temporal_mask: (B, S) mask for valid timepoints
            
        Returns:
            out: (B, S, N, D) temporally encoded tokens
        """
        B, S, N, D = x.shape
        
        # Reshape for temporal attention: group by spatial location
        # (B, S, N, D) -> (B*N, S, D)
        x = rearrange(x, 'b s n d -> (b n) s d')
        
        # Expand mask for all spatial locations
        if temporal_mask is not None:
            temporal_mask = repeat(temporal_mask, 'b s -> (b n) s', n=N)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, temporal_mask)
        
        x = self.norm(x)
        
        # Reshape back: (B*N, S, D) -> (B, S, N, D)
        x = rearrange(x, '(b n) s d -> b s n d', b=B, n=N)
        
        return x


class SpatialTransformerDecoder(nn.Module):
    """
    Spatial Transformer Decoder.
    
    For each timepoint, performs self-attention across all spatial locations.
    This captures spatial relationships at each time.
    
    ORIGINAL (3D): Attention across (H/p × W/p × D/p) locations for each timepoint
    CHANGED TO (2D): Attention across (H/p × W/p) locations for each timepoint
    
    Args:
        dim: Token dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    
    def __init__(
        self,
        dim: int = 240,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, N, D) tokens - S timepoints, N spatial patches
            
        Returns:
            out: (B, S, N, D) spatially decoded tokens
        """
        B, S, N, D = x.shape
        
        # Reshape for spatial attention: group by timepoint
        # (B, S, N, D) -> (B*S, N, D)
        x = rearrange(x, 'b s n d -> (b s) n d')
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back: (B*S, N, D) -> (B, S, N, D)
        x = rearrange(x, '(b s) n d -> b s n d', b=B, s=S)
        
        return x


class SequenceAwareViT2D(nn.Module):
    """
    Sequence-Aware Vision Transformer for 2D Longitudinal Medical Images.
    
    This is the core module of SADM that processes longitudinal image sequences
    and produces conditioning signals for the diffusion model.
    
    ORIGINAL (3D SADM):
    - Input: 3D volumes over time (B, L, C, H, W, D)
    - 3D patch embedding
    - 4D positional encoding (spatial + temporal)
    - Output: 3D conditioning tensor
    
    CHANGES FOR 2D:
    - Input: 2D images over time (B, S, C, H, W)
    - 2D patch embedding
    - 2D spatial + temporal positional encoding
    - Output: 2D conditioning tensor (B, cond_dim, H, W)
    - Added: days, treatments, genomics conditioning
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Patch size for tokenization
        in_channels: Number of input channels (e.g., 3 for T1, T1c, FLAIR)
        embed_dim: Transformer embedding dimension
        temporal_depth: Number of temporal transformer layers
        spatial_depth: Number of spatial transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        max_seq_len: Maximum sequence length (number of timepoints)
        dropout: Dropout rate
        use_labels: Whether to concatenate segmentation labels to input
        use_geno: Whether to use genomic features
        geno_dim: Dimension of genomic features
        num_treatments: Number of treatment types
    """
    
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
        temporal_depth: int = 4,
        spatial_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 4,
        dropout: float = 0.1,
        use_labels: bool = True,
        use_geno: bool = True,
        geno_dim: int = 16,
        num_treatments: int = 10,
        drop_path: float = 0.1,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_labels = use_labels
        self.use_geno = use_geno
        
        # Number of patches per image
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        
        # Patch embedding
        # If using labels, input has one extra channel
        actual_in_channels = in_channels + 1 if use_labels else in_channels
        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=actual_in_channels,
            embed_dim=embed_dim,
        )
        
        # Positional embeddings
        # Learnable spatial positional embedding
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        
        # Learnable temporal positional embedding (for discrete indices)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        # Continuous time embedding (for days)
        self.time_embed = TemporalPosEmbed(embed_dim)
        
        # Treatment embedding
        self.treatment_embed = TreatmentEmbed(num_treatments, embed_dim)
        
        # Genomic embedding
        if use_geno:
            self.geno_embed = GenoEmbed(geno_dim, embed_dim)
        else:
            self.geno_embed = None
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Temporal Encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            dim=embed_dim,
            depth=temporal_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout,
            attn_drop=dropout,
            drop_path=drop_path,
        )
        
        # Spatial Decoder
        self.spatial_decoder = SpatialTransformerDecoder(
            dim=embed_dim,
            depth=spatial_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout,
            attn_drop=dropout,
            drop_path=drop_path,
        )
        
        # Output head: aggregate temporal information and project to spatial conditioning
        # We use cross-attention to query target timepoint information
        self.target_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.target_query, std=0.02)
        
        self.target_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Final projection to spatial conditioning map
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * (patch_size ** 2)),
        )
        
        # Spatial refinement
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        days: Optional[torch.Tensor] = None,
        treatments: Optional[torch.Tensor] = None,
        geno: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Sequence-Aware Transformer.
        
        Args:
            images: (B, S, C, H, W) longitudinal image sequence
            labels: (B, S, H, W) segmentation labels (optional, used if use_labels=True)
            days: (B, S) days since baseline for each timepoint
            treatments: (B, S) treatment codes for each timepoint
            geno: (B, G) genomic features (time-invariant)
            temporal_mask: (B, S) mask for valid timepoints (1=valid, 0=missing)
            
        Returns:
            cond_spatial: (B, embed_dim, H, W) spatial conditioning map
            cond_tokens: (B, N, embed_dim) conditioning tokens for cross-attention
        """
        B, S, C, H, W = images.shape
        device = images.device
        
        # Validate sequence length
        assert S <= self.max_seq_len, f"Sequence length {S} exceeds max {self.max_seq_len}"
        
        # Concatenate labels if provided
        if self.use_labels and labels is not None:
            labels = labels.unsqueeze(2).float()  # (B, S, 1, H, W)
            images = torch.cat([images, labels], dim=2)  # (B, S, C+1, H, W)
        
        # Create mask if not provided
        if temporal_mask is None:
            temporal_mask = torch.ones(B, S, device=device)
        
        # ============ PATCH EMBEDDING ============
        # Process each timepoint's image into patches
        # (B, S, C, H, W) -> (B*S, C, H, W) -> (B*S, N, D) -> (B, S, N, D)
        images_flat = rearrange(images, 'b s c h w -> (b s) c h w')
        patches = self.patch_embed(images_flat)  # (B*S, N, D)
        patches = rearrange(patches, '(b s) n d -> b s n d', b=B, s=S)
        
        # ============ POSITIONAL ENCODING ============
        # Add spatial positional embedding

        # print("H,W:", H, W, "patch_size:", self.patch_size)
        # print("patches N:", patches.shape[2], "spatial_pos_embed N:", self.spatial_pos_embed.shape[1])

        patches = patches + self.spatial_pos_embed.unsqueeze(1)  # broadcast over S
        
        # Add temporal positional embedding (learnable, for discrete indices)
        temporal_pos = self.temporal_pos_embed[:, :S]  # (1, S, 1, D)
        patches = patches + temporal_pos
        
        # Add continuous time embedding (for days)
        if days is not None:
            time_emb = self.time_embed(days)  # (B, S, D)
            patches = patches + time_emb.unsqueeze(2)  # (B, S, 1, D) broadcast over N
        
        # Add treatment embedding
        if treatments is not None:
            treat_emb = self.treatment_embed(treatments)  # (B, S, D)
            patches = patches + treat_emb.unsqueeze(2)  # broadcast over N
        
        # Add genomic embedding (time-invariant, added to all timepoints)
        if self.use_geno and self.geno_embed is not None and geno is not None:
            geno_emb = self.geno_embed(geno)  # (B, D)
            patches = patches + geno_emb.unsqueeze(1).unsqueeze(2)  # broadcast over S and N
        
        # Dropout
        patches = self.pos_drop(patches)
        
        # ============ TEMPORAL ENCODING ============
        # For each spatial location, attend across time
        patches = self.temporal_encoder(patches, temporal_mask)  # (B, S, N, D)
        
        # ============ SPATIAL DECODING ============
        # For each timepoint, attend across spatial locations
        patches = self.spatial_decoder(patches)  # (B, S, N, D)
        
        # ============ OUTPUT HEAD ============
        # Option 1: Use last timepoint as conditioning
        # target_patches = patches[:, -1]  # (B, N, D)
        
        # Option 2: Cross-attend to aggregate temporal information
        # Flatten all patches for key/value: (B, S*N, D)
        all_patches = rearrange(patches, 'b s n d -> b (s n) d')
        
        # Query for each spatial location
        target_query = repeat(self.target_query, '1 1 d -> b n d', b=B, n=self.num_patches)
        
        # Cross-attention
        attended, _ = self.target_cross_attn(
            query=target_query,
            key=all_patches,
            value=all_patches,
        )  # (B, N, D)
        
        # Combine with last timepoint
        combined = attended + patches[:, -1]  # (B, N, D)
        
        # Save tokens for cross-attention in UNet
        cond_tokens = combined  # (B, N, D)
        
        # Project to spatial map
        # (B, N, D) -> (B, N, D * p^2)
        out = self.output_head(combined)
        
        # Reshape to spatial map
        # (B, N, D * p^2) -> (B, D, H, W)
        cond_spatial = rearrange(
            out,
            'b (h w) (d p1 p2) -> b d (h p1) (w p2)',
            h=self.num_patches_per_dim,
            w=self.num_patches_per_dim,
            p1=self.patch_size,
            p2=self.patch_size,
            d=self.embed_dim,
        )
        
        # Spatial refinement
        cond_spatial = cond_spatial + self.refine(cond_spatial)
        
        return cond_spatial, cond_tokens
    
    def get_conditioning(
        self,
        batch: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to extract conditioning from a data batch.
        
        Args:
            batch: Dictionary with 'image', 'label', 'days', 'treatment', 'geno'
            
        Returns:
            cond_global: (B, embed_dim) global conditioning (pooled)
            cond_spatial: (B, embed_dim, H, W) spatial conditioning
            cond_tokens: (B, N, embed_dim) conditioning tokens
        """
        images = batch['image']  # (B, S, C, H, W)
        labels = batch.get('label', None)
        days = batch.get('days', None)
        treatments = batch.get('treatment', batch.get('treatments', None))
        geno = batch.get('geno', None)
        
        cond_spatial, cond_tokens = self.forward(
            images=images,
            labels=labels,
            days=days,
            treatments=treatments,
            geno=geno,
        )
        
        # Global conditioning via spatial pooling
        cond_global = F.adaptive_avg_pool2d(cond_spatial, 1).flatten(1)  # (B, D)
        
        return cond_global, cond_spatial, cond_tokens


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    print("Creating SequenceAwareViT2D...")
    model = SequenceAwareViT2D(
        img_size=128,
        patch_size=8,
        in_channels=3,
        embed_dim=256,
        temporal_depth=4,
        spatial_depth=4,
        num_heads=8,
        max_seq_len=4,
        use_labels=True,
        use_geno=True,
        geno_dim=16,
    ).to(device)
    
    # Create test data matching your data loader
    B, S, C, H, W = 2, 4, 3, 128, 128
    G = 16
    
    batch = {
        'image': torch.randn(B, S, C, H, W).to(device),
        'label': torch.randint(0, 2, (B, S, H, W)).float().to(device),
        'days': torch.tensor([[0, 30, 60, 90], [0, 45, 90, 135]]).float().to(device),
        'treatment': torch.randint(0, 5, (B, S)).to(device),
        'geno': torch.randn(B, G).to(device),
    }
    
    # Test forward pass
    print("\nTesting forward pass...")
    cond_spatial, cond_tokens = model(
        images=batch['image'],
        labels=batch['label'],
        days=batch['days'],
        treatments=batch['treatment'],
        geno=batch['geno'],
    )
    
    print(f"  Input images: {batch['image'].shape}")
    print(f"  Spatial conditioning: {cond_spatial.shape}")
    print(f"  Conditioning tokens: {cond_tokens.shape}")
    
    # Test convenience method
    print("\nTesting get_conditioning...")
    cond_global, cond_spatial, cond_tokens = model.get_conditioning(batch)
    print(f"  Global conditioning: {cond_global.shape}")
    print(f"  Spatial conditioning: {cond_spatial.shape}")
    print(f"  Conditioning tokens: {cond_tokens.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    print("\nAll tests passed!")