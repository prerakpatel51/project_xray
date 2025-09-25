"""
Vision Mamba model for multi-label X-ray classification
Based on the Vision Mamba paper: https://arxiv.org/abs/2401.09417
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, List

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with flexible patch sizes"""

    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # Project patches
        x = self.proj(x)  # B, embed_dim, grid_H, grid_W
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class VisionMambaBlock(nn.Module):
    """Core Vision Mamba Block with bidirectional scanning"""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback to standard attention if Mamba not available
            self.mamba = nn.MultiheadAttention(dim, num_heads=8, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        if MAMBA_AVAILABLE:
            # Mamba forward pass
            x = self.mamba(x)
        else:
            # Fallback attention
            x, _ = self.mamba(x, x, x)

        x = self.dropout(x)
        return x + residual


class BidirectionalMamba(nn.Module):
    """Bidirectional scanning for 2D vision data"""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = VisionMambaBlock(dim, d_state, d_conv, expand)
        self.backward_mamba = VisionMambaBlock(dim, d_state, d_conv, expand)
        self.merge = nn.Linear(dim * 2, dim)

    def forward(self, x):
        B, N, D = x.shape

        # Forward scan (top-left to bottom-right)
        x_forward = self.forward_mamba(x)

        # Backward scan (bottom-right to top-left)
        x_backward = torch.flip(x, dims=[1])
        x_backward = self.backward_mamba(x_backward)
        x_backward = torch.flip(x_backward, dims=[1])

        # Merge both directions
        x_merged = torch.cat([x_forward, x_backward], dim=-1)
        x_merged = self.merge(x_merged)

        return x_merged


class VisionMamba(nn.Module):
    """Vision Mamba model for multi-label chest X-ray classification"""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 15,  # 15 medical conditions
        embed_dim: int = 768,
        depth: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_bidirectional: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_bidirectional = use_bidirectional

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Mamba blocks
        if use_bidirectional:
            self.blocks = nn.ModuleList([
                BidirectionalMamba(embed_dim, d_state, d_conv, expand)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                VisionMambaBlock(embed_dim, d_state, d_conv, expand, dropout)
                for _ in range(depth)
            ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize classifier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Forward pass through feature extraction"""
        # Patch embedding
        x = self.patch_embed(x)  # B, N, D

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        return x

    def forward(self, x):
        """Full forward pass"""
        # Feature extraction
        x = self.forward_features(x)  # B, N, D

        # Global pooling
        x = x.transpose(1, 2)  # B, D, N
        x = self.global_pool(x)  # B, D, 1
        x = x.flatten(1)  # B, D

        # Classification
        logits = self.classifier(x)  # B, num_classes

        return logits

    def get_attention_maps(self, x):
        """Generate attention/importance maps for visualization"""
        # This is a placeholder - actual implementation would depend on
        # how we want to visualize Mamba's selective attention
        x = self.forward_features(x)

        # Return spatial attention weights (simplified)
        batch_size = x.shape[0]
        grid_size = int(math.sqrt(x.shape[1]))
        attention_maps = x.mean(dim=-1).view(batch_size, grid_size, grid_size)

        return attention_maps


def create_vision_mamba_model(
    num_classes: int = 15,
    img_size: int = 512,
    model_size: str = "base"
) -> VisionMamba:
    """Factory function to create Vision Mamba models of different sizes"""

    if model_size == "tiny":
        config = {
            "embed_dim": 384,
            "depth": 6,
            "patch_size": 16,
        }
    elif model_size == "small":
        config = {
            "embed_dim": 576,
            "depth": 8,
            "patch_size": 16,
        }
    elif model_size == "base":
        config = {
            "embed_dim": 768,
            "depth": 12,
            "patch_size": 16,
        }
    elif model_size == "large":
        config = {
            "embed_dim": 1024,
            "depth": 16,
            "patch_size": 16,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    model = VisionMamba(
        img_size=img_size,
        num_classes=num_classes,
        in_chans=1,  # Grayscale X-rays
        **config
    )

    return model


# Medical condition class names for reference
MEDICAL_CONDITIONS = [
    'No Finding',
    'Infiltration',
    'Effusion',
    'Atelectasis',
    'Nodule',
    'Mass',
    'Pneumothorax',
    'Consolidation',
    'Pleural_Thickening',
    'Cardiomegaly',
    'Emphysema',
    'Edema',
    'Fibrosis',
    'Pneumonia',
    'Hernia'
]


if __name__ == "__main__":
    # Test the model
    print("Testing Vision Mamba model...")

    # Create model
    model = create_vision_mamba_model(
        num_classes=15,
        img_size=512,
        model_size="base"
    )

    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 512, 512)  # Batch of 2 grayscale X-rays

    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")  # Should be [2, 15]
        print(f"Output sample: {output[0][:5]}")  # First 5 logits

    print("\\nModel test completed successfully!")