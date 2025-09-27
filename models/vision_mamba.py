"""
Vision Transformer backbone with strategically placed Mamba blocks
Combines proven ViT performance with selective Mamba enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional, List

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=1024, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, grid_H, grid_W
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class MambaBlock(nn.Module):
    """Mamba block for selective state space modeling"""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
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
            # Fallback to attention if Mamba not available
            self.mamba = nn.MultiheadAttention(dim, num_heads=8, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        if MAMBA_AVAILABLE:
            x = self.mamba(x)
        else:
            x, _ = self.mamba(x, x, x)

        x = self.dropout(x)
        return x + residual


class TransformerBlock(nn.Module):
    """Standard Transformer block with multi-head attention"""

    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class ViTMambaHybrid(nn.Module):
    """Vision Transformer with strategically placed Mamba blocks"""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 1,
        num_classes: int = 15,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        mamba_layers: List[int] = [3, 6, 9],  # Which layers to use Mamba
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.mamba_layers = set(mamba_layers)

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

        # Build hybrid layers
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in self.mamba_layers:
                # Use Mamba block at specified layers
                block = MambaBlock(
                    dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout
                )
            else:
                # Use Transformer block otherwise
                block = TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
            self.blocks.append(block)

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Forward pass through feature extraction layers"""
        # Patch embedding
        x = self.patch_embed(x)  # B, N, D

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply blocks (mix of Transformer and Mamba)
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x

    def forward(self, x):
        """Full forward pass"""
        # Feature extraction
        x = self.forward_features(x)  # B, N, D

        # Global average pooling
        x = x.mean(dim=1)  # B, D

        # Classification
        logits = self.head(x)  # B, num_classes

        return logits


def create_vit_mamba_model(
    num_classes: int = 15,
    img_size: int = 1024,
    model_size: str = "base",
    mamba_layers: List[int] = [3, 6, 9]
) -> ViTMambaHybrid:
    """Factory function to create ViT-Mamba hybrid models"""

    if model_size == "tiny":
        config = {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "patch_size": 16,
        }
    elif model_size == "small":
        config = {
            "embed_dim": 576,
            "depth": 12,
            "num_heads": 9,
            "patch_size": 16,
        }
    elif model_size == "base":
        config = {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "patch_size": 16,
        }
    elif model_size == "large":
        config = {
            "embed_dim": 1024,
            "depth": 16,
            "num_heads": 16,
            "patch_size": 16,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    model = ViTMambaHybrid(
        img_size=img_size,
        num_classes=num_classes,
        in_chans=1,  # Grayscale X-rays
        mamba_layers=mamba_layers,
        dropout=0.1,
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
    print("Testing ViT-Mamba Hybrid model...")

    # Create model
    model = create_vit_mamba_model(
        num_classes=15,
        img_size=1024,
        model_size="base",
        mamba_layers=[3, 6, 9]  # Use Mamba at layers 3, 6, 9
    )

    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Show architecture
    print(f"\nArchitecture:")
    print(f"- Depth: {model.depth} layers")
    print(f"- Mamba layers: {sorted(model.mamba_layers)}")
    print(f"- Transformer layers: {[i for i in range(model.depth) if i not in model.mamba_layers]}")

    # Test forward pass with GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    model = model.to(device)
    dummy_input = torch.randn(1, 1, 1024, 1024).to(device)  # Smaller batch for testing

    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")  # Should be [1, 15]
        print(f"Output sample: {output[0][:5]}")  # First 5 logits

    print("\nModel test completed successfully!")