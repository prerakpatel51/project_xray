import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMamba(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        # Simple linear layers
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model, d_model)

        # Simple 1D convolution
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)

        # Split into two paths
        x1, x2 = self.linear1(x).chunk(2, dim=-1)

        # Apply activation
        x1 = F.relu(x1)

        # Simple convolution (transpose for conv1d)
        x1_conv = x1.transpose(1, 2)  # (batch, d_model, seq_len)
        x1_conv = self.conv(x1_conv)
        x1_conv = x1_conv.transpose(1, 2)  # back to (batch, seq_len, d_model)

        # Gate mechanism
        output = x1_conv * torch.sigmoid(x2)

        # Residual connection
        output = output + x

        return self.linear2(output)

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model

        # Simple learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, d_model) * 0.1)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        seq_len = x.size(1)

        # Add simple positional info (just first seq_len positions)
        x = x + self.pos_embed[:, :seq_len, :]

        return self.transformer(x)

class MambaFormer(nn.Module):
    def __init__(self, input_size=512*512*3, d_model=256, num_classes=15):
        super().__init__()

        # Project flattened image to model dimension
        self.input_proj = nn.Linear(input_size, d_model)

        # Simple Mamba layer
        self.mamba = SimpleMamba(d_model)

        # Simple Transformer layer
        self.transformer = SimpleTransformer(d_model, nhead=8, num_layers=2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 3, 512, 512)
        batch_size = x.size(0)

        # Flatten image: (batch_size, 3*512*512)
        x = x.view(batch_size, -1)

        # Project to model dimension: (batch_size, d_model)
        x = self.input_proj(x)

        # Add sequence dimension: (batch_size, 1, d_model)
        x = x.unsqueeze(1)

        # Process through Mamba
        x = self.mamba(x)

        # Process through Transformer
        x = self.transformer(x)

        # Remove sequence dimension and classify: (batch_size, d_model)
        x = x.squeeze(1)

        # Get final predictions: (batch_size, num_classes)
        return self.classifier(x)

def mamba_former(num_classes=15):
    """Create a simple MambaFormer model"""
    return MambaFormer(num_classes=num_classes)