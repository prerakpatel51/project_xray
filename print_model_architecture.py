#!/usr/bin/env python3
"""
Script to print Vision Mamba model architecture and details
"""

import torch
import torch.nn as nn
from models.vision_mamba import create_vision_mamba_model, MEDICAL_CONDITIONS
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_summary(model, model_name="Vision Mamba"):
    """Print detailed model summary"""
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} ARCHITECTURE SUMMARY")
    print(f"{'='*80}")

    # Model parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    print(f"\n{'='*80}")
    print("DETAILED ARCHITECTURE")
    print(f"{'='*80}")

    # Print each module with parameter counts
    total_counted = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                param_size = param_count * 4 / 1024  # KB
                print(f"{name:<50} {str(module):<30} {param_count:>10,} params ({param_size:>6.1f} KB)")
                total_counted += param_count

    print(f"{'='*80}")
    print(f"Total Counted: {total_counted:,} parameters")

    return model

def print_layer_by_layer(model, input_shape=(1, 1, 512, 512)):
    """Print layer-by-layer forward pass shapes"""
    print(f"\n{'='*80}")
    print("FORWARD PASS TENSOR SHAPES")
    print(f"{'='*80}")

    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)

    print(f"Input shape: {dummy_input.shape}")
    print(f"{'-'*80}")

    model.eval()
    with torch.no_grad():
        # Patch embedding
        x = model.patch_embed(dummy_input)
        print(f"After patch_embed: {x.shape}")

        # Add positional embedding
        x = x + model.pos_embed
        x = model.pos_dropout(x)
        print(f"After pos_embed: {x.shape}")

        # Through Mamba blocks
        for i, block in enumerate(model.blocks):
            x = block(x)
            print(f"After block_{i}: {x.shape}")

        # Normalization
        x = model.norm(x)
        print(f"After norm: {x.shape}")

        # Global pooling
        x_pooled = x.transpose(1, 2)
        x_pooled = model.global_pool(x_pooled)
        x_pooled = x_pooled.flatten(1)
        print(f"After global_pool: {x_pooled.shape}")

        # Classification head
        output = model.classifier(x_pooled)
        print(f"Final output: {output.shape}")

        # Apply sigmoid for probabilities
        probs = torch.sigmoid(output)
        print(f"After sigmoid: {probs.shape}")

def main():
    """Main function to print all model architectures"""
    print("VISION MAMBA MODEL ARCHITECTURES")
    print("=" * 80)

    # Print configurations for all model sizes
    model_configs = {
        'tiny': {'embed_dim': 384, 'depth': 6},
        'small': {'embed_dim': 576, 'depth': 8},
        'base': {'embed_dim': 768, 'depth': 12},
        'large': {'embed_dim': 1024, 'depth': 16}
    }

    print("\nMODEL SIZE CONFIGURATIONS:")
    print("-" * 50)
    for size, config in model_configs.items():
        print(f"{size.upper():<10} Embed Dim: {config['embed_dim']:<4} Depth: {config['depth']:<2} layers")

    # Create and analyze base model
    print(f"\n\nCreating BASE model for detailed analysis...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_vision_mamba_model(
        num_classes=len(MEDICAL_CONDITIONS),
        img_size=512,
        model_size="base"
    )
    model.to(device)

    # Print model summary
    print_model_summary(model, "Vision Mamba Base")

    # Print layer shapes
    print_layer_by_layer(model)

    # Print medical conditions
    print(f"\n{'='*80}")
    print("MEDICAL CONDITIONS (15 classes)")
    print(f"{'='*80}")
    for i, condition in enumerate(MEDICAL_CONDITIONS):
        print(f"{i:2d}. {condition}")

    # Print model structure
    print(f"\n{'='*80}")
    print("COMPLETE MODEL STRUCTURE")
    print(f"{'='*80}")
    print(model)

    # Test inference
    print(f"\n{'='*80}")
    print("TEST INFERENCE")
    print(f"{'='*80}")

    model.eval()
    with torch.no_grad():
        # Test batch
        test_input = torch.randn(2, 1, 512, 512).to(device)  # Batch of 2 grayscale X-rays
        output = model(test_input)
        probabilities = torch.sigmoid(output)

        print(f"Test input shape: {test_input.shape}")
        print(f"Model output shape: {output.shape}")
        print(f"Output probabilities shape: {probabilities.shape}")
        print(f"\nSample predictions for first image:")
        for i, (condition, prob) in enumerate(zip(MEDICAL_CONDITIONS, probabilities[0])):
            print(f"  {condition:<20}: {prob.item():.4f}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()