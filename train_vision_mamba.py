#!/usr/bin/env python3
"""
Main training script for Vision Mamba on chest X-ray classification
"""

import os
import sys
import torch
import numpy as np
import random
import argparse
from pathlib import Path

# Add project modules to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.vision_mamba import create_vit_mamba_model, MEDICAL_CONDITIONS
from data.medical_dataset import MedicalDataManager
from training.trainer import MedicalTrainer


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Vision Mamba on chest X-rays')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--csv_file', type=str, default='dataset/labels/labels.csv',
                       help='Path to labels CSV file')
    parser.add_argument('--images_dir', type=str, default='dataset/images',
                       help='Path to images directory')

    # Model arguments
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size variant')
    parser.add_argument('--img_size', type=int, default=1024,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for Vision Mamba')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')

    # Loss function
    parser.add_argument('--loss_type', type=str, default='bce',
                       choices=['bce', 'focal', 'asymmetric'],
                       help='Loss function type')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='Learning rate scheduler')

    # Data augmentation
    parser.add_argument('--balance_train', action='store_true', default=True,
                       help='Balance training data by oversampling rare classes')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=12,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Checkpoint saving frequency (save every N epochs)')
    parser.add_argument('--save_csv', action='store_true', default=True,
                       help='Save training/validation losses to CSV file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='vision-mamba-xray',
                       help='W&B project name')

    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def create_config(args):
    """Create configuration dictionary"""
    config = {
        # Model config
        'model_size': args.model_size,
        'mamba_layers': [3, 6, 9],
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'num_classes': len(MEDICAL_CONDITIONS),

        # Training config
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,

        # Loss config
        'loss_type': args.loss_type,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,

        # Optimizer config
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,

        # Data config
        'balance_train': args.balance_train,
        'num_workers': args.num_workers,

        # System config
        'seed': args.seed,
        'save_freq': args.save_freq,
    }

    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = setup_device(args.device)

    # Create configuration
    config = create_config(args)

    # Print configuration
    print("\\n" + "="*60)
    print("VISION MAMBA TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("="*60)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup data
    print("\\nSetting up data loaders...")
    data_manager = MedicalDataManager(
        csv_file=args.csv_file,
        images_dir=args.images_dir,
        img_size=args.img_size,
        balance_train=args.balance_train,
    )

    # Analyze dataset
    data_manager.analyze_dataset()

    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = data_manager.create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"\\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print(f"\\nCreating ViT-Mamba Hybrid model ({args.model_size})...")
    model = create_vit_mamba_model(
        num_classes=len(MEDICAL_CONDITIONS),
        img_size=args.img_size,
        model_size=args.model_size,
        mamba_layers=[3, 6, 9],  # Use Mamba at layers 3, 6, 9
    )

    # Move model to device
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Move class weights to device
    class_weights = class_weights.to(device)

    # Create trainer
    print("\\nInitializing trainer...")
    trainer = MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        config=config,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
    )

    # Start training
    print("\\nStarting training...")
    try:
        trainer.train(
            num_epochs=args.epochs,
            resume_from=args.resume
        )
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
        # Save current state
        trainer.save_checkpoint({}, is_best=False)
    except Exception as e:
        print(f"\\nTraining failed with error: {e}")
        raise

    print("\\nTraining completed successfully!")

    # Test on best model
    print("\\nEvaluating best model on test set...")
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        # Load best model
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Evaluate on test set
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for images, targets, metadata in test_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs = torch.sigmoid(outputs)

                test_predictions.append(probs.cpu().numpy())
                test_targets.append(targets.numpy())

        # Calculate test metrics
        y_true_test = np.vstack(test_targets)
        y_scores_test = np.vstack(test_predictions)

        test_metrics = trainer.metrics_calculator.compute_metrics(
            y_true_test, y_scores_test, y_scores_test
        )

        print("\\nTest Set Results:")
        print("-" * 40)
        print(f"AUC (macro): {test_metrics.get('macro_auc', 0):.4f}")
        print(f"AUC (micro): {test_metrics.get('micro_auc', 0):.4f}")
        print(f"F1 (macro):  {test_metrics.get('macro_f1', 0):.4f}")
        print(f"F1 (micro):  {test_metrics.get('micro_f1', 0):.4f}")
        print(f"Accuracy:    {test_metrics.get('accuracy', 0):.4f}")

        # Per-class results
        print("\\nPer-class AUC scores:")
        for i, condition in enumerate(MEDICAL_CONDITIONS):
            auc = test_metrics.get(f'auc_class_{i}', 0)
            print(f"  {condition:20s}: {auc:.4f}")

    print("\\nAll done! 🎉")


if __name__ == "__main__":
    main()