#!/usr/bin/env python3
"""
Evaluation script for Vision Mamba model on test set
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add project modules to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.vision_mamba import create_vit_mamba_model, MEDICAL_CONDITIONS
from data.medical_dataset import MedicalDataManager
from training.trainer import MedicalMetrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Vision Mamba on test set')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--csv_file', type=str, default='dataset/labels/labels.csv',
                       help='Path to labels CSV file')
    parser.add_argument('--images_dir', type=str, default='dataset/images',
                       help='Path to images directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_results', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    # Create model
    model = create_vit_mamba_model(
        num_classes=len(MEDICAL_CONDITIONS),
        img_size=config.get('img_size', 1024),
        model_size=config.get('model_size', 'base'),
        mamba_layers=config.get('mamba_layers', [3, 6, 9]),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    best_auc = checkpoint.get('best_val_auc', 'unknown')

    print(f"Model loaded from epoch {epoch}")
    print(f"Best validation AUC: {best_auc}")

    return model, config


def evaluate_model(model, test_loader, device, save_dir):
    """Evaluate model on test set"""
    print("\\nEvaluating model on test set...")

    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    # Collect predictions
    with torch.no_grad():
        for batch_idx, (images, targets, metadata) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)

            # Store results
            all_predictions.append((probabilities > 0.5).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches")

    # Concatenate all results
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_predictions)
    y_scores = np.vstack(all_probabilities)

    print(f"\\nEvaluation completed!")
    print(f"Test samples: {y_true.shape[0]}")
    print(f"Medical conditions: {y_true.shape[1]}")

    # Calculate metrics
    metrics_calc = MedicalMetrics()
    metrics = metrics_calc.compute_metrics(y_true, y_pred, y_scores)

    # Print overall metrics
    print("\\n" + "="*60)
    print("OVERALL TEST RESULTS")
    print("="*60)
    print(f"Macro AUC:     {metrics.get('macro_auc', 0):.4f}")
    print(f"Micro AUC:     {metrics.get('micro_auc', 0):.4f}")
    print(f"Macro F1:      {metrics.get('macro_f1', 0):.4f}")
    print(f"Micro F1:      {metrics.get('micro_f1', 0):.4f}")
    print(f"Accuracy:      {metrics.get('accuracy', 0):.4f}")
    print(f"Hamming Loss:  {metrics.get('hamming_loss', 0):.4f}")

    # Per-class results
    print("\\n" + "="*60)
    print("PER-CLASS RESULTS")
    print("="*60)
    print(f"{'Condition':20} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)

    results_data = []
    for i, condition in enumerate(MEDICAL_CONDITIONS):
        auc = metrics.get(f'auc_class_{i}', 0)
        f1 = metrics.get(f'f1_class_{i}', 0)
        precision = metrics.get(f'precision_class_{i}', 0)
        recall = metrics.get(f'recall_class_{i}', 0)

        print(f"{condition:20} {auc:8.4f} {f1:8.4f} {precision:10.4f} {recall:8.4f}")

        results_data.append({
            'condition': condition,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'support': y_true[:, i].sum()
        })

    # Save detailed results
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(save_dir, 'per_class_results.csv'), index=False)

    # Save overall metrics
    overall_metrics = {
        'macro_auc': metrics.get('macro_auc', 0),
        'micro_auc': metrics.get('micro_auc', 0),
        'macro_f1': metrics.get('macro_f1', 0),
        'micro_f1': metrics.get('micro_f1', 0),
        'accuracy': metrics.get('accuracy', 0),
        'hamming_loss': metrics.get('hamming_loss', 0)
    }

    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(os.path.join(save_dir, 'overall_results.csv'), index=False)

    # Create visualizations
    create_evaluation_plots(results_df, overall_metrics, save_dir)

    print(f"\\nResults saved to: {save_dir}")
    print("Files created:")
    print("  - per_class_results.csv")
    print("  - overall_results.csv")
    print("  - evaluation_plots.png")

    return metrics


def create_evaluation_plots(results_df, overall_metrics, save_dir):
    """Create evaluation visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Per-class AUC scores
    axes[0, 0].barh(results_df['condition'], results_df['auc'])
    axes[0, 0].set_xlabel('AUC Score')
    axes[0, 0].set_title('Per-Class AUC Scores')
    axes[0, 0].set_xlim(0, 1)

    # 2. Per-class F1 scores
    axes[0, 1].barh(results_df['condition'], results_df['f1'])
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_title('Per-Class F1 Scores')
    axes[0, 1].set_xlim(0, 1)

    # 3. Precision vs Recall
    axes[1, 0].scatter(results_df['recall'], results_df['precision'])
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)

    # Add condition labels
    for i, condition in enumerate(results_df['condition']):
        axes[1, 0].annotate(condition[:8],
                           (results_df['recall'].iloc[i], results_df['precision'].iloc[i]),
                           fontsize=8, alpha=0.7)

    # 4. Overall metrics
    metrics_names = list(overall_metrics.keys())
    metrics_values = list(overall_metrics.values())

    axes[1, 1].bar(metrics_names, metrics_values)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Overall Metrics')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Setup data
    print("\\nSetting up test data loader...")
    data_manager = MedicalDataManager(
        csv_file=args.csv_file,
        images_dir=args.images_dir,
        img_size=config.get('img_size', 1024),
        balance_train=False,  # No balancing for test set
    )

    # Create data loaders
    _, _, test_loader, _ = data_manager.create_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
    )

    print(f"Test batches: {len(test_loader)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate model
    metrics = evaluate_model(model, test_loader, device, args.save_results)

    print("\\nEvaluation completed successfully! 🎉")


if __name__ == "__main__":
    main()