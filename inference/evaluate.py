"""
Evaluation and inference tools for Vision Mamba medical classification
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import cv2
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Grad-CAM for visualization
try:
    from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not available. Install with: pip install grad-cam")


class MedicalEvaluator:
    """Comprehensive evaluation for medical image classification"""

    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)

    @torch.no_grad()
    def predict(self, data_loader, return_features=False):
        """Make predictions on a dataset"""
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_features = []
        all_metadata = []

        for images, targets, metadata in tqdm(data_loader, desc='Predicting'):
            images = images.to(self.device, non_blocking=True)

            # Forward pass
            if return_features:
                features = self.model.forward_features(images)
                outputs = self.model.classifier(features.mean(dim=1))
                all_features.append(features.cpu().numpy())
            else:
                outputs = self.model(images)

            # Store predictions and targets
            probabilities = torch.sigmoid(outputs)
            all_predictions.append(probabilities.cpu().numpy())
            all_targets.append(targets.numpy())
            all_metadata.extend(metadata['image_name'])

        # Concatenate all results
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        results = {
            'predictions': predictions,
            'targets': targets,
            'metadata': all_metadata
        }

        if return_features:
            results['features'] = np.vstack(all_features)

        return results

    def compute_metrics(self, predictions, targets, threshold=0.5):
        """Compute comprehensive evaluation metrics"""
        # Binary predictions
        binary_preds = (predictions > threshold).astype(int)

        metrics = {}

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            y_true_class = targets[:, i]
            y_pred_class = predictions[:, i]
            y_pred_binary_class = binary_preds[:, i]

            # Skip if no positive samples
            if y_true_class.sum() == 0:
                continue

            # AUC
            try:
                fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
                metrics[f'{class_name}_auc'] = auc(fpr, tpr)
            except:
                metrics[f'{class_name}_auc'] = 0.0

            # Average Precision
            try:
                metrics[f'{class_name}_ap'] = average_precision_score(y_true_class, y_pred_class)
            except:
                metrics[f'{class_name}_ap'] = 0.0

            # Sensitivity (Recall) and Specificity
            tp = np.sum((y_true_class == 1) & (y_pred_binary_class == 1))
            tn = np.sum((y_true_class == 0) & (y_pred_binary_class == 0))
            fp = np.sum((y_true_class == 0) & (y_pred_binary_class == 1))
            fn = np.sum((y_true_class == 1) & (y_pred_binary_class == 0))

            metrics[f'{class_name}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics[f'{class_name}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics[f'{class_name}_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics[f'{class_name}_f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        # Overall metrics
        try:
            metrics['macro_auc'] = np.mean([metrics[f'{name}_auc'] for name in self.class_names if f'{name}_auc' in metrics])
            metrics['macro_ap'] = np.mean([metrics[f'{name}_ap'] for name in self.class_names if f'{name}_ap' in metrics])
            metrics['macro_f1'] = np.mean([metrics[f'{name}_f1'] for name in self.class_names if f'{name}_f1' in metrics])
        except:
            metrics['macro_auc'] = 0.0
            metrics['macro_ap'] = 0.0
            metrics['macro_f1'] = 0.0

        # Subset accuracy (exact match)
        metrics['subset_accuracy'] = np.mean(np.all(binary_preds == targets, axis=1))

        # Hamming loss
        metrics['hamming_loss'] = np.mean(binary_preds != targets)

        return metrics

    def plot_roc_curves(self, predictions, targets, save_path=None):
        """Plot ROC curves for all classes"""
        n_classes = min(10, self.num_classes)  # Limit to top 10 for readability
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()

        for i in range(n_classes):
            y_true = targets[:, i]
            y_scores = predictions[:, i]

            # Skip if no positive samples
            if y_true.sum() == 0:
                axes[i].text(0.5, 0.5, 'No positive samples', ha='center', va='center')
                axes[i].set_title(f'{self.class_names[i]}\\n(No data)')
                continue

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            axes[i].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'AUC = {roc_auc:.3f}')
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{self.class_names[i]}\\n(AUC = {roc_auc:.3f})')
            axes[i].legend(loc="lower right")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_curves(self, predictions, targets, save_path=None):
        """Plot Precision-Recall curves for all classes"""
        n_classes = min(10, self.num_classes)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()

        for i in range(n_classes):
            y_true = targets[:, i]
            y_scores = predictions[:, i]

            if y_true.sum() == 0:
                axes[i].text(0.5, 0.5, 'No positive samples', ha='center', va='center')
                axes[i].set_title(f'{self.class_names[i]}\\n(No data)')
                continue

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)

            axes[i].plot(recall, precision, color='darkgreen', lw=2,
                        label=f'AP = {ap:.3f}')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f'{self.class_names[i]}\\n(AP = {ap:.3f})')
            axes[i].legend(loc="lower left")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrices(self, predictions, targets, threshold=0.5, save_path=None):
        """Plot confusion matrices for top classes"""
        binary_preds = (predictions > threshold).astype(int)

        # Find top classes by frequency
        class_frequencies = targets.sum(axis=0)
        top_indices = np.argsort(class_frequencies)[-6:]  # Top 6 classes

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, class_idx in enumerate(top_indices):
            cm = confusion_matrix(targets[:, class_idx], binary_preds[:, class_idx])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[idx])

            axes[idx].set_title(f'{self.class_names[class_idx]}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_evaluation_report(self, predictions, targets, save_path=None):
        """Create comprehensive evaluation report"""
        metrics = self.compute_metrics(predictions, targets)

        # Create DataFrame for better formatting
        report_data = []

        for class_name in self.class_names:
            if f'{class_name}_auc' in metrics:
                report_data.append({
                    'Class': class_name,
                    'AUC': metrics.get(f'{class_name}_auc', 0),
                    'AP': metrics.get(f'{class_name}_ap', 0),
                    'F1': metrics.get(f'{class_name}_f1', 0),
                    'Sensitivity': metrics.get(f'{class_name}_sensitivity', 0),
                    'Specificity': metrics.get(f'{class_name}_specificity', 0),
                    'Precision': metrics.get(f'{class_name}_precision', 0),
                })

        df_report = pd.DataFrame(report_data)

        # Add summary row
        summary_row = {
            'Class': 'MACRO AVERAGE',
            'AUC': metrics['macro_auc'],
            'AP': metrics['macro_ap'],
            'F1': metrics['macro_f1'],
            'Sensitivity': df_report['Sensitivity'].mean(),
            'Specificity': df_report['Specificity'].mean(),
            'Precision': df_report['Precision'].mean(),
        }
        df_report = pd.concat([df_report, pd.DataFrame([summary_row])], ignore_index=True)

        print("\\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        print(df_report.round(4).to_string(index=False))

        print(f"\\nOVERALL METRICS:")
        print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        print(f"Hamming Loss:    {metrics['hamming_loss']:.4f}")

        if save_path:
            df_report.to_csv(save_path.replace('.txt', '.csv'), index=False)
            with open(save_path, 'w') as f:
                f.write(df_report.to_string(index=False))

        return df_report, metrics


class GradCAMVisualizer:
    """Grad-CAM visualization for medical image interpretation"""

    def __init__(self, model, device, class_names, target_layers=None):
        self.model = model
        self.device = device
        self.class_names = class_names

        if not GRADCAM_AVAILABLE:
            print("Grad-CAM not available. Please install pytorch-grad-cam")
            return

        # Default target layer (last conv layer of Vision Mamba)
        if target_layers is None:
            # For Vision Mamba, we'll use the last Mamba block
            target_layers = [model.blocks[-1]]

        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate_cam(self, image, class_idx, transform=None):
        """Generate Grad-CAM for specific class"""
        if not GRADCAM_AVAILABLE:
            return None

        # Preprocess image
        if transform:
            input_tensor = transform(image).unsqueeze(0).to(self.device)
        else:
            input_tensor = image.unsqueeze(0).to(self.device)

        # Generate CAM
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        return grayscale_cam

    def visualize_predictions(self, image_path, transform, top_k=3, save_path=None):
        """Visualize top-k predictions with Grad-CAM"""
        if not GRADCAM_AVAILABLE:
            print("Grad-CAM not available")
            return

        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)

        # For display (convert to RGB)
        display_image = np.stack([image_array] * 3, axis=-1) / 255.0

        # Get model prediction
        input_tensor = transform(image_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs)[0]

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Create visualization
        fig, axes = plt.subplots(1, top_k + 1, figsize=(5 * (top_k + 1), 5))

        # Original image
        axes[0].imshow(display_image, cmap='gray' if len(display_image.shape) == 2 else None)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')

        # Grad-CAM for top predictions
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = self.class_names[idx]
            cam = self.generate_cam(image_array, idx, transform)

            if cam is not None:
                # Overlay CAM on image
                visualization = show_cam_on_image(display_image, cam, use_rgb=True)

                axes[i + 1].imshow(visualization)
                axes[i + 1].set_title(f'{class_name}\\nProb: {prob:.3f}')
                axes[i + 1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_model_for_inference(checkpoint_path, model_class, device):
    """Load trained model for inference"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (you'll need to adjust this based on your model creation)
    model = model_class()  # You'll need to pass the right parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation AUC: {checkpoint.get('best_val_auc', 'N/A')}")

    return model


if __name__ == "__main__":
    print("Evaluation tools loaded successfully!")
    print("Use MedicalEvaluator for comprehensive model evaluation.")
    print("Use GradCAMVisualizer for visual interpretation of predictions.")