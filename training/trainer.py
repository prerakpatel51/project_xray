"""
Advanced training pipeline for Vision Mamba medical image classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
import os
import time
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import csv


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-label classification"""

    def __init__(self, alpha=1, gamma=2, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        prob = torch.sigmoid(inputs)

        # Calculate binary cross entropy
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )

        # Calculate focal weight
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification with class imbalance"""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class MedicalMetrics:
    """Comprehensive metrics for medical multi-label classification"""

    @staticmethod
    def compute_metrics(y_true, y_pred, y_scores=None, threshold=0.5):
        """Compute comprehensive medical classification metrics"""
        metrics = {}

        # Convert predictions to binary
        y_pred_binary = (y_pred > threshold).astype(int)

        # Per-class and macro metrics
        if y_scores is not None:
            # AUC metrics
            try:
                metrics['macro_auc'] = roc_auc_score(y_true, y_scores, average='macro')
                metrics['micro_auc'] = roc_auc_score(y_true, y_scores, average='micro')
                metrics['weighted_auc'] = roc_auc_score(y_true, y_scores, average='weighted')

                # Per-class AUC
                auc_per_class = roc_auc_score(y_true, y_scores, average=None)
                for i, auc in enumerate(auc_per_class):
                    metrics[f'auc_class_{i}'] = auc

                # Average Precision
                metrics['macro_ap'] = average_precision_score(y_true, y_scores, average='macro')
                metrics['micro_ap'] = average_precision_score(y_true, y_scores, average='micro')

            except ValueError as e:
                print(f"Warning: Could not compute AUC metrics: {e}")

        # F1 scores
        metrics['macro_f1'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)

        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1

        # Additional metrics
        metrics['accuracy'] = (y_pred_binary == y_true).mean()

        # Hamming loss (fraction of wrong labels)
        metrics['hamming_loss'] = (y_pred_binary != y_true).mean()

        return metrics


class MedicalTrainer:
    """Advanced trainer for medical image classification"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        class_weights,
        device,
        config: Dict,
        save_dir: str = 'checkpoints',
        use_wandb: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights
        self.device = device
        self.config = config
        self.save_dir = save_dir
        self.use_wandb = use_wandb

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup loss function
        self._setup_loss_function()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Setup metrics
        self.metrics_calculator = MedicalMetrics()

        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_f1s = []

        # CSV logging
        self.csv_path = os.path.join(save_dir, 'training_log.csv')
        self._init_csv_logging()

        # Setup DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
            self.is_parallel = True
        else:
            self.is_parallel = False

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="vision-mamba-xray",
                config=config,
                name=f"vision_mamba_{config.get('model_size', 'base')}"
            )
            wandb.watch(self.model, log_freq=100)

    def _init_csv_logging(self):
        """Initialize CSV logging for losses and metrics"""
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'val_macro_auc', 'val_micro_auc',
            'val_macro_f1', 'val_micro_f1', 'val_macro_ap', 'val_accuracy',
            'val_hamming_loss', 'learning_rate', 'best_val_auc'
        ]

        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        print(f"CSV logging initialized: {self.csv_path}")

    def _log_to_csv(self, epoch, train_loss, val_metrics, lr):
        """Log training metrics to CSV file"""
        row_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics.get('loss', 0),
            'val_macro_auc': val_metrics.get('macro_auc', 0),
            'val_micro_auc': val_metrics.get('micro_auc', 0),
            'val_macro_f1': val_metrics.get('macro_f1', 0),
            'val_micro_f1': val_metrics.get('micro_f1', 0),
            'val_macro_ap': val_metrics.get('macro_ap', 0),
            'val_accuracy': val_metrics.get('accuracy', 0),
            'val_hamming_loss': val_metrics.get('hamming_loss', 0),
            'learning_rate': lr,
            'best_val_auc': self.best_val_auc
        }

        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
            writer.writerow(row_data)

    def _setup_loss_function(self):
        """Setup loss function based on configuration"""
        loss_type = self.config.get('loss_type', 'focal')

        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.get('focal_alpha', 1),
                gamma=self.config.get('focal_gamma', 2),
                pos_weight=self.class_weights
            )
        elif loss_type == 'asymmetric':
            self.criterion = AsymmetricLoss(
                gamma_neg=self.config.get('asym_gamma_neg', 4),
                gamma_pos=self.config.get('asym_gamma_pos', 1),
                clip=self.config.get('asym_clip', 0.05)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        print(f"Using {loss_type} loss function")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)

        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Scheduler
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None

    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, (images, targets, metadata) in enumerate(pbar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch
                })

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_outputs = []

        for images, targets, metadata in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Collect predictions
            total_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

        # Calculate metrics
        y_true = np.vstack(all_targets)
        y_scores = np.vstack(all_outputs)

        metrics = self.metrics_calculator.compute_metrics(
            y_true, y_scores, y_scores, threshold=0.5
        )

        metrics['loss'] = total_loss / len(self.val_loader)

        return metrics

    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        # Handle DataParallel model state dict
        if self.is_parallel:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'best_val_auc': self.best_val_auc,
            'best_val_f1': self.best_val_f1,
            'is_parallel': self.is_parallel,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with AUC: {metrics.get('macro_auc', 0):.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle DataParallel loading
        if self.is_parallel:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop"""
        # Resume training if checkpoint provided
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])

            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_aucs.append(val_metrics.get('macro_auc', 0))
            self.val_f1s.append(val_metrics.get('macro_f1', 0))

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('macro_auc', 0))
                else:
                    self.scheduler.step()

            # Check for best model
            current_auc = val_metrics.get('macro_auc', 0)
            current_f1 = val_metrics.get('macro_f1', 0)
            is_best = current_auc > self.best_val_auc

            if is_best:
                self.best_val_auc = current_auc
                self.best_val_f1 = current_f1

            # Log metrics
            print(f"\\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val AUC: {current_auc:.4f} (Best: {self.best_val_auc:.4f})")
            print(f"Val F1: {current_f1:.4f} (Best: {self.best_val_f1:.4f})")

            # CSV logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self._log_to_csv(epoch, train_metrics['loss'], val_metrics, current_lr)

            # Wandb logging
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }
                wandb.log(log_dict)

            # Save checkpoint EVERY epoch (not just at save_freq)
            self.save_checkpoint(val_metrics, is_best)
            print(f"Checkpoint saved: checkpoint_epoch_{epoch}.pth")

        print(f"\\nTraining completed!")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")

        # Save final training plots
        self.save_training_plots()

        if self.use_wandb:
            wandb.finish()

    def save_training_plots(self):
        """Save training progress plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        epochs = range(len(self.train_losses))
        axes[0, 0].plot(epochs, self.train_losses, label='Train')
        axes[0, 0].plot(epochs, self.val_losses, label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # AUC curve
        axes[0, 1].plot(epochs, self.val_aucs)
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].grid(True)

        # F1 curve
        axes[1, 0].plot(epochs, self.val_f1s)
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].grid(True)

        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, f'Best AUC: {self.best_val_auc:.4f}\\nBest F1: {self.best_val_f1:.4f}',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Final Results')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save training log
        training_log = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'val_f1s': self.val_f1s,
            'best_val_auc': self.best_val_auc,
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }

        with open(os.path.join(self.save_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use this module to train Vision Mamba models on medical X-ray data.")