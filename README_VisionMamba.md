# Vision Mamba for Medical X-ray Classification 🏥🔬

A state-of-the-art **Vision Mamba** implementation for multi-label chest X-ray classification on your dataset with 15 medical conditions.

## 🚀 Features

- **Vision Mamba Architecture**: Latest Mamba-based vision model for medical imaging
- **Multi-label Classification**: Handle 15 medical conditions simultaneously
- **Imbalanced Data Handling**: Focal loss, asymmetric loss, and class balancing
- **Advanced Augmentations**: Medical-specific data augmentations
- **Comprehensive Metrics**: AUC, AP, F1, sensitivity, specificity per class
- **Grad-CAM Visualization**: Interpretable AI for medical decisions
- **Professional Training Pipeline**: W&B integration, checkpointing, resuming

## 📊 Dataset

- **15 Medical Conditions**: No Finding, Infiltration, Effusion, Atelectasis, Nodule, Mass, Pneumothorax, Consolidation, Pleural_Thickening, Cardiomegaly, Emphysema, Edema, Fibrosis, Pneumonia, Hernia
- **110K+ X-ray Images**: Multi-label classification with heavy class imbalance
- **Patient Metadata**: Age, sex, view position integration

## 📁 Project Structure

```
project_xray/
├── models/
│   └── vision_mamba.py          # Vision Mamba architecture
├── data/
│   └── medical_dataset.py       # Enhanced dataset with medical transforms
├── training/
│   └── trainer.py               # Advanced training pipeline
├── inference/
│   └── evaluate.py              # Evaluation and Grad-CAM visualization
├── utils/
│   └── data_loader.py           # Original data loader (still functional)
├── dataset/
│   ├── images/                  # Your X-ray images
│   └── labels/
│       └── labels.csv           # Your labels file
├── checkpoints/                 # Model checkpoints (created during training)
├── train_vision_mamba.py        # Main training script
├── setup_environment.sh         # Environment setup script
├── requirements_mamba.txt        # Dependencies
└── visualization.ipynb          # Your existing visualization notebook
```

## 🛠️ Installation & Setup

### 1. Create New Environment
```bash
# Run the setup script
bash setup_environment.sh

# Or manually:
conda activate vision_mamba
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm einops transformers mamba-ssm causal-conv1d
pip install albumentations grad-cam wandb sklearn matplotlib seaborn
```

### 2. Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python models/vision_mamba.py  # Test model creation
```

## 🏃‍♂️ Quick Start

### 1. Basic Training
```bash
python train_vision_mamba.py \
    --model_size base \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --loss_type focal
```

### 2. Advanced Training with W&B
```bash
python train_vision_mamba.py \
    --model_size large \
    --batch_size 8 \
    --epochs 150 \
    --learning_rate 5e-5 \
    --loss_type asymmetric \
    --use_wandb \
    --balance_train
```

### 3. Resume Training
```bash
python train_vision_mamba.py \
    --resume checkpoints/checkpoint_epoch_50.pth \
    --epochs 100
```

## ⚙️ Configuration Options

### Model Sizes
- `tiny`: 384 dim, 6 layers (~22M params)
- `small`: 576 dim, 8 layers (~44M params)
- `base`: 768 dim, 12 layers (~86M params)
- `large`: 1024 dim, 16 layers (~307M params)

### Loss Functions
- `bce`: Binary Cross Entropy with class weights
- `focal`: Focal Loss (recommended for imbalanced data)
- `asymmetric`: Asymmetric Loss (best for multi-label)

### Key Parameters
```bash
--img_size 512              # Input image resolution
--patch_size 16             # Vision Mamba patch size
--batch_size 16             # Batch size (adjust for GPU memory)
--learning_rate 1e-4        # Learning rate
--weight_decay 0.01         # Weight decay
--grad_clip 1.0             # Gradient clipping
--balance_train             # Oversample rare conditions
```

## 📈 Training Pipeline

1. **Data Loading**: Enhanced medical dataset with augmentations
2. **Model**: Vision Mamba with bidirectional scanning
3. **Loss**: Focal/Asymmetric loss for imbalanced multi-label data
4. **Optimization**: AdamW with cosine annealing
5. **Validation**: Comprehensive metrics (AUC, AP, F1, sensitivity, specificity)
6. **Checkpointing**: Best model saving and resuming
7. **Logging**: W&B integration with training curves

## 🔍 Evaluation & Inference

### Comprehensive Evaluation
```python
from inference.evaluate import MedicalEvaluator
from models.vision_mamba import create_vision_mamba_model, MEDICAL_CONDITIONS

# Load trained model
model = load_model_for_inference('checkpoints/best_model.pth')

# Create evaluator
evaluator = MedicalEvaluator(model, device, MEDICAL_CONDITIONS)

# Evaluate on test set
results = evaluator.predict(test_loader)
metrics = evaluator.compute_metrics(results['predictions'], results['targets'])

# Generate plots
evaluator.plot_roc_curves(results['predictions'], results['targets'])
evaluator.plot_confusion_matrices(results['predictions'], results['targets'])
evaluator.create_evaluation_report(results['predictions'], results['targets'])
```

### Grad-CAM Visualization
```python
from inference.evaluate import GradCAMVisualizer

# Create visualizer
visualizer = GradCAMVisualizer(model, device, MEDICAL_CONDITIONS)

# Visualize predictions for an X-ray
visualizer.visualize_predictions(
    'dataset/images/00000001_000.png',
    transform=medical_transforms,
    top_k=3
)
```

## 🎯 Expected Performance

Based on similar medical imaging tasks:

| Metric | Expected Range |
|--------|----------------|
| **Macro AUC** | 0.75 - 0.85 |
| **Micro AUC** | 0.80 - 0.90 |
| **Macro F1** | 0.60 - 0.75 |
| **Common Conditions AUC** | 0.80 - 0.95 |
| **Rare Conditions AUC** | 0.65 - 0.80 |

## 🔧 Advanced Usage

### Custom Model Configuration
```python
from models.vision_mamba import VisionMamba

model = VisionMamba(
    img_size=512,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_classes=15,
    use_bidirectional=True,
    dropout=0.1
)
```

### Custom Loss Functions
```python
from training.trainer import FocalLoss, AsymmetricLoss

# Focal Loss
criterion = FocalLoss(alpha=1, gamma=2, pos_weight=class_weights)

# Asymmetric Loss
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `batch_size` or `img_size`
2. **Mamba not installing**: Install with conda: `conda install mamba-ssm -c conda-forge`
3. **Slow training**: Increase `num_workers` or use smaller model
4. **Poor performance**: Try `asymmetric` loss or increase `focal_gamma`

### Memory Optimization
```bash
# For limited GPU memory
python train_vision_mamba.py \
    --model_size small \
    --batch_size 8 \
    --img_size 384 \
    --grad_clip 0.5
```

## 📚 References

- [Vision Mamba Paper](https://arxiv.org/abs/2401.09417)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [ChestX-ray14 Dataset](https://arxiv.org/abs/1705.02315)

---

**Ready to train your Vision Mamba model? 🚀**

```bash
conda activate vision_mamba
python train_vision_mamba.py --use_wandb --model_size base
```