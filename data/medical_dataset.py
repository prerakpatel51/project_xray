"""
Enhanced medical X-ray dataset with advanced preprocessing and multi-label handling
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import cv2
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Using torchvision transforms instead.")
    from torchvision import transforms


# Medical conditions mapping
MEDICAL_CONDITIONS = [
    'No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule',
    'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
    'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia', 'Hernia'
]

CONDITION_TO_IDX = {condition: idx for idx, condition in enumerate(MEDICAL_CONDITIONS)}


class MedicalTransforms:
    """Advanced medical imaging transformations"""

    @staticmethod
    def get_train_transforms(img_size=512):
        """Training transformations with medical-specific augmentations"""
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
            # Resize and normalize
            A.Resize(img_size, img_size),

            # Medical-specific augmentations
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),  # Enhance contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

            # Geometric augmentations (conservative for medical images)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            A.HorizontalFlip(p=0.5),

            # Noise and blur (simulate real-world conditions)
            A.OneOf([
                A.GaussNoise(var_limit=0.001, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # Normalize to [0, 1] and convert to tensor
            A.Normalize(mean=[0.485], std=[0.229]),  # ImageNet stats for grayscale
            ToTensorV2(),
        ])
        else:
            # Fallback to torchvision transforms
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    @staticmethod
    def get_val_transforms(img_size=512):
        """Validation transformations (minimal processing)"""
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    @staticmethod
    def get_test_transforms(img_size=512):
        """Test transformations with TTA options"""
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])


class MedicalXRayDataset(Dataset):
    """Enhanced dataset for multi-label chest X-ray classification"""

    def __init__(
        self,
        csv_file,
        images_dir,
        image_list=None,
        transforms=None,
        img_size=512,
        balance_classes=False,
    ):
        self.images_dir = images_dir
        self.transforms = transforms
        self.img_size = img_size

        # Load and filter dataframe
        self.df = pd.read_csv(csv_file)
        if image_list is not None:
            self.df = self.df[self.df['Image Index'].isin(image_list)]

        self.df = self.df.reset_index(drop=True)

        # Process labels for multi-label classification
        self._process_labels()

        # Calculate class weights for imbalanced data
        self._calculate_class_weights()

        # Optional class balancing
        if balance_classes:
            self._balance_dataset()

    def _process_labels(self):
        """Convert string labels to multi-hot encoded vectors"""
        # Parse labels from string format
        label_lists = []
        for _, row in self.df.iterrows():
            labels = row['Finding Labels'].split('|') if pd.notna(row['Finding Labels']) else []
            # Clean and filter labels
            labels = [label.strip() for label in labels if label.strip() in CONDITION_TO_IDX]
            label_lists.append(labels)

        # Create multi-label binarizer
        mlb = MultiLabelBinarizer(classes=MEDICAL_CONDITIONS)
        self.labels = mlb.fit_transform(label_lists)
        self.mlb = mlb

        # Store label statistics
        self.label_counts = np.sum(self.labels, axis=0)
        self.total_samples = len(self.labels)

    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        # Calculate positive/negative ratios for each class
        pos_counts = self.label_counts
        neg_counts = self.total_samples - pos_counts

        # Use balanced class weights
        self.pos_weights = neg_counts / (pos_counts + 1e-8)  # Avoid division by zero
        self.class_weights = torch.FloatTensor(self.pos_weights)

    def _balance_dataset(self):
        """Balance dataset using oversampling of rare conditions"""
        # Find samples with rare conditions (< 1000 cases)
        rare_threshold = 1000
        rare_condition_indices = np.where(self.label_counts < rare_threshold)[0]

        # Get samples containing rare conditions
        rare_samples_mask = np.any(self.labels[:, rare_condition_indices] == 1, axis=1)
        rare_sample_indices = np.where(rare_samples_mask)[0]

        # Oversample rare conditions
        oversample_factor = 2
        oversampled_indices = np.tile(rare_sample_indices, oversample_factor)

        # Combine original and oversampled indices
        all_indices = np.concatenate([np.arange(len(self.df)), oversampled_indices])

        # Update dataframe and labels
        self.df = self.df.iloc[all_indices].reset_index(drop=True)
        self.labels = self.labels[all_indices]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image info
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        image_path = os.path.join(self.images_dir, image_name)

        # Load image
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                # Fallback to PIL
                image = Image.open(image_path).convert('L')
                image = np.array(image)

            # Keep as grayscale (single channel for medical X-rays)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Apply transformations
        if self.transforms:
            if ALBUMENTATIONS_AVAILABLE:
                # Albumentations transforms
                transformed = self.transforms(image=image)
                image = transformed['image']
            else:
                # Torchvision transforms
                image = self.transforms(image)
                # Ensure single channel for grayscale
                if image.shape[0] == 3:
                    image = image[0:1]  # Keep only first channel
        else:
            # Basic preprocessing for grayscale
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.astype(np.float32) / 255.0
            # Add channel dimension for grayscale [H, W] -> [1, H, W]
            image = torch.from_numpy(image).unsqueeze(0)

        # Get multi-label target
        target = torch.FloatTensor(self.labels[idx])

        # Additional metadata
        metadata = {
            'image_name': image_name,
            'patient_age': row.get('Patient Age', -1),
            'patient_sex': row.get('Patient Sex', 'Unknown'),
            'view_position': row.get('View Position', 'Unknown'),
        }

        return image, target, metadata


class MedicalDataManager:
    """Manager for medical X-ray dataset with train/val/test splits"""

    def __init__(
        self,
        csv_file='dataset/labels/labels.csv',
        images_dir='dataset/images',
        train_list=None,
        test_list=None,
        img_size=512,
        val_split=0.2,
        balance_train=True,
    ):
        self.csv_file = csv_file
        self.images_dir = images_dir
        self.img_size = img_size
        self.val_split = val_split
        self.balance_train = balance_train

        # Load train/test splits if provided
        if train_list and os.path.exists(train_list):
            with open(train_list, 'r') as f:
                self.train_val_images = [line.strip() for line in f.readlines()]
        else:
            # Use all available images for training
            df = pd.read_csv(csv_file)
            self.train_val_images = df['Image Index'].tolist()

        if test_list and os.path.exists(test_list):
            with open(test_list, 'r') as f:
                self.test_images = [line.strip() for line in f.readlines()]
        else:
            # Reserve 20% for testing
            all_images = pd.read_csv(csv_file)['Image Index'].tolist()
            train_val, test = train_test_split(all_images, test_size=0.2, random_state=42)
            self.train_val_images = train_val
            self.test_images = test

    def create_datasets(self):
        """Create train/val/test datasets"""
        # Split train_val into train and validation
        train_images, val_images = train_test_split(
            self.train_val_images,
            test_size=self.val_split,
            random_state=42
        )

        print(f"Dataset splits:")
        print(f"  Train: {len(train_images):,} images")
        print(f"  Validation: {len(val_images):,} images")
        print(f"  Test: {len(self.test_images):,} images")

        # Create datasets with appropriate transforms
        train_dataset = MedicalXRayDataset(
            csv_file=self.csv_file,
            images_dir=self.images_dir,
            image_list=train_images,
            transforms=MedicalTransforms.get_train_transforms(self.img_size),
            img_size=self.img_size,
            balance_classes=self.balance_train,
        )

        val_dataset = MedicalXRayDataset(
            csv_file=self.csv_file,
            images_dir=self.images_dir,
            image_list=val_images,
            transforms=MedicalTransforms.get_val_transforms(self.img_size),
            img_size=self.img_size,
        )

        test_dataset = MedicalXRayDataset(
            csv_file=self.csv_file,
            images_dir=self.images_dir,
            image_list=self.test_images,
            transforms=MedicalTransforms.get_test_transforms(self.img_size),
            img_size=self.img_size,
        )

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self, batch_size=16, num_workers=4):
        """Create data loaders with appropriate sampling"""
        train_dataset, val_dataset, test_dataset = self.create_datasets()

        # Create weighted sampler for training (to handle class imbalance)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader, train_dataset.class_weights

    def analyze_dataset(self):
        """Analyze dataset statistics"""
        df = pd.read_csv(self.csv_file)

        print("\\nDataset Analysis:")
        print("="*50)

        # Overall statistics
        print(f"Total images: {len(df):,}")
        print(f"Unique patients: {df['Patient ID'].nunique():,}")

        # Age statistics
        if 'Patient Age' in df.columns:
            ages = df['Patient Age'].dropna()
            print(f"Age range: {ages.min():.0f} - {ages.max():.0f} years")
            print(f"Mean age: {ages.mean():.1f} ± {ages.std():.1f} years")

        # Gender distribution
        if 'Patient Sex' in df.columns:
            gender_counts = df['Patient Sex'].value_counts()
            print("Gender distribution:")
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count:,} ({count/len(df)*100:.1f}%)")

        # Label analysis
        all_labels = []
        for labels_str in df['Finding Labels'].dropna():
            labels = [label.strip() for label in labels_str.split('|')]
            all_labels.extend(labels)

        label_counts = Counter(all_labels)
        print(f"\\nMedical conditions ({len(label_counts)} total):")
        for condition, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {condition}: {count:,} ({count/len(all_labels)*100:.1f}%)")


if __name__ == "__main__":
    # Test the dataset
    data_manager = MedicalDataManager(
        csv_file='/home1/ppatel2025/project_xray/dataset/labels/labels.csv',
        images_dir='/home1/ppatel2025/project_xray/dataset/images',
        img_size=512,
    )

    # Analyze dataset
    data_manager.analyze_dataset()

    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = data_manager.create_dataloaders(
        batch_size=4, num_workers=2
    )

    print(f"\\nClass weights: {class_weights[:5]}")  # Show first 5

    # Test a batch
    print("\\nTesting data loader...")
    for batch_idx, (images, targets, metadata) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Sample target: {targets[0]}")
        print(f"  Sample metadata: {metadata['image_name'][0]}")
        break

    print("Dataset testing completed!")