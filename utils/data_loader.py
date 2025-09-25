import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import Counter

class XRayDataset(Dataset):
    def __init__(self, csv_file, images_dir, image_list=None, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

        if image_list:
            self.labels_df = self.labels_df[self.labels_df['Image Index'].isin(image_list)]

        self.labels_df = self.labels_df.reset_index(drop=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_name = row['Image Index']
        image_path = os.path.join(self.images_dir, image_name)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = row['Finding Labels']

        return image, labels, image_name

class XRayDataManager:
    def __init__(self, csv_file='dataset/labels/labels.csv',
                 images_dir='dataset/images',
                 train_list='dataset/labels/train_val_list.txt',
                 test_list='dataset/labels/test_list.txt',
                 stats_file='dataset_stats.json'):
        self.csv_file = csv_file
        self.images_dir = images_dir
        self.train_list = train_list
        self.test_list = test_list

        # Make stats_file path relative to project root
        if not os.path.isabs(stats_file):
            # Find project root (directory containing 'dataset' folder)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != os.path.dirname(project_root):
                if os.path.exists(os.path.join(project_root, 'dataset')):
                    break
                project_root = os.path.dirname(project_root)
            self.stats_file = os.path.join(project_root, stats_file)
        else:
            self.stats_file = stats_file

        self.mean = None
        self.std = None

    def calculate_mean_std(self, dataset):
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_samples = 0

        for data, _, _ in dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        std /= total_samples
        return mean, std

    def save_mean_std(self, mean, std, filepath):
        stats = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f)

    def load_mean_std(self, filepath):
        with open(filepath, 'r') as f:
            stats = json.load(f)
        return torch.tensor(stats['mean']), torch.tensor(stats['std'])

    def normalize(self, tensor):
        """
        Normalize a tensor using the dataset's mean and std.

        Args:
            tensor: Input tensor to normalize (can be single image or batch)
                   Shape: [C, H, W] or [B, C, H, W]

        Returns:
            Normalized tensor with same shape as input
        """
        if self.mean is None or self.std is None:
            self._load_or_calculate_stats()

        # Ensure mean and std are on the same device as tensor
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)

        # Handle both single image [C, H, W] and batch [B, C, H, W]
        if tensor.dim() == 3:
            # Single image: [C, H, W]
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif tensor.dim() == 4:
            # Batch: [B, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")

        return (tensor - mean) / std

    def denormalize(self, tensor):
        """
        Denormalize a tensor using the dataset's mean and std.

        Args:
            tensor: Normalized tensor to denormalize
                   Shape: [C, H, W] or [B, C, H, W]

        Returns:
            Denormalized tensor with same shape as input
        """
        if self.mean is None or self.std is None:
            self._load_or_calculate_stats()

        # Ensure mean and std are on the same device as tensor
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)

        # Handle both single image [C, H, W] and batch [B, C, H, W]
        if tensor.dim() == 3:
            # Single image: [C, H, W]
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif tensor.dim() == 4:
            # Batch: [B, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")

        return tensor * std + mean

    def analyze_labels(self):
        df = pd.read_csv(self.csv_file)

        all_labels = []
        for labels_str in df['Finding Labels']:
            labels = [label.strip() for label in labels_str.split('|')]
            all_labels.extend(labels)

        label_counts = Counter(all_labels)

        # print("All unique classes and their frequencies:")
        # print("-" * 50)
        # for label, count in sorted(label_counts.items()):
        #     print(f"{label}: {count}")

        # print(f"\nTotal unique classes: {len(label_counts)}")
        # print(f"Total label instances: {sum(label_counts.values())}")

        return label_counts

    def _load_or_calculate_stats(self):
        if os.path.exists(self.stats_file):
            print("Loading saved mean and std...")
            self.mean, self.std = self.load_mean_std(self.stats_file)
            print(f"Mean: {self.mean}")
            print(f"Std: {self.std}")
        else:
            transform_no_norm = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])

            with open(self.train_list, 'r') as f:
                train_val_images = [line.strip() for line in f.readlines()]

            temp_dataset = XRayDataset(self.csv_file, self.images_dir, train_val_images, transform_no_norm)

            print("Calculating dataset mean and std...")
            self.mean, self.std = self.calculate_mean_std(temp_dataset)
            print(f"Mean: {self.mean}")
            print(f"Std: {self.std}")

            self.save_mean_std(self.mean, self.std, self.stats_file)
            print(f"Saved stats to {self.stats_file}")

    def create_datasets(self):
        self._load_or_calculate_stats()

        with open(self.train_list, 'r') as f:
            train_val_images = [line.strip() for line in f.readlines()]

        with open(self.test_list, 'r') as f:
            test_images = [line.strip() for line in f.readlines()]

        train_images, val_images = train_test_split(train_val_images, test_size=0.2, random_state=42)

        print(f"Train images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")
        print(f"Test images: {len(test_images)}")

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
        ])

        train_dataset = XRayDataset(self.csv_file, self.images_dir, train_images, transform)
        val_dataset = XRayDataset(self.csv_file, self.images_dir, val_images, transform)
        test_dataset = XRayDataset(self.csv_file, self.images_dir, test_images, transform)

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self, batch_size=32, shuffle_train=True):
        train_dataset, val_dataset, test_dataset = self.create_datasets()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_manager = XRayDataManager(csv_file='../dataset/labels/labels.csv',
                                   images_dir='../dataset/images',
                                   train_list='../dataset/labels/train_val_list.txt',
                                   test_list='../dataset/labels/test_list.txt')

    print("Analyzing dataset labels...")
    label_counts = data_manager.analyze_labels()

    print("\n" + "="*60 + "\n")

    train_dataset, val_dataset, test_dataset = data_manager.create_datasets()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader, val_loader, test_loader = data_manager.create_dataloaders()

    print("\nTesting train loader:")
    for batch_idx, (images, labels, image_names) in enumerate(train_loader):
        print(f"Batch {batch_idx}: {images.shape}, Labels: {labels}")
        if batch_idx == 0:
            break

    print("\nTesting val loader:")
    for batch_idx, (images, labels, image_names) in enumerate(val_loader):
        print(f"Batch {batch_idx}: {images.shape}, Labels: {labels}")
        if batch_idx == 0:
            break