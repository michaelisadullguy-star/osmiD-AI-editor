"""
Part 2: Dataset loader for training
Loads correlated imagery and masks for training
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OSMFeatureDataset(Dataset):
    """
    Dataset for loading satellite imagery and corresponding feature masks
    """

    FEATURE_CLASSES = {
        'background': 0,
        'building': 1,
        'lawn': 2,
        'natural_wood': 3,
        'artificial_forest': 4,
        'water_body': 5,
        'farmland': 6
    }

    def __init__(
        self,
        data_dir: str,
        cities: List[str],
        transform=None,
        img_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize dataset

        Args:
            data_dir: Directory containing correlated data
            cities: List of cities to include
            transform: Albumentations transform pipeline
            img_size: Target image size (height, width)
        """
        self.data_dir = data_dir
        self.cities = cities
        self.img_size = img_size

        # Default transform if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform

        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict]:
        """Prepare list of all training samples"""
        samples = []

        for city in self.cities:
            # Load training metadata
            metadata_path = os.path.join(self.data_dir, 'correlated', f"{city}_training_data.json")

            if not os.path.exists(metadata_path):
                print(f"Warning: No training data for {city}, skipping...")
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check if imagery exists
            imagery_path = os.path.join(self.data_dir, 'imagery', f"{city}_z{metadata['zoom']}.png")
            if not os.path.exists(imagery_path):
                print(f"Warning: No imagery for {city}, skipping...")
                continue

            samples.append({
                'city': city,
                'imagery_path': imagery_path,
                'metadata': metadata
            })

        return samples

    def _load_masks(self, city: str) -> np.ndarray:
        """
        Load all feature masks and combine into single multi-class mask

        Args:
            city: City name

        Returns:
            Multi-class mask (H, W) with values 0-5
        """
        mask_dir = os.path.join(self.data_dir, 'correlated')

        # Initialize combined mask
        first_mask = cv2.imread(
            os.path.join(mask_dir, f"{city}_building_mask.png"),
            cv2.IMREAD_GRAYSCALE
        )
        h, w = first_mask.shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        # Load each feature mask
        for feature_name, class_id in self.FEATURE_CLASSES.items():
            if feature_name == 'background':
                continue

            mask_path = os.path.join(mask_dir, f"{city}_{feature_name}_mask.png")

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                combined_mask[mask > 127] = class_id

        return combined_mask

    def _extract_contours(self, mask: np.ndarray, point_distance: float = 3.0) -> np.ndarray:
        """
        Extract contours from mask with specified point spacing

        Args:
            mask: Binary or multi-class mask
            point_distance: Distance between contour points in meters (default 3m)

        Returns:
            Contour mask (H, W)
        """
        contour_mask = np.zeros_like(mask, dtype=np.uint8)

        # Process each class separately
        for class_id in range(1, 6):
            class_mask = (mask == class_id).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(
                class_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw contours with specified thickness
            # Approximate point spacing based on image resolution
            # Assuming ~0.6m per pixel at zoom 17
            pixel_distance = max(1, int(point_distance / 0.6))

            cv2.drawContours(contour_mask, contours, -1, class_id, pixel_distance)

        return contour_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample

        Returns:
            Dictionary with:
            - image: (3, H, W) tensor
            - mask: (H, W) tensor with class labels
            - contour: (H, W) tensor with contour labels
        """
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['imagery_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load masks
        mask = self._load_masks(sample['city'])

        # Extract contours
        contour = self._extract_contours(mask, point_distance=3.0)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, masks=[mask, contour])
            image = transformed['image']
            mask = torch.from_numpy(transformed['masks'][0]).long()
            contour = torch.from_numpy(transformed['masks'][1]).long()

        return {
            'image': image,
            'mask': mask,
            'contour': contour,
            'city': sample['city']
        }


def create_dataloaders(
    data_dir: str,
    train_cities: List[str],
    val_cities: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 512)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders

    Args:
        data_dir: Root data directory
        train_cities: List of cities for training
        val_cities: List of cities for validation
        batch_size: Batch size
        num_workers: Number of worker processes
        img_size: Image size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training transform with augmentation
    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Validation transform without augmentation
    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = OSMFeatureDataset(data_dir, train_cities, train_transform, img_size)
    val_dataset = OSMFeatureDataset(data_dir, val_cities, val_transform, img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    train_cities = ['paris', 'london', 'new_york', 'hong_kong', 'moscow']
    val_cities = ['tokyo', 'singapore']

    train_loader, val_loader = create_dataloaders(
        data_dir='./data',
        train_cities=train_cities,
        val_cities=val_cities,
        batch_size=4
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Test loading one batch
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Contour shape: {batch['contour'].shape}")
        break
