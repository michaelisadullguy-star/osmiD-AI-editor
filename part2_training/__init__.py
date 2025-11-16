"""
Part 2: Training Module
Machine learning pipeline for feature identification and contour detection
"""

from .model import FeatureSegmentationModel, UNet
from .dataset import OSMFeatureDataset, create_dataloaders
from .train import Trainer, CombinedLoss

__all__ = [
    'FeatureSegmentationModel',
    'UNet',
    'OSMFeatureDataset',
    'create_dataloaders',
    'Trainer',
    'CombinedLoss'
]
