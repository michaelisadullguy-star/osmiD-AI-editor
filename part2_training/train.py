"""
Part 2: Training Pipeline
Trains the feature detection model with contour accuracy (3m spacing)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import yaml
from typing import Dict
import numpy as np

from model import FeatureSegmentationModel
from dataset import create_dataloaders


class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation and contour detection
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        contour_weight: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.contour_weight = contour_weight

        # Segmentation loss (Cross Entropy)
        self.seg_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Contour loss (Focal Loss for imbalanced contours)
        self.contour_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss

        Args:
            predictions: Dict with 'segmentation' and 'contours' predictions
            targets: Dict with 'mask' and 'contour' ground truths

        Returns:
            Dictionary of losses
        """
        # Segmentation loss
        seg_loss = self.seg_loss(predictions['segmentation'], targets['mask'])

        # Contour loss
        contour_loss = self.contour_loss(predictions['contours'], targets['contour'])

        # Combined loss
        total_loss = self.seg_weight * seg_loss + self.contour_weight * contour_loss

        return {
            'total': total_loss,
            'segmentation': seg_loss,
            'contour': contour_loss
        }


class Trainer:
    """Training manager"""

    def __init__(self, config: Dict):
        """
        Initialize trainer

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = FeatureSegmentationModel(
            n_channels=3,
            n_classes=config['n_classes'],
            bilinear=config.get('bilinear', False)
        ).to(self.device)

        # Create loss function
        self.criterion = CombinedLoss(
            seg_weight=config.get('seg_weight', 1.0),
            contour_weight=config.get('contour_weight', 0.5)
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )

        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=config['data_dir'],
            train_cities=config['train_cities'],
            val_cities=config['val_cities'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            img_size=tuple(config.get('img_size', [512, 512]))
        )

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=config.get('log_dir', './logs'))

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        seg_loss = 0
        contour_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]")

        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            contours = batch['contour'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)

            # Calculate loss
            losses = self.criterion(
                predictions,
                {'mask': masks, 'contour': contours}
            )

            # Backward pass
            losses['total'].backward()
            self.optimizer.step()

            # Update metrics
            total_loss += losses['total'].item()
            seg_loss += losses['segmentation'].item()
            contour_loss += losses['contour'].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'seg': losses['segmentation'].item(),
                'cont': losses['contour'].item()
            })

        # Average losses
        n_batches = len(self.train_loader)
        metrics = {
            'total_loss': total_loss / n_batches,
            'seg_loss': seg_loss / n_batches,
            'contour_loss': contour_loss / n_batches
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        seg_loss = 0
        contour_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]")

            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                contours = batch['contour'].to(self.device)

                # Forward pass
                predictions = self.model(images)

                # Calculate loss
                losses = self.criterion(
                    predictions,
                    {'mask': masks, 'contour': contours}
                )

                # Update metrics
                total_loss += losses['total'].item()
                seg_loss += losses['segmentation'].item()
                contour_loss += losses['contour'].item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': losses['total'].item()
                })

        # Average losses
        n_batches = len(self.val_loader)
        metrics = {
            'total_loss': total_loss / n_batches,
            'seg_loss': seg_loss / n_batches,
            'contour_loss': contour_loss / n_batches
        }

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            'checkpoint_latest.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                'checkpoint_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Training cities: {self.config['train_cities']}")
        print(f"Validation cities: {self.config['val_cities']}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config['epochs']):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('SegLoss/train', train_metrics['seg_loss'], epoch)
            self.writer.add_scalar('SegLoss/val', val_metrics['seg_loss'], epoch)
            self.writer.add_scalar('ContourLoss/train', train_metrics['contour_loss'], epoch)
            self.writer.add_scalar('ContourLoss/val', val_metrics['contour_loss'], epoch)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(seg: {train_metrics['seg_loss']:.4f}, "
                  f"cont: {train_metrics['contour_loss']:.4f})")
            print(f"  Val Loss:   {val_metrics['total_loss']:.4f} "
                  f"(seg: {val_metrics['seg_loss']:.4f}, "
                  f"cont: {val_metrics['contour_loss']:.4f})")

            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']

            self.save_checkpoint(is_best)

        print("\n✓ Training completed!")
        self.writer.close()


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Default configuration
    config = {
        'data_dir': './data',
        'checkpoint_dir': './models/checkpoints',
        'log_dir': './logs',
        'train_cities': ['paris', 'london', 'new_york', 'hong_kong', 'moscow'],
        'val_cities': ['tokyo', 'singapore'],
        'n_classes': 6,
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_workers': 4,
        'img_size': [512, 512],
        'seg_weight': 1.0,
        'contour_weight': 0.5,
        'bilinear': False
    }

    trainer = Trainer(config)
    trainer.train()
