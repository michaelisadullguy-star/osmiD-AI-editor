"""
Utility script to run model training
"""

import os
import argparse
import yaml
from part2_training.train import Trainer


def main():
    """Run training with command-line arguments"""
    parser = argparse.ArgumentParser(description='Train osmiD-AI-editor model')

    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Root data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./models/checkpoints',
                       help='Checkpoint save directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='TensorBoard log directory')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (overrides other args)')

    args = parser.parse_args()

    # Load config from file or use command-line args
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'data_dir': args.data_dir,
            'checkpoint_dir': args.checkpoint_dir,
            'log_dir': args.log_dir,
            'train_cities': ['paris', 'london', 'new_york', 'hong_kong', 'moscow'],
            'val_cities': ['tokyo', 'singapore'],
            'n_classes': 6,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': 1e-5,
            'num_workers': args.num_workers,
            'img_size': [args.img_size, args.img_size],
            'seg_weight': 1.0,
            'contour_weight': 0.5,
            'bilinear': False
        }

    print("=" * 80)
    print("osmiD-AI-editor - Model Training")
    print("=" * 80)
    print()
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
