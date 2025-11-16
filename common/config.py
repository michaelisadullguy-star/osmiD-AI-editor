"""
Centralized configuration management
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration with sensible defaults"""

    # API Configuration
    mapbox_access_token: Optional[str] = None
    osm_api_url: str = "https://api.openstreetmap.org"

    # Paths
    data_dir: str = './data'
    checkpoint_dir: str = './models/checkpoints'
    log_dir: str = './logs'

    # Dataset Configuration
    train_cities: List[str] = field(default_factory=lambda: [
        'paris', 'london', 'new_york', 'hong_kong', 'moscow'
    ])
    val_cities: List[str] = field(default_factory=lambda: ['tokyo', 'singapore'])

    # Model Configuration
    n_classes: int = 6
    bilinear: bool = False

    # Training Hyperparameters
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Data Loading
    num_workers: int = 4
    img_size: List[int] = field(default_factory=lambda: [512, 512])

    # Loss Weights
    seg_weight: float = 1.0
    contour_weight: float = 0.5

    # Feature Detection
    contour_point_distance: float = 3.0  # meters
    min_feature_area: int = 100  # pixels

    # Safety Configuration
    dry_run: bool = True  # Default to dry-run mode for safety
    require_review: bool = True  # Require human review before upload
    max_features_per_upload: int = 100  # Limit features per upload

    @classmethod
    def load(cls, config_path: str = 'config.yaml', env_file: str = '.env') -> 'Config':
        """
        Load configuration from YAML file and environment variables

        Args:
            config_path: Path to YAML configuration file
            env_file: Path to .env file

        Returns:
            Config instance
        """
        # Load environment variables
        if Path(env_file).exists():
            load_dotenv(env_file)

        # Load YAML config if exists
        config_data = {}
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}

        # Override with environment variables (higher priority)
        env_overrides = {
            'mapbox_access_token': os.getenv('MAPBOX_ACCESS_TOKEN'),
            'osm_api_url': os.getenv('OSM_API_URL'),
            'data_dir': os.getenv('TRAINING_DATA_DIR'),
            'checkpoint_dir': os.getenv('MODEL_OUTPUT_DIR'),
            'batch_size': os.getenv('BATCH_SIZE'),
            'epochs': os.getenv('EPOCHS'),
            'learning_rate': os.getenv('LEARNING_RATE'),
        }

        # Merge configs (env vars override file)
        for key, value in env_overrides.items():
            if value is not None:
                # Type conversion for numeric values
                if key in ['batch_size', 'epochs']:
                    value = int(value)
                elif key in ['learning_rate', 'weight_decay']:
                    value = float(value)
                config_data[key] = value

        return cls(**config_data)

    def save(self, config_path: str = 'config.yaml'):
        """
        Save configuration to YAML file

        Args:
            config_path: Path to save configuration
        """
        # Convert to dict, excluding sensitive data
        config_dict = {
            'data_dir': self.data_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'train_cities': self.train_cities,
            'val_cities': self.val_cities,
            'n_classes': self.n_classes,
            'bilinear': self.bilinear,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_workers': self.num_workers,
            'img_size': self.img_size,
            'seg_weight': self.seg_weight,
            'contour_weight': self.contour_weight,
            'contour_point_distance': self.contour_point_distance,
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def validate(self):
        """Validate configuration values"""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if len(self.img_size) != 2:
            raise ValueError(f"img_size must be [height, width], got {self.img_size}")

        if not self.train_cities:
            raise ValueError("train_cities cannot be empty")

        if not self.val_cities:
            raise ValueError("val_cities cannot be empty")

        # Check for overlap
        overlap = set(self.train_cities) & set(self.val_cities)
        if overlap:
            raise ValueError(f"train_cities and val_cities overlap: {overlap}")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config):
    """Set global configuration instance"""
    global _config
    _config = config
