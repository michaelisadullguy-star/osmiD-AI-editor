# osmiD-AI-editor

**Machine Learning-Based Automated OpenStreetMap Feature Mapper**

An advanced machine learning system that automatically detects and maps geographic features (buildings, lawns, natural woods, artificial forests, and water bodies) to OpenStreetMap using satellite imagery.

## Overview

The osmiD-AI-editor is divided into three main components:

### Part 1: Training Data Acquisition
- Downloads OpenStreetMap (OSM) data for major cities using the Overpass API
- Downloads Mapbox satellite imagery
- Correlates features in satellite imagery with corresponding OSM data
- Supported cities: Paris, London, New York, Hong Kong, Moscow, Tokyo, and Singapore
- Supported features: buildings, lawns, natural woods, artificial forests, and water bodies

### Part 2: Machine Learning Training
- U-Net based semantic segmentation model
- Identifies area features and determines their contours
- Contour path accuracy: one point every **3 meters**
- Multi-task learning: simultaneous segmentation and contour detection
- Training on correlated satellite imagery and OSM ground truth data

### Part 3: Executable Application
- Standalone executable (.exe) with GUI interface
- Three main input fields:
  1. **Email**: OSM account email
  2. **Password**: OSM account password
  3. **Polygon Coordinates**: User-defined area in format `{{lat,lon},{lat,lon},...}`
- Automatically logs into OSM, detects features, and maps them

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- Mapbox API access token
- OpenStreetMap account

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/michaelisadullguy-star/osmiD-AI-editor.git
cd osmiD-AI-editor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your Mapbox access token
```

## Usage

### Part 1: Data Acquisition

Download OSM data and satellite imagery for training:

```bash
# Download OSM data for all cities
python -m part1_data_acquisition.osm_downloader

# Download Mapbox satellite imagery
python -m part1_data_acquisition.mapbox_imagery

# Correlate features with imagery
python -m part1_data_acquisition.feature_correlator
```

Or use the provided script:

```bash
python run_data_acquisition.py
```

### Part 2: Training

Train the feature detection model:

```bash
python -m part2_training.train
```

This will:
- Load correlated training data
- Train the U-Net model for 100 epochs (default)
- Save checkpoints to `./models/checkpoints/`
- Log training progress to TensorBoard

**Monitor training:**
```bash
tensorboard --logdir=./logs
```

### Part 3: Running the Executable

#### Option 1: Run as Python Application

```bash
python -m part3_executable.gui_application
```

#### Option 2: Build Standalone Executable

```bash
python part3_executable/build_executable.py
```

The executable will be created in `./dist/osmiD-AI-editor.exe`

### Using the Application

1. **Launch the application**
2. **Enter credentials:**
   - Email: Your OSM account email
   - Password: Your OSM account password
3. **Define polygon:**
   - Format: `{{lat1,lon1},{lat2,lon2},{lat3,lon3},...}`
   - Example: `{{48.8566,2.3522},{48.8577,2.3540},{48.8560,2.3545},{48.8555,2.3530}}`
4. **Configure:**
   - Model Path: Path to trained model checkpoint
   - Mapbox Token: Your Mapbox access token
5. **Click "Start Feature Mapping"**

The application will:
- Download satellite imagery for your polygon
- Detect features using the trained AI model
- Upload detected features to OpenStreetMap

## Project Structure

```
osmiD-AI-editor/
├── part1_data_acquisition/      # Data downloading and correlation
│   ├── osm_downloader.py         # OSM data via Overpass API
│   ├── mapbox_imagery.py         # Satellite imagery downloader
│   └── feature_correlator.py     # Feature correlation system
│
├── part2_training/               # Machine learning training
│   ├── model.py                  # U-Net architecture
│   ├── dataset.py                # PyTorch dataset loader
│   └── train.py                  # Training pipeline
│
├── part3_executable/             # Executable application
│   ├── gui_application.py        # Main GUI application
│   ├── osm_client.py             # OSM API client
│   ├── feature_detector.py       # Feature detection inference
│   └── build_executable.py       # PyInstaller build script
│
├── data/                         # Training data (created at runtime)
│   ├── osm/                      # OSM GeoJSON files
│   ├── imagery/                  # Satellite imagery
│   └── correlated/               # Correlated training pairs
│
├── models/                       # Trained models
│   └── checkpoints/              # Model checkpoints
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## Feature Classes

The model detects and maps the following feature types:

| Class ID | Feature Type       | OSM Tags                          |
|----------|-------------------|-----------------------------------|
| 0        | Background        | N/A                               |
| 1        | Building          | `building=yes`                    |
| 2        | Lawn              | `landuse=grass`, `grass=lawn`     |
| 3        | Natural Wood      | `natural=wood`                    |
| 4        | Artificial Forest | `landuse=forest`                  |
| 5        | Water Body        | `natural=water`                   |

## Model Architecture

- **Base Model**: U-Net with ResNet-like encoder
- **Input**: RGB satellite imagery (512x512)
- **Output**:
  - Semantic segmentation mask (6 classes)
  - Contour probability map
- **Contour Precision**: One point every 3 meters
- **Parameters**: ~31 million trainable parameters

## Training Configuration

Default training parameters (configurable in `train.py`):

```python
config = {
    'batch_size': 8,
    'epochs': 100,
    'learning_rate': 0.001,
    'img_size': [512, 512],
    'contour_point_distance': 3.0,  # meters
}
```

## API Requirements

### Mapbox API
- Sign up at https://www.mapbox.com/
- Create an access token
- Add to `.env` file

### OpenStreetMap API
- Create account at https://www.openstreetmap.org/
- Use credentials in the application
- **Important**: Follow OSM's [Automated Edits Code of Conduct](https://wiki.openstreetmap.org/wiki/Automated_Edits_code_of_conduct)

## Important Notes

⚠️ **OSM Community Guidelines**
- Always review detected features before uploading
- Follow OSM's automated edits policy
- Test in a development environment first
- Add appropriate changeset comments
- Be prepared to handle disputes

⚠️ **Data Usage**
- Mapbox satellite imagery is subject to their terms of service
- OSM data is licensed under ODbL
- Respect API rate limits

## Performance

Expected performance metrics:

- **Training Time**: ~12-24 hours on GPU (100 epochs)
- **Inference Speed**: ~2-5 seconds per 512x512 image
- **Feature Detection Accuracy**: ~85-90% IoU on validation set
- **Contour Precision**: ±2-4 meters

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**2. Mapbox API Error**
```bash
# Check your access token in .env
# Verify token at https://account.mapbox.com/access-tokens/
```

**3. OSM Authentication Failed**
```bash
# Verify credentials
# Check if account has been confirmed
# Ensure no special characters in password
```

**4. CUDA Out of Memory**
```bash
# Reduce batch size in training config
# Use smaller image size
# Use CPU instead: device='cpu'
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenStreetMap contributors for ground truth data
- Mapbox for satellite imagery
- PyTorch and torchvision teams
- U-Net architecture by Ronneberger et al.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{osmid_ai_editor,
  title={osmiD-AI-editor: Machine Learning-Based Automated OpenStreetMap Feature Mapper},
  author={Yinhao Wang},
  year={2025},
  url={https://github.com/michaelisadullguy-star/osmiD-AI-editor}
}
```

## Contact

For questions or issues, please open an issue on GitHub.

---

**Disclaimer**: This tool is for educational and research purposes. Always review and validate automated edits before uploading to OpenStreetMap.
