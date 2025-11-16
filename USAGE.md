# Usage Guide

Complete guide for using the osmiD-AI-editor system.

## Quick Start

### 1. Data Acquisition (Part 1)

Download training data for the 7 major cities:

```bash
python run_data_acquisition.py
```

This will:
- Download OSM data from Overpass API (~10-30 minutes)
- Download satellite imagery from Mapbox (~15-45 minutes)
- Correlate features with imagery (~5-15 minutes)

**Total time**: 30-90 minutes depending on your internet connection

**Output**:
- `./data/osm/` - GeoJSON files with OSM features
- `./data/imagery/` - Satellite imagery (PNG files)
- `./data/correlated/` - Training pairs (masks and overlays)

### 2. Training (Part 2)

Train the AI model:

```bash
python run_training.py
```

**Options**:
```bash
python run_training.py --batch-size 8 --epochs 100 --lr 0.001
```

**Expected time**:
- With GPU (NVIDIA RTX 3080+): 8-12 hours
- With GPU (older models): 16-24 hours
- CPU only: 3-7 days (not recommended)

**Monitor training**:
```bash
tensorboard --logdir=./logs
```
Then open http://localhost:6006 in your browser

**Output**:
- `./models/checkpoints/checkpoint_best.pth` - Best model
- `./models/checkpoints/checkpoint_latest.pth` - Latest checkpoint
- `./logs/` - TensorBoard logs

### 3. Running the Application (Part 3)

#### Option A: Run as Python application

```bash
python run_gui.py
```

#### Option B: Build standalone executable

```bash
python part3_executable/build_executable.py
```

The executable will be created at: `./dist/osmiD-AI-editor.exe`

## Detailed Usage

### Part 1: Data Acquisition

#### Download specific cities only

```python
from part1_data_acquisition.osm_downloader import OSMDownloader

downloader = OSMDownloader(output_dir='./data/osm')
downloader.download_city_data('paris')
downloader.download_city_data('london')
```

#### Download specific features only

```python
from part1_data_acquisition.osm_downloader import OSMDownloader

downloader = OSMDownloader()
downloader.download_city_data(
    'paris',
    features=['buildings', 'water_body']  # Only buildings and water
)
```

#### Custom bounding box

```python
from part1_data_acquisition.mapbox_imagery import MapboxImageryDownloader

downloader = MapboxImageryDownloader()
custom_bbox = [48.85, 2.29, 48.87, 2.40]  # [south, west, north, east]
downloader.download_bbox_imagery(custom_bbox, 'custom_area', zoom=17)
```

### Part 2: Training

#### Resume from checkpoint

```python
from part2_training.train import Trainer

config = {
    'data_dir': './data',
    'checkpoint_dir': './models/checkpoints',
    # ... other config ...
}

trainer = Trainer(config)

# Load checkpoint
checkpoint = torch.load('./models/checkpoints/checkpoint_latest.pth')
trainer.model.load_state_dict(checkpoint['model_state_dict'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.epoch = checkpoint['epoch']
trainer.best_val_loss = checkpoint['best_val_loss']

# Continue training
trainer.train()
```

#### Custom training configuration

Create a `custom_config.yaml`:

```yaml
data_dir: ./data
checkpoint_dir: ./models/checkpoints
log_dir: ./logs

train_cities:
  - paris
  - london

val_cities:
  - tokyo

batch_size: 16
epochs: 50
learning_rate: 0.0005
img_size: [256, 256]
```

Then run:
```bash
python run_training.py --config custom_config.yaml
```

#### Transfer learning from pretrained model

```python
from part2_training.model import FeatureSegmentationModel

model = FeatureSegmentationModel(n_channels=3, n_classes=6)

# Load pretrained weights
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Freeze encoder layers
for param in model.unet.inc.parameters():
    param.requires_grad = False
for param in model.unet.down1.parameters():
    param.requires_grad = False

# Train only decoder and heads
# ... continue with training ...
```

### Part 3: Using the Application

#### Application Interface

The GUI has three main sections:

1. **Login & Configuration**
   - **Email**: Your OSM account email
   - **Password**: Your OSM account password
   - **Polygon Coordinates**: Area to map

2. **Configuration**
   - **Model Path**: Path to trained model checkpoint
   - **Mapbox Token**: Your Mapbox access token

3. **Output Log**
   - Shows real-time progress
   - Displays detected features
   - Shows upload status

#### Polygon Format

The polygon coordinates should be entered as:

```
{{lat1,lon1},{lat2,lon2},{lat3,lon3},...,{latN,lonN}}
```

**Example** (area in Paris):
```
{{48.8566,2.3522},{48.8577,2.3540},{48.8560,2.3545},{48.8555,2.3530}}
```

**Tips**:
- Coordinates should form a closed polygon
- Use at least 3 points
- Keep polygon size reasonable (< 1 km²)
- Use latitude, longitude format (not longitude, latitude)

#### Getting Polygon Coordinates

**Method 1: OpenStreetMap**
1. Go to [openstreetmap.org](https://www.openstreetmap.org/)
2. Navigate to your area of interest
3. Click on desired points
4. Note the coordinates from the URL or info panel

**Method 2: GeoJSON.io**
1. Go to [geojson.io](http://geojson.io/)
2. Draw a polygon on the map
3. Copy coordinates from the GeoJSON output

**Method 3: Bounding Box Tool**
1. Go to [boundingbox.klokantech.com](http://boundingbox.klokantech.com/)
2. Select area
3. Copy coordinates in CSV format
4. Convert to required format

#### Programmatic Usage

You can also use the detector programmatically:

```python
from part3_executable.feature_detector import FeatureDetector
from part3_executable.osm_client import OSMClient
import numpy as np
from PIL import Image

# Load model
detector = FeatureDetector('./models/checkpoints/checkpoint_best.pth')

# Load imagery
image = np.array(Image.open('satellite_image.png'))
bbox = [48.85, 2.29, 48.87, 2.40]

# Detect features
features = detector.detect_features_in_polygon(image, bbox)

# Upload to OSM
client = OSMClient()
client.authenticate('email@example.com', 'password')
way_ids = client.upload_features(features)

print(f"Uploaded {len(way_ids)} features")
```

## Advanced Usage

### Batch Processing Multiple Areas

```python
from part3_executable.feature_detector import FeatureDetector
from part3_executable.osm_client import OSMClient

detector = FeatureDetector('./models/checkpoints/checkpoint_best.pth')
client = OSMClient()
client.authenticate('email', 'password')

# Define multiple areas
areas = [
    {'name': 'area1', 'bbox': [48.85, 2.29, 48.87, 2.40]},
    {'name': 'area2', 'bbox': [48.86, 2.30, 48.88, 2.42]},
]

for area in areas:
    # Download imagery for area
    # ... download code ...

    # Detect features
    features = detector.detect_features_in_polygon(image, area['bbox'])

    # Upload
    client.upload_features(
        features,
        changeset_comment=f"AI mapping for {area['name']}"
    )
```

### Custom Feature Types

To add support for new feature types, modify:

1. **part1_data_acquisition/osm_downloader.py**:
```python
feature_queries = {
    'buildings': f'way["building"]({bbox_str});',
    'parks': f'way["leisure"="park"]({bbox_str});',  # Add new feature
}
```

2. **part2_training/model.py**: Update `n_classes`

3. **part3_executable/feature_detector.py**:
```python
FEATURE_CLASSES = {
    0: 'background',
    1: 'building',
    2: 'lawn',
    3: 'natural_wood',
    4: 'artificial_forest',
    5: 'water_body',
    6: 'park',  # Add new class
}
```

4. **part3_executable/osm_client.py**: Add new tags

## Best Practices

### Before Uploading to OSM

1. **Test on small areas first**
2. **Review detected features** in the log output
3. **Verify model accuracy** on validation imagery
4. **Check OSM wiki** for proper tagging
5. **Announce your edit** on OSM community forums
6. **Monitor feedback** after upload

### Model Training Tips

1. **Use validation loss** to prevent overfitting
2. **Monitor both losses** (segmentation and contour)
3. **Experiment with loss weights** (seg_weight, contour_weight)
4. **Use data augmentation** for better generalization
5. **Save checkpoints frequently**

### Performance Optimization

1. **Use GPU** for training (10-50x faster)
2. **Increase batch size** if you have enough VRAM
3. **Use mixed precision training** for faster training
4. **Reduce image size** if running out of memory
5. **Use multiple workers** for data loading

## Troubleshooting Common Issues

See [README.md#troubleshooting](README.md#troubleshooting) for common issues and solutions.

## Examples

### Example 1: Map a small neighborhood

```bash
# 1. Download data
python run_data_acquisition.py

# 2. Train model
python run_training.py --epochs 50

# 3. Launch GUI
python run_gui.py

# 4. In GUI:
# Email: your_email@example.com
# Password: your_password
# Polygon: {{48.8566,2.3522},{48.8577,2.3540},{48.8560,2.3545},{48.8555,2.3530}}
# Click "Start Feature Mapping"
```

### Example 2: Fine-tune on custom area

```python
# Add your custom area to training data
from part1_data_acquisition.osm_downloader import OSMDownloader

downloader = OSMDownloader()
custom_bbox = [your_south, your_west, your_north, your_east]
# Download data for custom area
# ... add to training pipeline ...

# Fine-tune model
# ... train with custom data included ...
```

## Next Steps

- Read [INSTALL.md](INSTALL.md) for installation details
- Check [README.md](README.md) for overview
- Review code documentation in each module
- Join OSM community for support
