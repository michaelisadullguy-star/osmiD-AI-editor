# Installation Guide

This guide will walk you through setting up the osmiD-AI-editor project.

## System Requirements

### Minimum Requirements
- Operating System: Windows 10/11, Linux, or macOS
- Python: 3.8 or higher
- RAM: 8 GB
- Storage: 50 GB free space (for training data and models)
- Internet connection (for downloading data)

### Recommended Requirements
- Operating System: Ubuntu 20.04+ or Windows 10/11
- Python: 3.9 or 3.10
- RAM: 16 GB or more
- GPU: NVIDIA GPU with 8GB+ VRAM (for faster training)
- Storage: 100 GB+ SSD
- Fast internet connection

## Step-by-Step Installation

### 1. Install Python

#### Windows
Download and install Python from [python.org](https://www.python.org/downloads/)

Make sure to check "Add Python to PATH" during installation.

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### macOS
```bash
brew install python@3.9
```

### 2. Clone the Repository

```bash
git clone https://github.com/michaelisadullguy-star/osmiD-AI-editor.git
cd osmiD-AI-editor
```

### 3. Create Virtual Environment (Recommended)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you have a CUDA-compatible GPU, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API tokens:

```bash
# Mapbox API Configuration
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here

# OSM API Configuration (default is fine)
OSM_API_URL=https://api.openstreetmap.org/api/0.6
```

### 6. Get API Keys

#### Mapbox Access Token
1. Sign up at [mapbox.com](https://www.mapbox.com/)
2. Go to [Account > Access tokens](https://account.mapbox.com/access-tokens/)
3. Create a new token or copy your default public token
4. Add it to your `.env` file

#### OpenStreetMap Account
1. Create account at [openstreetmap.org](https://www.openstreetmap.org/user/new)
2. Confirm your email
3. Read the [Automated Edits Code of Conduct](https://wiki.openstreetmap.org/wiki/Automated_Edits_code_of_conduct)
4. You'll use these credentials in the GUI application

### 7. Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; import cv2; import osmapi; print('✓ All dependencies installed successfully')"
```

## Optional: Install Development Tools

### TensorBoard (for monitoring training)
```bash
pip install tensorboard
```

### Jupyter Notebook (for data exploration)
```bash
pip install jupyter
```

## Troubleshooting

### Issue: PyTorch CUDA not available

**Solution**: Make sure you have NVIDIA drivers installed and install PyTorch with CUDA support:

```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: opencv-python installation fails

**Solution**: Install system dependencies first:

**Ubuntu/Debian:**
```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
```

**Fedora/CentOS:**
```bash
sudo yum install mesa-libGL
```

### Issue: osmapi installation fails

**Solution**: Upgrade pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
pip install osmapi
```

### Issue: PyQt5 installation fails on Linux

**Solution**: Install Qt dependencies:

**Ubuntu/Debian:**
```bash
sudo apt install python3-pyqt5 libxcb-xinerama0
```

**Fedora:**
```bash
sudo dnf install python3-qt5
```

## Next Steps

After installation:

1. **Download training data**: `python run_data_acquisition.py`
2. **Train the model**: `python run_training.py`
3. **Launch GUI**: `python run_gui.py`

See [README.md](README.md) for detailed usage instructions.

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting section in README.md](README.md#troubleshooting)
2. Search existing [GitHub Issues](https://github.com/michaelisadullguy-star/osmiD-AI-editor/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```
