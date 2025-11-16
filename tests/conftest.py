"""
Pytest configuration and fixtures for osmiD-AI-editor tests
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from common.coordinates import LatLon, LonLat, BoundingBox


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def paris_bbox():
    """Paris bounding box for testing"""
    return BoundingBox(
        south=48.815573,
        west=2.224199,
        north=48.902145,
        east=2.469920
    )


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing"""
    return [
        LatLon(lat=48.8566, lon=2.3522),  # Eiffel Tower
        LatLon(lat=48.8606, lon=2.3376),  # Arc de Triomphe
        LatLon(lat=48.8530, lon=2.3499),  # Notre-Dame
        LatLon(lat=48.8566, lon=2.3522),  # Close polygon
    ]


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    return np.random.rand(512, 512, 3).astype(np.float32)


@pytest.fixture
def sample_mask():
    """Create a sample segmentation mask"""
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Add some features
    mask[100:200, 100:200] = 1  # Building
    mask[300:400, 300:400] = 2  # Lawn
    return mask
