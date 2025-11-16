"""
Centralized constants for osmiD-AI-editor
Eliminates code duplication across modules
"""

from typing import Dict, List

# City bounding boxes [south, west, north, east]
CITIES: Dict[str, List[float]] = {
    'paris': [48.815573, 2.224199, 48.902145, 2.469920],
    'london': [51.286760, -0.510375, 51.691874, 0.334015],
    'new_york': [40.477399, -74.259090, 40.917577, -73.700272],
    'hong_kong': [22.153689, 113.835079, 22.561968, 114.406844],
    'moscow': [55.491878, 37.319336, 55.957565, 37.967987],
    'tokyo': [35.528874, 139.560547, 35.817813, 139.910278],
    'singapore': [1.205764, 103.604736, 1.470974, 104.028320]
}

# Feature class mapping
FEATURE_CLASSES: Dict[int, str] = {
    0: 'background',
    1: 'building',
    2: 'lawn',
    3: 'natural_wood',
    4: 'artificial_forest',
    5: 'water_body'
}

# Reverse mapping: name to class ID
FEATURE_CLASS_NAMES: Dict[str, int] = {
    name: class_id for class_id, name in FEATURE_CLASSES.items()
}

# Zoom level to meters per pixel resolution
ZOOM_LEVEL_RESOLUTIONS: Dict[int, float] = {
    14: 4.8,   # meters per pixel
    15: 2.4,
    16: 1.2,
    17: 0.6,
    18: 0.3,
    19: 0.15,
    20: 0.075
}

# Default configuration values
DEFAULT_ZOOM: int = 17
CONTOUR_POINT_DISTANCE_METERS: float = 3.0
MIN_FEATURE_AREA_PIXELS: int = 100
DEFAULT_IMAGE_SIZE: tuple = (512, 512)

# OSM feature tags mapping
OSM_FEATURE_TAGS: Dict[str, Dict[str, str]] = {
    'building': {'building': 'yes'},
    'lawn': {'landuse': 'grass', 'grass': 'lawn'},
    'natural_wood': {'natural': 'wood'},
    'artificial_forest': {'landuse': 'forest'},
    'water_body': {'natural': 'water'}
}

# API configuration
OSM_API_BASE_URL: str = "https://api.openstreetmap.org"
OSM_API_VERSION: str = "0.6"
MAPBOX_SATELLITE_STYLE: str = "mapbox/satellite-v9"

# Rate limiting (seconds)
OSM_API_RATE_LIMIT_DELAY: float = 0.5
OSM_API_NODE_CREATE_DELAY: float = 0.1
OVERPASS_API_RATE_LIMIT_DELAY: float = 2.0

# Retry configuration
MAX_RETRIES: int = 4
RETRY_BASE_DELAY: float = 2.0

# Application metadata
APP_NAME: str = "osmiD-AI-editor"
APP_VERSION: str = "1.0.0"
APP_USER_AGENT: str = f"{APP_NAME}/{APP_VERSION}"

# Allowed file paths for security
ALLOWED_MODEL_DIR: str = "./models/checkpoints"
ALLOWED_DATA_DIR: str = "./data"
