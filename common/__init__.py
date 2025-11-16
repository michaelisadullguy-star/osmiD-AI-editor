"""
Common utilities and shared modules for osmiD-AI-editor
"""

from .constants import (
    CITIES,
    FEATURE_CLASSES,
    ZOOM_LEVEL_RESOLUTIONS,
    DEFAULT_ZOOM,
    CONTOUR_POINT_DISTANCE_METERS
)
from .coordinates import LatLon, LonLat, BoundingBox
from .validation import (
    validate_coordinates,
    validate_polygon,
    validate_file_path,
    validate_email
)
from .logging_config import setup_logging

__all__ = [
    'CITIES',
    'FEATURE_CLASSES',
    'ZOOM_LEVEL_RESOLUTIONS',
    'DEFAULT_ZOOM',
    'CONTOUR_POINT_DISTANCE_METERS',
    'LatLon',
    'LonLat',
    'BoundingBox',
    'validate_coordinates',
    'validate_polygon',
    'validate_file_path',
    'validate_email',
    'setup_logging',
]
