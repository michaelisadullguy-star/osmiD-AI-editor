"""
Input validation utilities for security and data integrity
"""

import os
import re
from typing import List
from pathlib import Path
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from .coordinates import LatLon, BoundingBox


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude bounds

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        True if valid

    Raises:
        ValueError: If coordinates are out of bounds
    """
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat} (must be -90 to 90)")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon} (must be -180 to 180)")
    return True


def validate_polygon(coords: List[LatLon], min_points: int = 3) -> Polygon:
    """
    Validate polygon geometry

    Args:
        coords: List of LatLon coordinates
        min_points: Minimum number of points required

    Returns:
        Validated Shapely Polygon

    Raises:
        ValueError: If polygon is invalid
    """
    if len(coords) < min_points:
        raise ValueError(f"Polygon must have at least {min_points} points, got {len(coords)}")

    # Validate each coordinate
    for coord in coords:
        coord.validate()

    # Create polygon (GeoJSON format: [lon, lat])
    geojson_coords = [coord.to_geojson() for coord in coords]

    # Ensure polygon is closed
    if geojson_coords[0] != geojson_coords[-1]:
        geojson_coords.append(geojson_coords[0])

    poly = Polygon(geojson_coords)

    if not poly.is_valid:
        error_msg = explain_validity(poly)
        raise ValueError(f"Invalid polygon: {error_msg}")

    if poly.area == 0:
        raise ValueError("Polygon has zero area")

    # Check for self-intersection
    if not poly.is_simple:
        raise ValueError("Polygon is self-intersecting")

    return poly


def validate_bounding_box(bbox: BoundingBox) -> BoundingBox:
    """
    Validate bounding box

    Args:
        bbox: BoundingBox to validate

    Returns:
        Validated bounding box

    Raises:
        ValueError: If bounding box is invalid
    """
    return bbox.validate()


def validate_file_path(
    file_path: str,
    allowed_dirs: List[str],
    allowed_extensions: List[str] = None,
    must_exist: bool = False
) -> Path:
    """
    Validate file path for security (prevent path traversal)

    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed directory prefixes
        allowed_extensions: List of allowed file extensions (e.g., ['.pth', '.pt'])
        must_exist: Whether the file must exist

    Returns:
        Validated absolute Path object

    Raises:
        ValueError: If path is invalid or insecure
    """
    # Convert to absolute path
    abs_path = Path(file_path).resolve()

    # Check if path is within allowed directories
    is_allowed = False
    for allowed_dir in allowed_dirs:
        allowed_abs = Path(allowed_dir).resolve()
        try:
            abs_path.relative_to(allowed_abs)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        allowed_str = ", ".join(allowed_dirs)
        raise ValueError(f"File path must be within allowed directories: {allowed_str}")

    # Check file extension
    if allowed_extensions and abs_path.suffix not in allowed_extensions:
        ext_str = ", ".join(allowed_extensions)
        raise ValueError(f"File must have one of these extensions: {ext_str}")

    # Check existence
    if must_exist and not abs_path.exists():
        raise ValueError(f"File does not exist: {abs_path}")

    return abs_path


def validate_email(email: str) -> bool:
    """
    Validate email address format

    Args:
        email: Email address to validate

    Returns:
        True if valid

    Raises:
        ValueError: If email is invalid
    """
    if not email or len(email) > 254:
        raise ValueError("Invalid email address")

    # Basic email validation regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError(f"Invalid email format: {email}")

    return True


def validate_model_path(model_path: str) -> Path:
    """
    Validate model checkpoint path

    Args:
        model_path: Path to model checkpoint

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid
    """
    return validate_file_path(
        model_path,
        allowed_dirs=['./models/checkpoints', './models'],
        allowed_extensions=['.pth', '.pt'],
        must_exist=True
    )


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem use

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Remove path separators and other dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')

    # Limit length
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        max_name_len = max_length - len(ext)
        sanitized = name[:max_name_len] + ext

    # Ensure it's not empty
    if not sanitized:
        sanitized = 'unnamed'

    return sanitized
