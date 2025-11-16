"""
Coordinate handling with explicit types to prevent lat/lon confusion
Addresses critical bug where coordinate order could map features incorrectly
"""

from typing import NamedTuple, List, Tuple
import numpy as np


class LatLon(NamedTuple):
    """
    Latitude/Longitude pair (OSM/API standard order)

    This is the standard for:
    - OpenStreetMap API
    - Most geographic databases
    - Human-readable coordinates

    Example: Paris = LatLon(lat=48.8566, lon=2.3522)
    """
    lat: float
    lon: float

    def to_geojson(self) -> List[float]:
        """Convert to GeoJSON [lon, lat] format"""
        return [self.lon, self.lat]

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (lat, lon)"""
        return (self.lat, self.lon)

    def validate(self) -> 'LatLon':
        """Validate coordinate bounds"""
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"Invalid latitude: {self.lat} (must be -90 to 90)")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"Invalid longitude: {self.lon} (must be -180 to 180)")
        return self

    def __str__(self) -> str:
        return f"({self.lat:.6f}, {self.lon:.6f})"


class LonLat(NamedTuple):
    """
    Longitude/Latitude pair (GeoJSON standard order)

    This is the standard for:
    - GeoJSON format
    - Most mapping libraries
    - Shapely geometries

    Example: Paris = LonLat(lon=2.3522, lat=48.8566)
    """
    lon: float
    lat: float

    def to_latlon(self) -> LatLon:
        """Convert to LatLon (OSM/API format)"""
        return LatLon(lat=self.lat, lon=self.lon)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (lon, lat)"""
        return (self.lon, self.lat)

    def validate(self) -> 'LonLat':
        """Validate coordinate bounds"""
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"Invalid latitude: {self.lat} (must be -90 to 90)")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"Invalid longitude: {self.lon} (must be -180 to 180)")
        return self

    def __str__(self) -> str:
        return f"[{self.lon:.6f}, {self.lat:.6f}]"


class BoundingBox(NamedTuple):
    """
    Geographic bounding box with explicit component names
    Order: [south, west, north, east] (min_lat, min_lon, max_lat, max_lon)
    """
    south: float  # min latitude
    west: float   # min longitude
    north: float  # max latitude
    east: float   # max longitude

    def to_list(self) -> List[float]:
        """Convert to list [south, west, north, east]"""
        return [self.south, self.west, self.north, self.east]

    def validate(self) -> 'BoundingBox':
        """Validate bounding box"""
        if not (-90 <= self.south <= 90):
            raise ValueError(f"Invalid south latitude: {self.south}")
        if not (-90 <= self.north <= 90):
            raise ValueError(f"Invalid north latitude: {self.north}")
        if not (-180 <= self.west <= 180):
            raise ValueError(f"Invalid west longitude: {self.west}")
        if not (-180 <= self.east <= 180):
            raise ValueError(f"Invalid east longitude: {self.east}")

        if self.south >= self.north:
            raise ValueError(f"South ({self.south}) must be less than north ({self.north})")
        if self.west >= self.east:
            raise ValueError(f"West ({self.west}) must be less than east ({self.east})")

        return self

    def center(self) -> LatLon:
        """Get center point of bounding box"""
        center_lat = (self.south + self.north) / 2
        center_lon = (self.west + self.east) / 2
        return LatLon(lat=center_lat, lon=center_lon)

    def contains(self, point: LatLon) -> bool:
        """Check if point is within bounding box"""
        return (self.south <= point.lat <= self.north and
                self.west <= point.lon <= self.east)

    def area_degrees(self) -> float:
        """Calculate area in square degrees"""
        return (self.north - self.south) * (self.east - self.west)

    @classmethod
    def from_list(cls, bbox: List[float]) -> 'BoundingBox':
        """Create from list [south, west, north, east]"""
        if len(bbox) != 4:
            raise ValueError(f"Bounding box must have 4 values, got {len(bbox)}")
        return cls(south=bbox[0], west=bbox[1], north=bbox[2], east=bbox[3])

    @classmethod
    def from_points(cls, points: List[LatLon]) -> 'BoundingBox':
        """Create bounding box from list of points"""
        if not points:
            raise ValueError("Cannot create bounding box from empty point list")

        lats = [p.lat for p in points]
        lons = [p.lon for p in points]

        return cls(
            south=min(lats),
            west=min(lons),
            north=max(lats),
            east=max(lons)
        )

    def __str__(self) -> str:
        return f"BBox(S:{self.south:.4f}, W:{self.west:.4f}, N:{self.north:.4f}, E:{self.east:.4f})"


def latlon_to_pixel(
    coord: LatLon,
    bbox: BoundingBox,
    img_width: int,
    img_height: int
) -> Tuple[int, int]:
    """
    Convert lat/lon coordinates to pixel coordinates in image

    Args:
        coord: Latitude/Longitude coordinate
        bbox: Bounding box of the image
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x, y) pixel coordinates
    """
    # Normalize coordinates to [0, 1]
    x_norm = (coord.lon - bbox.west) / (bbox.east - bbox.west)
    y_norm = (bbox.north - coord.lat) / (bbox.north - bbox.south)  # Inverted for image coordinates

    # Convert to pixel coordinates
    x_pixel = int(x_norm * img_width)
    y_pixel = int(y_norm * img_height)

    # Clamp to image bounds
    x_pixel = max(0, min(img_width - 1, x_pixel))
    y_pixel = max(0, min(img_height - 1, y_pixel))

    return (x_pixel, y_pixel)


def pixel_to_latlon(
    pixel_coords: np.ndarray,
    bbox: BoundingBox,
    img_width: int,
    img_height: int
) -> List[LatLon]:
    """
    Convert pixel coordinates to lat/lon

    Args:
        pixel_coords: Pixel coordinates as (N, 2) array [x, y]
        bbox: Bounding box of the image
        img_width: Image width
        img_height: Image height

    Returns:
        List of LatLon coordinates
    """
    coords = []

    for x, y in pixel_coords:
        # Normalize to [0, 1]
        x_norm = x / img_width
        y_norm = y / img_height

        # Convert to lat/lon
        lon = bbox.west + x_norm * (bbox.east - bbox.west)
        lat = bbox.north - y_norm * (bbox.north - bbox.south)  # Inverted for image coordinates

        coords.append(LatLon(lat=lat, lon=lon))

    return coords
