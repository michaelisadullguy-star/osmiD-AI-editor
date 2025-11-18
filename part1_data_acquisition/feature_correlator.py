"""
Part 1: Feature Correlation System
Correlates features in satellite imagery with corresponding OSM objects
"""

import json
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point, shape
from shapely.ops import transform
import pyproj
from typing import Dict, List, Tuple
import os


class FeatureCorrelator:
    """Correlate satellite imagery features with OSM data"""

    def __init__(self, osm_data_dir: str = './data/osm', imagery_dir: str = './data/imagery'):
        """
        Initialize feature correlator

        Args:
            osm_data_dir: Directory containing OSM GeoJSON files
            imagery_dir: Directory containing satellite imagery
        """
        self.osm_data_dir = osm_data_dir
        self.imagery_dir = imagery_dir

    def load_osm_data(self, city: str) -> Dict:
        """
        Load OSM GeoJSON data for a city

        Args:
            city: City name

        Returns:
            GeoJSON dictionary
        """
        filepath = os.path.join(self.osm_data_dir, f"{city}.geojson")
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_imagery(self, city: str, zoom: int = 16) -> np.ndarray:
        """
        Load satellite imagery for a city

        Args:
            city: City name
            zoom: Zoom level

        Returns:
            Image as numpy array
        """
        filepath = os.path.join(self.imagery_dir, f"{city}_z{zoom}.png")
        img = Image.open(filepath)
        return np.array(img)

    def latlon_to_pixel(
        self,
        lat: float,
        lon: float,
        bbox: List[float],
        img_width: int,
        img_height: int
    ) -> Tuple[int, int]:
        """
        Convert lat/lon coordinates to pixel coordinates in image

        Args:
            lat: Latitude
            lon: Longitude
            bbox: Bounding box [south, west, north, east]
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        south, west, north, east = bbox

        # Normalize coordinates to [0, 1]
        x_norm = (lon - west) / (east - west)
        y_norm = (north - lat) / (north - south)  # Inverted for image coordinates

        # Convert to pixel coordinates
        x_pixel = int(x_norm * img_width)
        y_pixel = int(y_norm * img_height)

        return (x_pixel, y_pixel)

    def create_feature_mask(
        self,
        geojson: Dict,
        bbox: List[float],
        img_width: int,
        img_height: int,
        feature_type: str = None
    ) -> np.ndarray:
        """
        Create a binary mask from OSM features

        Args:
            geojson: GeoJSON data
            bbox: Bounding box [south, west, north, east]
            img_width: Image width
            img_height: Image height
            feature_type: Specific feature type to mask (None for all)

        Returns:
            Binary mask as numpy array
        """
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for feature in geojson['features']:
            # Filter by feature type if specified
            if feature_type and feature['properties']['feature_type'] != feature_type:
                continue

            # Get geometry
            geom = shape(feature['geometry'])

            # Convert coordinates to pixels
            if geom.geom_type == 'Polygon':
                exterior_coords = list(geom.exterior.coords)
                pixel_coords = [
                    self.latlon_to_pixel(lat, lon, bbox, img_width, img_height)
                    for lon, lat in exterior_coords
                ]

                # Draw filled polygon on mask
                pts = np.array(pixel_coords, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

        return mask

    def correlate_features(
        self,
        city: str,
        bbox: List[float],
        zoom: int = 16,
        output_dir: str = './data/correlated'
    ) -> Dict[str, np.ndarray]:
        """
        Correlate OSM features with satellite imagery

        Args:
            city: City name
            bbox: Bounding box [south, west, north, east]
            zoom: Zoom level
            output_dir: Directory to save correlated data

        Returns:
            Dictionary mapping feature types to masks
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Correlating features for {city}...")

        # Load data
        osm_data = self.load_osm_data(city)
        imagery = self.load_imagery(city, zoom)

        img_height, img_width = imagery.shape[:2]

        # Feature types to process
        feature_types = ['building', 'lawn', 'natural_wood', 'artificial_forest', 'water_body', 'farmland']

        masks = {}

        for feature_type in feature_types:
            print(f"  Creating mask for {feature_type}...")

            # Create feature mask
            mask = self.create_feature_mask(
                osm_data,
                bbox,
                img_width,
                img_height,
                feature_type
            )

            masks[feature_type] = mask

            # Save mask
            mask_path = os.path.join(output_dir, f"{city}_{feature_type}_mask.png")
            cv2.imwrite(mask_path, mask)

            # Create overlay visualization
            overlay = imagery.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

            overlay_path = os.path.join(output_dir, f"{city}_{feature_type}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Save complete training pair
        training_data = {
            'city': city,
            'bbox': bbox,
            'zoom': zoom,
            'imagery_path': os.path.join(self.imagery_dir, f"{city}_z{zoom}.png"),
            'feature_counts': {
                ft: int(np.sum(masks[ft] > 0))
                for ft in feature_types
            }
        }

        training_data_path = os.path.join(output_dir, f"{city}_training_data.json")
        with open(training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"✓ Correlated {len(feature_types)} feature types for {city}")
        print(f"  Feature counts: {training_data['feature_counts']}")

        return masks

    def create_training_dataset(
        self,
        cities: List[str],
        cities_bbox: Dict[str, List[float]],
        zoom: int = 16
    ):
        """
        Create complete training dataset for all cities

        Args:
            cities: List of city names
            cities_bbox: Dictionary mapping city names to bounding boxes
            zoom: Zoom level
        """
        for city in cities:
            if city not in cities_bbox:
                print(f"Warning: No bounding box for {city}, skipping...")
                continue

            try:
                self.correlate_features(city, cities_bbox[city], zoom)
            except Exception as e:
                print(f"✗ Error correlating features for {city}: {str(e)}")
                continue


if __name__ == "__main__":
    CITIES = {
        'paris': [48.815573, 2.224199, 48.902145, 2.469920],
        'london': [51.286760, -0.510375, 51.691874, 0.334015],
        'new_york': [40.477399, -74.259090, 40.917577, -73.700272],
        'hong_kong': [22.153689, 113.835079, 22.561968, 114.406844],
        'moscow': [55.491878, 37.319336, 55.957565, 37.967987],
        'tokyo': [35.528874, 139.560547, 35.817813, 139.910278],
        'singapore': [1.205764, 103.604736, 1.470974, 104.028320]
    }

    correlator = FeatureCorrelator()
    correlator.create_training_dataset(list(CITIES.keys()), CITIES, zoom=16)
