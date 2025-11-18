"""
Part 3: Feature Detection and Mapping
Uses trained model to detect features and convert to OSM coordinates
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict
from shapely.geometry import Polygon, shape
from shapely.ops import transform
import pyproj
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part2_training.model import FeatureSegmentationModel


class FeatureDetector:
    """Detect features in satellite imagery using trained model"""

    FEATURE_CLASSES = {
        0: 'background',
        1: 'building',
        2: 'lawn',
        3: 'natural_wood',
        4: 'artificial_forest',
        5: 'water_body',
        6: 'farmland'
    }

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize feature detector

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = FeatureSegmentationModel(
            n_channels=3,
            n_classes=checkpoint['config'].get('n_classes', 7),
            bilinear=checkpoint['config'].get('bilinear', False)
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Loaded model from {model_path}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input

        Args:
            image: Input image as numpy array (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std

        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return image.float()

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict features in image

        Args:
            image: Input image as numpy array (H, W, 3)

        Returns:
            Dictionary with segmentation and contour predictions
        """
        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Post-process
        seg_pred = torch.argmax(predictions['segmentation'], dim=1).squeeze().cpu().numpy()
        contour_pred = torch.argmax(predictions['contours'], dim=1).squeeze().cpu().numpy()

        return {
            'segmentation': seg_pred,
            'contours': contour_pred
        }

    def extract_contours(
        self,
        mask: np.ndarray,
        class_id: int,
        min_area: int = 100
    ) -> List[np.ndarray]:
        """
        Extract contours for a specific class

        Args:
            mask: Segmentation mask
            class_id: Class ID to extract
            min_area: Minimum contour area in pixels

        Returns:
            List of contours as numpy arrays
        """
        # Create binary mask for class
        binary_mask = (mask == class_id).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area
        filtered_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cnt) >= min_area
        ]

        return filtered_contours

    def simplify_contour(
        self,
        contour: np.ndarray,
        point_distance: float = 3.0,
        pixel_resolution: float = 0.6
    ) -> np.ndarray:
        """
        Simplify contour to have points approximately every 3 meters

        Args:
            contour: Input contour
            point_distance: Desired distance between points in meters (default 3m)
            pixel_resolution: Meters per pixel (default 0.6m at zoom 17)

        Returns:
            Simplified contour
        """
        # Calculate epsilon for Douglas-Peucker algorithm
        epsilon = point_distance / pixel_resolution

        # Simplify contour
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

        return simplified

    def pixel_to_latlon(
        self,
        pixel_coords: np.ndarray,
        bbox: List[float],
        img_width: int,
        img_height: int
    ) -> List[Tuple[float, float]]:
        """
        Convert pixel coordinates to lat/lon

        Args:
            pixel_coords: Pixel coordinates (N, 2) array
            bbox: Bounding box [south, west, north, east]
            img_width: Image width
            img_height: Image height

        Returns:
            List of (lat, lon) tuples
        """
        south, west, north, east = bbox

        latlon_coords = []

        for x, y in pixel_coords:
            # Normalize to [0, 1]
            x_norm = x / img_width
            y_norm = y / img_height

            # Convert to lat/lon
            lon = west + x_norm * (east - west)
            lat = north - y_norm * (north - south)  # Inverted for image coordinates

            latlon_coords.append((lat, lon))

        return latlon_coords

    def detect_features_in_polygon(
        self,
        image: np.ndarray,
        polygon_bbox: List[float],
        min_area: int = 100
    ) -> List[Dict]:
        """
        Detect all features within a specified polygon

        Args:
            image: Satellite imagery
            polygon_bbox: Bounding box [south, west, north, east]
            min_area: Minimum feature area in pixels

        Returns:
            List of detected features with coordinates
        """
        print("Running feature detection...")

        # Predict
        predictions = self.predict(image)
        mask = predictions['segmentation']

        img_height, img_width = mask.shape

        # Extract features for each class
        detected_features = []

        for class_id, class_name in self.FEATURE_CLASSES.items():
            if class_id == 0:  # Skip background
                continue

            print(f"  Extracting {class_name} features...")

            # Extract contours
            contours = self.extract_contours(mask, class_id, min_area)

            # Process each contour
            for contour in contours:
                # Simplify to 3m spacing
                simplified = self.simplify_contour(contour)

                # Convert to lat/lon
                pixel_coords = simplified.squeeze()

                if len(pixel_coords.shape) == 1:
                    pixel_coords = pixel_coords.reshape(-1, 2)

                latlon_coords = self.pixel_to_latlon(
                    pixel_coords,
                    polygon_bbox,
                    img_width,
                    img_height
                )

                detected_features.append({
                    'type': class_name,
                    'coordinates': latlon_coords,
                    'area': cv2.contourArea(contour)
                })

        print(f"✓ Detected {len(detected_features)} features")

        # Print feature summary
        feature_counts = {}
        for feature in detected_features:
            feature_type = feature['type']
            feature_counts[feature_type] = feature_counts.get(feature_type, 0) + 1

        for feature_type, count in feature_counts.items():
            print(f"    {feature_type}: {count}")

        return detected_features


if __name__ == "__main__":
    # Test detector
    detector = FeatureDetector('./models/checkpoints/checkpoint_best.pth')

    # Load test image
    test_image = np.random.rand(512, 512, 3) * 255
    test_bbox = [48.8, 2.3, 48.9, 2.4]

    features = detector.detect_features_in_polygon(test_image, test_bbox)
    print(f"Detected {len(features)} features")
