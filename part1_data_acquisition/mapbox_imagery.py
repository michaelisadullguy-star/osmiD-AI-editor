"""
Part 1: Mapbox Satellite Imagery Downloader
Downloads satellite imagery tiles for specified bounding boxes using Mapbox Static Images API
"""

import os
import requests
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import math
from dotenv import load_dotenv


class MapboxImageryDownloader:
    """Download Mapbox satellite imagery for specified areas"""

    def __init__(self, access_token: str = None, output_dir: str = './data/imagery'):
        """
        Initialize Mapbox imagery downloader

        Args:
            access_token: Mapbox access token (or set MAPBOX_ACCESS_TOKEN env var)
            output_dir: Directory to save downloaded imagery
        """
        load_dotenv()
        self.access_token = access_token or os.getenv('MAPBOX_ACCESS_TOKEN')
        if not self.access_token:
            raise ValueError("Mapbox access token required. Set MAPBOX_ACCESS_TOKEN environment variable.")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.base_url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"

    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """
        Convert lat/lon to tile numbers

        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            zoom: Zoom level

        Returns:
            Tuple of (x_tile, y_tile)
        """
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x_tile = int((lon_deg + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x_tile, y_tile)

    def download_bbox_imagery(
        self,
        bbox: List[float],
        city_name: str,
        zoom: int = 16,
        width: int = 1280,
        height: int = 1280
    ) -> str:
        """
        Download satellite imagery for a bounding box

        Args:
            bbox: Bounding box [south, west, north, east]
            city_name: Name for the output file
            zoom: Zoom level (higher = more detail, default 16)
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Path to saved imagery file
        """
        south, west, north, east = bbox

        # Calculate center point
        center_lon = (west + east) / 2
        center_lat = (south + north) / 2

        # Build request URL
        # Format: /styles/v1/{username}/{style_id}/static/{lon},{lat},{zoom}/{width}x{height}
        url = (
            f"{self.base_url}/"
            f"{center_lon},{center_lat},{zoom}/"
            f"{width}x{height}"
            f"?access_token={self.access_token}"
        )

        print(f"Downloading imagery for {city_name} at zoom {zoom}...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Load and save image
            img = Image.open(BytesIO(response.content))
            output_path = os.path.join(self.output_dir, f"{city_name}_z{zoom}.png")
            img.save(output_path)

            print(f"✓ Saved imagery to {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Error downloading imagery for {city_name}: {str(e)}")
            raise

    def download_tiled_imagery(
        self,
        bbox: List[float],
        city_name: str,
        zoom: int = 17,
        tile_size: int = 512
    ) -> List[str]:
        """
        Download tiled imagery for better coverage of large areas

        Args:
            bbox: Bounding box [south, west, north, east]
            city_name: Name prefix for output files
            zoom: Zoom level
            tile_size: Size of each tile in pixels

        Returns:
            List of paths to saved tile files
        """
        south, west, north, east = bbox

        # Get tile ranges
        x_min, y_max = self.deg2num(south, west, zoom)
        x_max, y_min = self.deg2num(north, east, zoom)

        print(f"Downloading {(x_max-x_min+1) * (y_max-y_min+1)} tiles for {city_name}...")

        tile_paths = []

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # Build tile URL
                url = (
                    f"https://api.mapbox.com/v4/mapbox.satellite/{zoom}/{x}/{y}@2x.png"
                    f"?access_token={self.access_token}"
                )

                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    # Save tile
                    tile_filename = f"{city_name}_tile_{zoom}_{x}_{y}.png"
                    tile_path = os.path.join(self.output_dir, tile_filename)

                    with open(tile_path, 'wb') as f:
                        f.write(response.content)

                    tile_paths.append(tile_path)

                except Exception as e:
                    print(f"✗ Failed to download tile {x},{y}: {str(e)}")
                    continue

        print(f"✓ Downloaded {len(tile_paths)} tiles for {city_name}")
        return tile_paths

    def download_all_cities(self, cities_bbox: dict, zoom: int = 16):
        """
        Download imagery for multiple cities

        Args:
            cities_bbox: Dictionary mapping city names to bounding boxes
            zoom: Zoom level for imagery
        """
        for city_name, bbox in cities_bbox.items():
            try:
                self.download_bbox_imagery(bbox, city_name, zoom)
            except Exception as e:
                print(f"Failed to download imagery for {city_name}: {str(e)}")
                continue


if __name__ == "__main__":
    # Example usage
    CITIES = {
        'paris': [48.815573, 2.224199, 48.902145, 2.469920],
        'london': [51.286760, -0.510375, 51.691874, 0.334015],
        'new_york': [40.477399, -74.259090, 40.917577, -73.700272],
        'hong_kong': [22.153689, 113.835079, 22.561968, 114.406844],
        'moscow': [55.491878, 37.319336, 55.957565, 37.967987],
        'tokyo': [35.528874, 139.560547, 35.817813, 139.910278],
        'singapore': [1.205764, 103.604736, 1.470974, 104.028320]
    }

    downloader = MapboxImageryDownloader()
    downloader.download_all_cities(CITIES, zoom=16)
