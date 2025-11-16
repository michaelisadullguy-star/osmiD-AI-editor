"""
Part 1: OSM Data Downloader using Overpass API
Downloads OpenStreetMap data for specified cities including buildings,
lawns, natural woods, artificial forests, and water bodies.
"""

import overpy
import json
import os
from typing import Dict, List, Tuple
import time


class OSMDownloader:
    """Download OSM data for specified cities using Overpass API"""

    # City bounding boxes [south, west, north, east]
    CITIES = {
        'paris': [48.815573, 2.224199, 48.902145, 2.469920],
        'london': [51.286760, -0.510375, 51.691874, 0.334015],
        'new_york': [40.477399, -74.259090, 40.917577, -73.700272],
        'hong_kong': [22.153689, 113.835079, 22.561968, 114.406844],
        'moscow': [55.491878, 37.319336, 55.957565, 37.967987],
        'tokyo': [35.528874, 139.560547, 35.817813, 139.910278],
        'singapore': [1.205764, 103.604736, 1.470974, 104.028320]
    }

    def __init__(self, output_dir: str = './data/osm'):
        """
        Initialize OSM downloader

        Args:
            output_dir: Directory to save downloaded OSM data
        """
        self.api = overpy.Overpass()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_query(self, bbox: List[float], features: List[str]) -> str:
        """
        Build Overpass QL query for specified features

        Args:
            bbox: Bounding box [south, west, north, east]
            features: List of feature types to download

        Returns:
            Overpass QL query string
        """
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        # Feature queries
        feature_queries = {
            'buildings': f'way["building"]({bbox_str});',
            'lawns': f'way["landuse"="grass"]({bbox_str});',
            'natural_woods': f'way["natural"="wood"]({bbox_str});',
            'artificial_forests': f'way["landuse"="forest"]({bbox_str});',
            'water_bodies': f'(way["natural"="water"]({bbox_str}); way["waterway"]({bbox_str});)'
        }

        query_parts = []
        for feature in features:
            if feature in feature_queries:
                query_parts.append(feature_queries[feature])

        query = f"""
        [out:json][timeout:300];
        (
          {' '.join(query_parts)}
        );
        out geom;
        """

        return query

    def download_city_data(self, city: str, features: List[str] = None) -> Dict:
        """
        Download OSM data for a specific city

        Args:
            city: City name (must be in CITIES dict)
            features: List of features to download (default: all)

        Returns:
            Dictionary containing OSM data
        """
        if city not in self.CITIES:
            raise ValueError(f"City {city} not in available cities: {list(self.CITIES.keys())}")

        if features is None:
            features = ['buildings', 'lawns', 'natural_woods', 'artificial_forests', 'water_bodies']

        bbox = self.CITIES[city]
        query = self.build_query(bbox, features)

        print(f"Downloading OSM data for {city}...")
        print(f"Query: {query[:100]}...")

        try:
            result = self.api.query(query)

            # Convert result to GeoJSON format
            geojson = self._convert_to_geojson(result, city, features)

            # Save to file
            output_file = os.path.join(self.output_dir, f"{city}.geojson")
            with open(output_file, 'w') as f:
                json.dump(geojson, f, indent=2)

            print(f"✓ Saved {len(geojson['features'])} features to {output_file}")

            return geojson

        except Exception as e:
            print(f"✗ Error downloading data for {city}: {str(e)}")
            raise

    def _convert_to_geojson(self, result, city: str, features: List[str]) -> Dict:
        """
        Convert Overpass API result to GeoJSON format

        Args:
            result: Overpass API result
            city: City name
            features: List of features

        Returns:
            GeoJSON dictionary
        """
        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "city": city,
                "features": features
            },
            "features": []
        }

        # Process ways
        for way in result.ways:
            coords = [[float(node.lon), float(node.lat)] for node in way.nodes]

            # Determine feature type
            feature_type = self._determine_feature_type(way.tags)

            feature = {
                "type": "Feature",
                "id": way.id,
                "geometry": {
                    "type": "Polygon" if coords[0] == coords[-1] else "LineString",
                    "coordinates": [coords] if coords[0] == coords[-1] else coords
                },
                "properties": {
                    "osm_id": way.id,
                    "feature_type": feature_type,
                    "tags": way.tags
                }
            }

            geojson["features"].append(feature)

        return geojson

    def _determine_feature_type(self, tags: Dict) -> str:
        """Determine feature type from OSM tags"""
        if 'building' in tags:
            return 'building'
        elif tags.get('landuse') == 'grass':
            return 'lawn'
        elif tags.get('natural') == 'wood':
            return 'natural_wood'
        elif tags.get('landuse') == 'forest':
            return 'artificial_forest'
        elif 'water' in tags.get('natural', '') or 'waterway' in tags:
            return 'water_body'
        else:
            return 'unknown'

    def download_all_cities(self, features: List[str] = None):
        """
        Download OSM data for all predefined cities

        Args:
            features: List of features to download (default: all)
        """
        for city in self.CITIES.keys():
            try:
                self.download_city_data(city, features)
                # Rate limiting
                time.sleep(2)
            except Exception as e:
                print(f"Failed to download {city}: {str(e)}")
                continue


if __name__ == "__main__":
    downloader = OSMDownloader()
    downloader.download_all_cities()
