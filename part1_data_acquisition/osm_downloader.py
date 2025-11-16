"""
Part 1: OSM Data Downloader using Overpass API
Downloads OpenStreetMap data for specified cities including buildings,
lawns, natural woods, artificial forests, and water bodies.

Security improvements:
- Proper error handling with specific exceptions
- Logging instead of print statements
- Rate limiting with exponential backoff
"""

import overpy
import json
import os
from typing import Dict, List
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CITIES, setup_logging
from common.retry import retry_with_backoff, RateLimiter
from common.logging_config import LoggerMixin

logger = setup_logging('osmid.data_acquisition')


class OSMDownloader(LoggerMixin):
    """Download OSM data for specified cities using Overpass API"""

    def __init__(self, output_dir: str = './data/osm'):
        """
        Initialize OSM downloader

        Args:
            output_dir: Directory to save downloaded OSM data
        """
        self.api = overpy.Overpass()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rate_limiter = RateLimiter(calls_per_second=0.5)  # 1 call per 2 seconds

        self.logger.info(f"Initialized OSM downloader, output dir: {output_dir}")

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
            else:
                self.logger.warning(f"Unknown feature type: {feature}")

        query = f"""
        [out:json][timeout:300];
        (
          {' '.join(query_parts)}
        );
        out geom;
        """

        return query

    @retry_with_backoff(max_retries=4, base_delay=2.0, exceptions=(overpy.exception.OverpassTooManyRequests, overpy.exception.OverpassGatewayTimeout))
    def _query_with_retry(self, query: str):
        """Execute Overpass query with retry logic"""
        with self.rate_limiter:
            return self.api.query(query)

    def download_city_data(self, city: str, features: List[str] = None) -> Dict:
        """
        Download OSM data for a specific city

        Args:
            city: City name (must be in CITIES dict)
            features: List of features to download (default: all)

        Returns:
            Dictionary containing OSM data

        Raises:
            ValueError: If city is not in CITIES
            overpy.exception.OverpassException: If API query fails
        """
        if city not in CITIES:
            raise ValueError(f"City '{city}' not in available cities: {list(CITIES.keys())}")

        if features is None:
            features = ['buildings', 'lawns', 'natural_woods', 'artificial_forests', 'water_bodies']

        bbox = CITIES[city]
        query = self.build_query(bbox, features)

        self.logger.info(f"Downloading OSM data for {city}...")
        self.logger.debug(f"Query: {query[:100]}...")

        try:
            result = self._query_with_retry(query)

            # Convert result to GeoJSON format
            geojson = self._convert_to_geojson(result, city, features)

            # Save to file
            output_file = os.path.join(self.output_dir, f"{city}.geojson")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2)

            self.logger.info(f"Saved {len(geojson['features'])} features to {output_file}")

            return geojson

        except overpy.exception.OverpassTooManyRequests as e:
            self.logger.error(f"Rate limit exceeded for {city}: {e}")
            raise
        except overpy.exception.OverpassGatewayTimeout as e:
            self.logger.error(f"Gateway timeout for {city}: {e}")
            raise
        except overpy.exception.OverpassException as e:
            self.logger.error(f"Overpass API error for {city}: {e}")
            raise
        except (IOError, OSError) as e:
            self.logger.error(f"File I/O error for {city}: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error downloading data for {city}")
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
                "features": features,
                "generator": "osmiD-AI-editor v1.0"
            },
            "features": []
        }

        # Process ways
        for way in result.ways:
            try:
                coords = [[float(node.lon), float(node.lat)] for node in way.nodes]

                if not coords:
                    self.logger.warning(f"Skipping way {way.id} with no coordinates")
                    continue

                # Determine feature type
                feature_type = self._determine_feature_type(way.tags)

                # Determine geometry type
                is_closed = len(coords) > 2 and coords[0] == coords[-1]
                geometry_type = "Polygon" if is_closed else "LineString"
                geometry_coords = [coords] if is_closed else coords

                feature = {
                    "type": "Feature",
                    "id": way.id,
                    "geometry": {
                        "type": geometry_type,
                        "coordinates": geometry_coords
                    },
                    "properties": {
                        "osm_id": way.id,
                        "feature_type": feature_type,
                        "tags": way.tags
                    }
                }

                geojson["features"].append(feature)

            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Error processing way {way.id}: {e}")
                continue

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
        successful = 0
        failed = 0

        for city in CITIES.keys():
            try:
                self.download_city_data(city, features)
                successful += 1
            except Exception as e:
                self.logger.error(f"Failed to download {city}: {e}")
                failed += 1
                continue

        self.logger.info(f"Download complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    downloader = OSMDownloader()
    downloader.download_all_cities()
