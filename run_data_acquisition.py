"""
Utility script to run complete data acquisition pipeline
Downloads OSM data, satellite imagery, and creates correlated training data
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from part1_data_acquisition.osm_downloader import OSMDownloader
from part1_data_acquisition.mapbox_imagery import MapboxImageryDownloader
from part1_data_acquisition.feature_correlator import FeatureCorrelator


# City configurations
CITIES = {
    'paris': [48.815573, 2.224199, 48.902145, 2.469920],
    'london': [51.286760, -0.510375, 51.691874, 0.334015],
    'new_york': [40.477399, -74.259090, 40.917577, -73.700272],
    'hong_kong': [22.153689, 113.835079, 22.561968, 114.406844],
    'moscow': [55.491878, 37.319336, 55.957565, 37.967987],
    'tokyo': [35.528874, 139.560547, 35.817813, 139.910278],
    'singapore': [1.205764, 103.604736, 1.470974, 104.028320]
}


def main():
    """Run complete data acquisition pipeline"""
    print("=" * 80)
    print("osmiD-AI-editor - Data Acquisition Pipeline")
    print("=" * 80)
    print()

    # Step 1: Download OSM data
    print("STEP 1: Downloading OSM data for all cities...")
    print("-" * 80)
    try:
        osm_downloader = OSMDownloader(output_dir='./data/osm')
        osm_downloader.download_all_cities()
        print("\n✓ OSM data download completed\n")
    except Exception as e:
        print(f"\n✗ Error downloading OSM data: {str(e)}\n")
        return

    # Step 2: Download satellite imagery
    print("STEP 2: Downloading satellite imagery...")
    print("-" * 80)

    mapbox_token = os.getenv('MAPBOX_ACCESS_TOKEN')
    if not mapbox_token:
        print("✗ Error: MAPBOX_ACCESS_TOKEN not set in .env file")
        print("Please add your Mapbox access token to .env file")
        return

    try:
        imagery_downloader = MapboxImageryDownloader(
            access_token=mapbox_token,
            output_dir='./data/imagery'
        )
        imagery_downloader.download_all_cities(CITIES, zoom=16)
        print("\n✓ Satellite imagery download completed\n")
    except Exception as e:
        print(f"\n✗ Error downloading imagery: {str(e)}\n")
        return

    # Step 3: Correlate features
    print("STEP 3: Correlating features with imagery...")
    print("-" * 80)
    try:
        correlator = FeatureCorrelator(
            osm_data_dir='./data/osm',
            imagery_dir='./data/imagery'
        )
        correlator.create_training_dataset(
            cities=list(CITIES.keys()),
            cities_bbox=CITIES,
            zoom=16
        )
        print("\n✓ Feature correlation completed\n")
    except Exception as e:
        print(f"\n✗ Error correlating features: {str(e)}\n")
        return

    # Summary
    print("=" * 80)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 80)
    print()
    print("Training data is ready in the following directories:")
    print("  - OSM data:        ./data/osm/")
    print("  - Imagery:         ./data/imagery/")
    print("  - Correlated data: ./data/correlated/")
    print()
    print("Next steps:")
    print("  1. Review the downloaded data")
    print("  2. Run training: python -m part2_training.train")
    print()


if __name__ == "__main__":
    main()
