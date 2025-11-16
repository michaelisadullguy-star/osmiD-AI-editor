"""
Critical tests for coordinate conversion
THESE TESTS MUST PASS - Incorrect coordinate conversion could map features to wrong locations!
"""

import pytest
import numpy as np
from common.coordinates import (
    LatLon, LonLat, BoundingBox,
    latlon_to_pixel, pixel_to_latlon
)


class TestLatLon:
    """Test LatLon coordinate type"""

    def test_create_latlon(self):
        """Test creating LatLon coordinate"""
        coord = LatLon(lat=48.8566, lon=2.3522)
        assert coord.lat == 48.8566
        assert coord.lon == 2.3522

    def test_latlon_to_geojson(self):
        """Test conversion to GeoJSON format [lon, lat]"""
        coord = LatLon(lat=48.8566, lon=2.3522)
        geojson = coord.to_geojson()
        assert geojson == [2.3522, 48.8566]  # [lon, lat] order!

    def test_latlon_validation_valid(self):
        """Test validation accepts valid coordinates"""
        coord = LatLon(lat=48.8566, lon=2.3522)
        assert coord.validate() == coord

    def test_latlon_validation_invalid_lat(self):
        """Test validation rejects invalid latitude"""
        with pytest.raises(ValueError, match="Invalid latitude"):
            LatLon(lat=999, lon=2.3522).validate()

        with pytest.raises(ValueError, match="Invalid latitude"):
            LatLon(lat=-91, lon=2.3522).validate()

    def test_latlon_validation_invalid_lon(self):
        """Test validation rejects invalid longitude"""
        with pytest.raises(ValueError, match="Invalid longitude"):
            LatLon(lat=48.8566, lon=999).validate()

        with pytest.raises(ValueError, match="Invalid longitude"):
            LatLon(lat=48.8566, lon=-181).validate()


class TestLonLat:
    """Test LonLat coordinate type"""

    def test_create_lonlat(self):
        """Test creating LonLat coordinate"""
        coord = LonLat(lon=2.3522, lat=48.8566)
        assert coord.lon == 2.3522
        assert coord.lat == 48.8566

    def test_lonlat_to_latlon(self):
        """Test conversion to LatLon format"""
        coord = LonLat(lon=2.3522, lat=48.8566)
        latlon = coord.to_latlon()
        assert isinstance(latlon, LatLon)
        assert latlon.lat == 48.8566
        assert latlon.lon == 2.3522


class TestBoundingBox:
    """Test BoundingBox class"""

    def test_create_bbox(self):
        """Test creating bounding box"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.5)
        assert bbox.south == 48.8
        assert bbox.west == 2.2
        assert bbox.north == 48.9
        assert bbox.east == 2.5

    def test_bbox_center(self):
        """Test bounding box center calculation"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)
        center = bbox.center()
        assert center.lat == pytest.approx(48.85)
        assert center.lon == pytest.approx(2.3)

    def test_bbox_contains(self):
        """Test point containment in bounding box"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)

        # Point inside
        assert bbox.contains(LatLon(lat=48.85, lon=2.3))

        # Point outside
        assert not bbox.contains(LatLon(lat=49.0, lon=2.3))
        assert not bbox.contains(LatLon(lat=48.85, lon=3.0))

    def test_bbox_validation(self):
        """Test bounding box validation"""
        # Valid bbox
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)
        assert bbox.validate() == bbox

        # Invalid: south >= north
        with pytest.raises(ValueError, match="South .* must be less than north"):
            BoundingBox(south=48.9, west=2.2, north=48.8, east=2.4).validate()

        # Invalid: west >= east
        with pytest.raises(ValueError, match="West .* must be less than east"):
            BoundingBox(south=48.8, west=2.4, north=48.9, east=2.2).validate()


class TestCoordinateConversion:
    """
    CRITICAL TESTS: Coordinate conversion between pixels and lat/lon
    These tests verify that features are mapped to the correct geographic locations
    """

    def test_pixel_to_latlon_center(self):
        """Test that center pixel maps to center lat/lon"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)
        img_width, img_height = 1000, 1000

        # Center pixel should map to center lat/lon
        pixel_coords = np.array([[500, 500]])
        result = pixel_to_latlon(pixel_coords, bbox, img_width, img_height)

        expected_lat = (48.8 + 48.9) / 2  # 48.85
        expected_lon = (2.2 + 2.4) / 2    # 2.3

        assert len(result) == 1
        assert result[0].lat == pytest.approx(expected_lat, abs=0.001)
        assert result[0].lon == pytest.approx(expected_lon, abs=0.001)

    def test_pixel_to_latlon_corners(self):
        """Test corner pixels map to corner coordinates"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)
        img_width, img_height = 1000, 1000

        # Top-left corner (0, 0) -> (north, west)
        result = pixel_to_latlon(np.array([[0, 0]]), bbox, img_width, img_height)
        assert result[0].lat == pytest.approx(bbox.north, abs=0.001)
        assert result[0].lon == pytest.approx(bbox.west, abs=0.001)

        # Bottom-right corner (width-1, height-1) -> (south, east)
        result = pixel_to_latlon(np.array([[img_width-1, img_height-1]]), bbox, img_width, img_height)
        assert result[0].lat == pytest.approx(bbox.south, abs=0.001)
        assert result[0].lon == pytest.approx(bbox.east, abs=0.001)

    def test_latlon_to_pixel_center(self):
        """Test that center lat/lon maps to center pixel"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)
        img_width, img_height = 1000, 1000

        center_coord = LatLon(lat=48.85, lon=2.3)
        x, y = latlon_to_pixel(center_coord, bbox, img_width, img_height)

        assert x == pytest.approx(500, abs=1)
        assert y == pytest.approx(500, abs=1)

    def test_latlon_to_pixel_corners(self):
        """Test corner coordinates map to corner pixels"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)
        img_width, img_height = 1000, 1000

        # North-west corner
        nw = LatLon(lat=bbox.north, lon=bbox.west)
        x, y = latlon_to_pixel(nw, bbox, img_width, img_height)
        assert x == pytest.approx(0, abs=1)
        assert y == pytest.approx(0, abs=1)

        # South-east corner
        se = LatLon(lat=bbox.south, lon=bbox.east)
        x, y = latlon_to_pixel(se, bbox, img_width, img_height)
        assert x == pytest.approx(img_width-1, abs=1)
        assert y == pytest.approx(img_height-1, abs=1)

    def test_roundtrip_conversion(self):
        """Test that pixel -> latlon -> pixel gives same result"""
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)
        img_width, img_height = 1000, 1000

        # Original pixel coordinates
        original_pixels = np.array([[250, 250], [500, 500], [750, 750]])

        # Convert to lat/lon
        latlons = pixel_to_latlon(original_pixels, bbox, img_width, img_height)

        # Convert back to pixels
        for i, latlon in enumerate(latlons):
            x, y = latlon_to_pixel(latlon, bbox, img_width, img_height)
            assert x == pytest.approx(original_pixels[i][0], abs=1)
            assert y == pytest.approx(original_pixels[i][1], abs=1)

    def test_coordinate_order_consistency(self):
        """
        CRITICAL TEST: Verify coordinate order is consistent throughout
        This prevents the lat/lon vs lon/lat confusion bug
        """
        bbox = BoundingBox(south=48.8, west=2.2, north=48.9, east=2.4)

        # Create LatLon (lat, lon) order
        latlon = LatLon(lat=48.85, lon=2.3)

        # Convert to GeoJSON should give [lon, lat]
        geojson = latlon.to_geojson()
        assert geojson[0] == latlon.lon  # First element is longitude
        assert geojson[1] == latlon.lat  # Second element is latitude

        # Create LonLat (lon, lat) order
        lonlat = LonLat(lon=2.3, lat=48.85)

        # Convert to LatLon
        converted = lonlat.to_latlon()
        assert converted.lat == lonlat.lat
        assert converted.lon == lonlat.lon
