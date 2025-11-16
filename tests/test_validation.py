"""
Tests for input validation - security critical
"""

import pytest
from pathlib import Path
from common.validation import (
    validate_coordinates,
    validate_polygon,
    validate_file_path,
    validate_email,
    validate_model_path,
    sanitize_filename
)
from common.coordinates import LatLon, BoundingBox


class TestCoordinateValidation:
    """Test coordinate validation"""

    def test_valid_coordinates(self):
        """Test valid coordinates pass validation"""
        assert validate_coordinates(48.8566, 2.3522) is True
        assert validate_coordinates(0, 0) is True
        assert validate_coordinates(90, 180) is True
        assert validate_coordinates(-90, -180) is True

    def test_invalid_latitude(self):
        """Test invalid latitude raises error"""
        with pytest.raises(ValueError, match="Invalid latitude"):
            validate_coordinates(91, 0)

        with pytest.raises(ValueError, match="Invalid latitude"):
            validate_coordinates(-91, 0)

        with pytest.raises(ValueError, match="Invalid latitude"):
            validate_coordinates(999, 0)

    def test_invalid_longitude(self):
        """Test invalid longitude raises error"""
        with pytest.raises(ValueError, match="Invalid longitude"):
            validate_coordinates(0, 181)

        with pytest.raises(ValueError, match="Invalid longitude"):
            validate_coordinates(0, -181)

        with pytest.raises(ValueError, match="Invalid longitude"):
            validate_coordinates(0, 999)


class TestPolygonValidation:
    """Test polygon validation"""

    def test_valid_polygon(self):
        """Test valid polygon passes validation"""
        coords = [
            LatLon(lat=48.85, lon=2.30),
            LatLon(lat=48.86, lon=2.30),
            LatLon(lat=48.86, lon=2.31),
            LatLon(lat=48.85, lon=2.31),
            LatLon(lat=48.85, lon=2.30),  # Closed
        ]
        poly = validate_polygon(coords)
        assert poly.is_valid
        assert poly.area > 0

    def test_polygon_too_few_points(self):
        """Test polygon with too few points raises error"""
        coords = [
            LatLon(lat=48.85, lon=2.30),
            LatLon(lat=48.86, lon=2.30),
        ]
        with pytest.raises(ValueError, match="at least 3 points"):
            validate_polygon(coords)

    def test_polygon_zero_area(self):
        """Test polygon with zero area raises error"""
        coords = [
            LatLon(lat=48.85, lon=2.30),
            LatLon(lat=48.85, lon=2.30),
            LatLon(lat=48.85, lon=2.30),
        ]
        with pytest.raises(ValueError, match="zero area"):
            validate_polygon(coords)

    def test_self_intersecting_polygon(self):
        """Test self-intersecting polygon raises error"""
        # Bowtie shape (self-intersecting)
        coords = [
            LatLon(lat=48.85, lon=2.30),
            LatLon(lat=48.86, lon=2.31),
            LatLon(lat=48.86, lon=2.30),
            LatLon(lat=48.85, lon=2.31),
            LatLon(lat=48.85, lon=2.30),
        ]
        with pytest.raises(ValueError, match="self-intersecting"):
            validate_polygon(coords)


class TestFilePathValidation:
    """Test file path validation - prevents path traversal attacks"""

    def test_valid_file_path(self, tmp_path):
        """Test valid file path passes validation"""
        test_file = tmp_path / "test.pth"
        test_file.touch()

        result = validate_file_path(
            str(test_file),
            allowed_dirs=[str(tmp_path)],
            allowed_extensions=['.pth'],
            must_exist=True
        )
        assert isinstance(result, Path)
        assert result.exists()

    def test_path_traversal_blocked(self, tmp_path):
        """Test path traversal attack is blocked"""
        malicious_path = tmp_path / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError, match="must be within allowed directories"):
            validate_file_path(
                str(malicious_path),
                allowed_dirs=[str(tmp_path)],
                allowed_extensions=['.pth']
            )

    def test_invalid_extension_blocked(self, tmp_path):
        """Test file with invalid extension is blocked"""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with pytest.raises(ValueError, match="must have one of these extensions"):
            validate_file_path(
                str(test_file),
                allowed_dirs=[str(tmp_path)],
                allowed_extensions=['.pth', '.pt']
            )

    def test_nonexistent_file_blocked(self, tmp_path):
        """Test nonexistent file is blocked when must_exist=True"""
        nonexistent = tmp_path / "nonexistent.pth"

        with pytest.raises(ValueError, match="does not exist"):
            validate_file_path(
                str(nonexistent),
                allowed_dirs=[str(tmp_path)],
                must_exist=True
            )


class TestEmailValidation:
    """Test email validation"""

    def test_valid_emails(self):
        """Test valid emails pass validation"""
        valid_emails = [
            "user@example.com",
            "test.user@example.com",
            "user+tag@example.co.uk",
            "user_123@test-domain.com",
        ]
        for email in valid_emails:
            assert validate_email(email) is True

    def test_invalid_emails(self):
        """Test invalid emails raise error"""
        invalid_emails = [
            "",  # Empty
            "not-an-email",  # No @
            "@example.com",  # No local part
            "user@",  # No domain
            "user@domain",  # No TLD
            "user name@example.com",  # Space in local part
            "a" * 255 + "@example.com",  # Too long
        ]
        for email in invalid_emails:
            with pytest.raises(ValueError, match="Invalid email"):
                validate_email(email)


class TestFilenameSanitization:
    """Test filename sanitization"""

    def test_sanitize_normal_filename(self):
        """Test normal filename is unchanged"""
        assert sanitize_filename("model.pth") == "model.pth"
        assert sanitize_filename("my_model_v1.pth") == "my_model_v1.pth"

    def test_sanitize_path_separators(self):
        """Test path separators are removed"""
        assert sanitize_filename("../../../etc/passwd") == ".._.._.._etc_passwd"
        assert sanitize_filename("C:\\Windows\\System32") == "C__Windows_System32"

    def test_sanitize_dangerous_characters(self):
        """Test dangerous characters are removed"""
        assert sanitize_filename("file<>name.pth") == "file__name.pth"
        assert sanitize_filename("file|name*.pth") == "file_name_.pth"

    def test_sanitize_length_limit(self):
        """Test filename length is limited"""
        long_name = "a" * 300 + ".pth"
        result = sanitize_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".pth")
