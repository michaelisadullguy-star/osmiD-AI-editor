"""
Tests for common modules
"""

import pytest
from common import CITIES, FEATURE_CLASSES
from common.config import Config
from common.retry import retry_with_backoff, RateLimiter
import time


class TestConstants:
    """Test common constants"""

    def test_cities_defined(self):
        """Test all expected cities are defined"""
        expected_cities = ['paris', 'london', 'new_york', 'hong_kong', 'moscow', 'tokyo', 'singapore']
        for city in expected_cities:
            assert city in CITIES

    def test_city_bbox_format(self):
        """Test city bounding boxes have correct format"""
        for city, bbox in CITIES.items():
            assert len(bbox) == 4, f"{city} bbox should have 4 values [south, west, north, east]"
            south, west, north, east = bbox
            assert south < north, f"{city}: south must be less than north"
            assert west < east, f"{city}: west must be less than east"
            assert -90 <= south <= 90, f"{city}: invalid south latitude"
            assert -90 <= north <= 90, f"{city}: invalid north latitude"
            assert -180 <= west <= 180, f"{city}: invalid west longitude"
            assert -180 <= east <= 180, f"{city}: invalid east longitude"

    def test_feature_classes(self):
        """Test feature classes are defined"""
        assert 0 in FEATURE_CLASSES  # background
        assert 1 in FEATURE_CLASSES  # building
        assert 2 in FEATURE_CLASSES  # lawn
        assert 3 in FEATURE_CLASSES  # natural_wood
        assert 4 in FEATURE_CLASSES  # artificial_forest
        assert 5 in FEATURE_CLASSES  # water_body
        assert len(FEATURE_CLASSES) == 6


class TestConfig:
    """Test configuration management"""

    def test_default_config(self):
        """Test default configuration"""
        config = Config()
        assert config.batch_size > 0
        assert config.epochs > 0
        assert config.learning_rate > 0
        assert config.n_classes == 6
        assert config.dry_run is True  # Default to safe mode
        assert config.require_review is True

    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        config.validate()  # Should not raise

        # Invalid batch size
        config_invalid = Config(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config_invalid.validate()

        # Invalid learning rate
        config_invalid = Config(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config_invalid.validate()

        # Overlapping cities
        config_invalid = Config(
            train_cities=['paris', 'london'],
            val_cities=['paris', 'tokyo']  # Paris in both
        )
        with pytest.raises(ValueError, match="overlap"):
            config_invalid.validate()


class TestRetryLogic:
    """Test retry and rate limiting"""

    def test_retry_success_no_retry(self):
        """Test successful call doesn't retry"""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def successful_call():
            call_count[0] += 1
            return "success"

        result = successful_call()
        assert result == "success"
        assert call_count[0] == 1  # Called only once

    def test_retry_eventually_succeeds(self):
        """Test retry logic with eventual success"""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def fails_twice_then_succeeds():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = fails_twice_then_succeeds()
        assert result == "success"
        assert call_count[0] == 3  # Called 3 times

    def test_retry_max_retries_exceeded(self):
        """Test retry gives up after max retries"""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count[0] == 3  # Tried 3 times then gave up

    def test_rate_limiter(self):
        """Test rate limiter enforces delay"""
        limiter = RateLimiter(calls_per_second=10)  # 10 calls/sec = 0.1s between calls

        start_time = time.time()

        # Make 3 calls
        for _ in range(3):
            with limiter:
                pass

        elapsed = time.time() - start_time

        # Should take at least 0.2 seconds (2 delays of 0.1s each)
        assert elapsed >= 0.2, f"Rate limiter didn't enforce delay, elapsed: {elapsed}s"
