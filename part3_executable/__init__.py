"""
Part 3: Executable Application Module
GUI application for automated OSM feature mapping
"""

from .osm_client import OSMClient
from .feature_detector import FeatureDetector
from .gui_application import MainWindow

__all__ = ['OSMClient', 'FeatureDetector', 'MainWindow']
