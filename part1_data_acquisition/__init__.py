"""
Part 1: Data Acquisition Module
Downloads and correlates OSM data with satellite imagery for training
"""

from .osm_downloader import OSMDownloader
from .mapbox_imagery import MapboxImageryDownloader
from .feature_correlator import FeatureCorrelator

__all__ = ['OSMDownloader', 'MapboxImageryDownloader', 'FeatureCorrelator']
