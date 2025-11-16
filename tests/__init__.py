"""
Test suite for osmiD-AI-editor

Run tests with: pytest tests/ -v
Run with coverage: pytest tests/ --cov=. --cov-report=html
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
