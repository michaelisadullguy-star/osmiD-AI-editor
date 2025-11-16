"""
Utility script to launch the GUI application
"""

import sys
import os

# Add part3_executable to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'part3_executable'))

from part3_executable.gui_application import main

if __name__ == "__main__":
    print("=" * 80)
    print("osmiD-AI-editor - GUI Application")
    print("=" * 80)
    print()
    print("Launching application...")
    print()
    main()
