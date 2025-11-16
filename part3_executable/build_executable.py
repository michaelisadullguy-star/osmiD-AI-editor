"""
Build executable using PyInstaller
Creates standalone .exe for Windows
"""

import PyInstaller.__main__
import os
import sys


def build_executable():
    """Build standalone executable"""

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(script_dir, 'icon.ico')  # Optional: add icon file

    # PyInstaller options
    options = [
        'gui_application.py',  # Main script
        '--name=osmiD-AI-editor',  # Executable name
        '--onefile',  # Single executable
        '--windowed',  # No console window (for GUI)
        '--clean',  # Clean cache
        f'--distpath={os.path.join(script_dir, "..", "dist")}',  # Output directory
        f'--workpath={os.path.join(script_dir, "..", "build")}',  # Build directory
        '--add-data=../part2_training/model.py:part2_training',  # Include model
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=cv2',
        '--hidden-import=osmapi',
        '--hidden-import=PyQt5',
        '--collect-all=torch',
        '--collect-all=torchvision',
    ]

    # Add icon if exists
    if os.path.exists(icon_path):
        options.append(f'--icon={icon_path}')

    print("Building executable...")
    print(f"Options: {' '.join(options)}")

    PyInstaller.__main__.run(options)

    print("\n✓ Build completed!")
    print(f"Executable location: {os.path.join(script_dir, '..', 'dist', 'osmiD-AI-editor.exe')}")


if __name__ == "__main__":
    build_executable()
