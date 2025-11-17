"""
Part 3: GUI Application
Main executable with user interface for OSM feature mapping
"""

import sys
import os
import re
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QMessageBox, QGroupBox, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QDesktopServices

from osm_client import OSMClient
from feature_detector import FeatureDetector


class MappingWorker(QThread):
    """Worker thread for feature detection and mapping"""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, object)  # success, message, changeset_info

    def __init__(
        self,
        email: str,
        password: str,
        polygon_coords: str,
        model_path: str,
        mapbox_token: str
    ):
        super().__init__()
        self.email = email
        self.password = password
        self.polygon_coords = polygon_coords
        self.model_path = model_path
        self.mapbox_token = mapbox_token

    def parse_polygon(self, polygon_str: str):
        """
        Parse polygon string to list of coordinates

        Expected format: {{lat1,lon1},{lat2,lon2},...,{latN,lonN}}

        Returns:
            List of (lat, lon) tuples and bounding box
        """
        # Remove whitespace
        polygon_str = polygon_str.strip()

        # Extract coordinate pairs
        pattern = r'\{([0-9.-]+),([0-9.-]+)\}'
        matches = re.findall(pattern, polygon_str)

        if not matches:
            raise ValueError("Invalid polygon format. Expected: {{lat,lon},{lat,lon},...}")

        coords = [(float(lat), float(lon)) for lat, lon in matches]

        if len(coords) < 3:
            raise ValueError("Polygon must have at least 3 points")

        # Calculate bounding box
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]

        bbox = [min(lats), min(lons), max(lats), max(lons)]  # south, west, north, east

        return coords, bbox

    def download_imagery(self, bbox, zoom=17, width=1280, height=1280):
        """Download satellite imagery for the polygon"""
        south, west, north, east = bbox

        # Calculate center
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2

        # Build Mapbox URL
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
            f"{center_lon},{center_lat},{zoom}/{width}x{height}"
            f"?access_token={self.mapbox_token}"
        )

        self.progress.emit("Downloading satellite imagery...")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Load image
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)

        self.progress.emit(f"✓ Downloaded imagery ({img_array.shape[0]}x{img_array.shape[1]})")

        return img_array

    def run(self):
        """Main worker execution"""
        try:
            # Parse polygon
            self.progress.emit("Parsing polygon coordinates...")
            coords, bbox = self.parse_polygon(self.polygon_coords)
            self.progress.emit(f"✓ Parsed {len(coords)} coordinates")
            self.progress.emit(f"  Bounding box: {bbox}")

            # Download imagery
            imagery = self.download_imagery(bbox)

            # Initialize detector
            self.progress.emit("Loading AI model...")
            detector = FeatureDetector(self.model_path, device='cpu')
            self.progress.emit("✓ Model loaded")

            # Detect features
            self.progress.emit("\nDetecting features...")
            features = detector.detect_features_in_polygon(imagery, bbox)
            self.progress.emit(f"✓ Detected {len(features)} features")

            if len(features) == 0:
                self.finished.emit(True, "No features detected in the specified area.")
                return

            # Authenticate with OSM
            self.progress.emit("\nAuthenticating with OpenStreetMap...")
            osm_client = OSMClient()

            if not osm_client.authenticate(self.email, self.password):
                self.finished.emit(False, "Authentication failed. Please check your credentials.")
                return

            self.progress.emit("✓ Authenticated successfully")

            # Upload features
            self.progress.emit("\nUploading features to OpenStreetMap...")
            changeset_info = osm_client.upload_features(
                features,
                changeset_comment="AI-assisted feature mapping via osmiD-AI-editor"
            )

            self.progress.emit(f"\n✓ Successfully uploaded {changeset_info['total_features']} features!")
            self.progress.emit(f"  Changeset ID: {changeset_info['changeset_id']}")
            self.progress.emit(f"  Changeset URL: {changeset_info['changeset_url']}")

            self.finished.emit(True, f"Successfully mapped {changeset_info['total_features']} features!", changeset_info)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.progress.emit(f"\n✗ {error_msg}")
            self.finished.emit(False, error_msg, None)


class ChangesetDialog(QDialog):
    """Dialog to display changeset information"""

    def __init__(self, changeset_info: dict, parent=None):
        super().__init__(parent)
        self.changeset_info = changeset_info
        self.init_ui()

    def init_ui(self):
        """Initialize dialog UI"""
        self.setWindowTitle("OSM Changeset Created")
        self.setModal(True)
        self.setMinimumWidth(500)

        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Title
        title = QLabel("✓ Successfully Created OSM Changeset")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4CAF50;")
        layout.addWidget(title)

        # Changeset info group
        info_group = QGroupBox("Changeset Details")
        info_layout = QVBoxLayout()

        # Changeset ID
        changeset_id_layout = QHBoxLayout()
        changeset_id_label = QLabel("Changeset ID:")
        changeset_id_label.setStyleSheet("font-weight: bold;")
        changeset_id_value = QLabel(str(self.changeset_info['changeset_id']))
        changeset_id_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        changeset_id_layout.addWidget(changeset_id_label)
        changeset_id_layout.addWidget(changeset_id_value)
        changeset_id_layout.addStretch()
        info_layout.addLayout(changeset_id_layout)

        # Total features
        total_layout = QHBoxLayout()
        total_label = QLabel("Total Features:")
        total_label.setStyleSheet("font-weight: bold;")
        total_value = QLabel(str(self.changeset_info['total_features']))
        total_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        total_layout.addWidget(total_label)
        total_layout.addWidget(total_value)
        total_layout.addStretch()
        info_layout.addLayout(total_layout)

        # Feature breakdown
        if self.changeset_info.get('feature_counts'):
            breakdown_label = QLabel("Feature Breakdown:")
            breakdown_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            info_layout.addWidget(breakdown_label)

            for feature_type, count in self.changeset_info['feature_counts'].items():
                feature_layout = QHBoxLayout()
                feature_name = QLabel(f"  • {feature_type.replace('_', ' ').title()}:")
                feature_count = QLabel(str(count))
                feature_layout.addWidget(feature_name)
                feature_layout.addWidget(feature_count)
                feature_layout.addStretch()
                info_layout.addLayout(feature_layout)

        # Comment
        comment_label = QLabel("Comment:")
        comment_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        info_layout.addWidget(comment_label)

        comment_value = QLabel(self.changeset_info['comment'])
        comment_value.setWordWrap(True)
        comment_value.setStyleSheet("margin-left: 10px; font-style: italic;")
        info_layout.addWidget(comment_value)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # URL section
        url_group = QGroupBox("View on OpenStreetMap")
        url_layout = QVBoxLayout()

        url_label = QLabel("Click the link below to view your changeset on OpenStreetMap:")
        url_layout.addWidget(url_label)

        # Clickable URL
        url_text = QLabel(f'<a href="{self.changeset_info["changeset_url"]}">{self.changeset_info["changeset_url"]}</a>')
        url_text.setOpenExternalLinks(True)
        url_text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        url_text.setStyleSheet("font-size: 11pt; color: #2196F3;")
        url_layout.addWidget(url_text)

        url_group.setLayout(url_layout)
        layout.addWidget(url_group)

        # Buttons
        button_layout = QHBoxLayout()

        # Open in browser button
        open_button = QPushButton("Open in Browser")
        open_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        open_button.clicked.connect(self.open_in_browser)
        button_layout.addWidget(open_button)

        # Close button
        close_button = QPushButton("Close")
        close_button.setStyleSheet("padding: 8px;")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def open_in_browser(self):
        """Open changeset URL in default browser"""
        QDesktopServices.openUrl(QUrl(self.changeset_info['changeset_url']))


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("osmiD-AI-editor - Automated OSM Feature Mapper")
        self.setGeometry(100, 100, 800, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(15)
        central_widget.setLayout(layout)

        # Title
        title_label = QLabel("osmiD-AI-editor")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        subtitle_label = QLabel("AI-Powered Automated OSM Feature Mapping")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)

        # Input group
        input_group = QGroupBox("Login & Configuration")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        # Email input
        email_layout = QHBoxLayout()
        email_label = QLabel("Email:")
        email_label.setMinimumWidth(100)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("your.email@example.com")
        email_layout.addWidget(email_label)
        email_layout.addWidget(self.email_input)
        input_layout.addLayout(email_layout)

        # Password input
        password_layout = QHBoxLayout()
        password_label = QLabel("Password:")
        password_label.setMinimumWidth(100)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Your OSM password")
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        input_layout.addLayout(password_layout)

        # Polygon input
        polygon_layout = QVBoxLayout()
        polygon_label = QLabel("Polygon Coordinates:")
        polygon_label.setMinimumWidth(100)
        self.polygon_input = QTextEdit()
        self.polygon_input.setPlaceholderText(
            "Enter polygon coordinates in format:\n"
            "{{lat1,lon1},{lat2,lon2},{lat3,lon3},...}\n\n"
            "Example:\n"
            "{{48.8566,2.3522},{48.8577,2.3540},{48.8560,2.3545},{48.8555,2.3530}}"
        )
        self.polygon_input.setMaximumHeight(120)
        polygon_layout.addWidget(polygon_label)
        polygon_layout.addWidget(self.polygon_input)
        input_layout.addLayout(polygon_layout)

        layout.addWidget(input_group)

        # Model path input
        model_layout = QHBoxLayout()
        model_label = QLabel("Model Path:")
        model_label.setMinimumWidth(100)
        self.model_input = QLineEdit()
        self.model_input.setText("./models/checkpoints/checkpoint_best.pth")
        self.model_input.setPlaceholderText("Path to trained model checkpoint")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_input)
        layout.addLayout(model_layout)

        # Mapbox token input
        mapbox_layout = QHBoxLayout()
        mapbox_label = QLabel("Mapbox Token:")
        mapbox_label.setMinimumWidth(100)
        self.mapbox_input = QLineEdit()
        self.mapbox_input.setPlaceholderText("Your Mapbox access token")
        mapbox_layout.addWidget(mapbox_label)
        mapbox_layout.addWidget(self.mapbox_input)
        layout.addLayout(mapbox_layout)

        # Start button
        self.start_button = QPushButton("Start Feature Mapping")
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self.start_mapping)
        layout.addWidget(self.start_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log output
        log_label = QLabel("Output Log:")
        layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        layout.addWidget(self.log_output)

        # Status bar
        self.statusBar().showMessage("Ready")

    def log_message(self, message: str):
        """Add message to log output"""
        self.log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_mapping(self):
        """Start the mapping process"""
        # Validate inputs
        email = self.email_input.text().strip()
        password = self.password_input.text()
        polygon = self.polygon_input.toPlainText().strip()
        model_path = self.model_input.text().strip()
        mapbox_token = self.mapbox_input.text().strip()

        if not email:
            QMessageBox.warning(self, "Input Error", "Please enter your email")
            return

        if not password:
            QMessageBox.warning(self, "Input Error", "Please enter your password")
            return

        if not polygon:
            QMessageBox.warning(self, "Input Error", "Please enter polygon coordinates")
            return

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Input Error", "Model checkpoint not found")
            return

        if not mapbox_token:
            QMessageBox.warning(self, "Input Error", "Please enter Mapbox access token")
            return

        # Confirm action
        reply = QMessageBox.question(
            self,
            "Confirm Upload",
            "This will upload detected features to OpenStreetMap.\n"
            "Are you sure you want to proceed?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Clear log
        self.log_output.clear()

        # Disable button and show progress
        self.start_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start worker thread
        self.worker = MappingWorker(email, password, polygon, model_path, mapbox_token)
        self.worker.progress.connect(self.log_message)
        self.worker.finished.connect(self.mapping_finished)
        self.worker.start()

        self.statusBar().showMessage("Processing...")

    def mapping_finished(self, success: bool, message: str, changeset_info: dict):
        """Handle mapping completion"""
        # Re-enable button
        self.start_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            self.statusBar().showMessage("Completed successfully!")

            # Show changeset dialog if we have changeset info
            if changeset_info:
                dialog = ChangesetDialog(changeset_info, self)
                dialog.exec_()
            else:
                # Fallback to simple message if no changeset info
                QMessageBox.information(self, "Success", message)
        else:
            self.statusBar().showMessage("Failed")
            QMessageBox.critical(self, "Error", message)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
