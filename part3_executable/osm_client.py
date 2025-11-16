"""
Part 3: OSM API Client
Handles authentication and interaction with OpenStreetMap API
"""

import osmapi
from typing import List, Tuple, Dict
import time


class OSMClient:
    """Client for interacting with OpenStreetMap API"""

    def __init__(self, email: str = None, password: str = None):
        """
        Initialize OSM client

        Args:
            email: OSM account email
            password: OSM account password
        """
        self.email = email
        self.password = password
        self.api = None
        self.changeset_id = None
        self.authenticated = False

    def authenticate(self, email: str, password: str) -> bool:
        """
        Authenticate with OSM API

        Args:
            email: OSM account email
            password: OSM account password

        Returns:
            True if authentication successful
        """
        try:
            self.email = email
            self.password = password

            # Initialize API with credentials
            self.api = osmapi.OsmApi(
                username=email,
                password=password,
                api="https://api.openstreetmap.org"
            )

            # Test authentication by getting user details
            user_details = self.api.UserDetails()

            if user_details:
                self.authenticated = True
                print(f"✓ Authenticated as: {user_details.get('display_name', email)}")
                return True
            else:
                self.authenticated = False
                print("✗ Authentication failed")
                return False

        except Exception as e:
            print(f"✗ Authentication error: {str(e)}")
            self.authenticated = False
            return False

    def create_changeset(self, comment: str = "Automated feature mapping") -> int:
        """
        Create a new changeset

        Args:
            comment: Changeset comment

        Returns:
            Changeset ID
        """
        if not self.authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")

        try:
            changeset = self.api.ChangesetCreate({
                "comment": comment,
                "created_by": "osmiD-AI-editor v1.0"
            })

            self.changeset_id = changeset
            print(f"✓ Created changeset: {changeset}")
            return changeset

        except Exception as e:
            print(f"✗ Error creating changeset: {str(e)}")
            raise

    def close_changeset(self):
        """Close the current changeset"""
        if self.changeset_id:
            try:
                self.api.ChangesetClose()
                print(f"✓ Closed changeset: {self.changeset_id}")
                self.changeset_id = None
            except Exception as e:
                print(f"✗ Error closing changeset: {str(e)}")

    def create_way(
        self,
        nodes: List[Tuple[float, float]],
        tags: Dict[str, str]
    ) -> int:
        """
        Create a way (polygon or line) in OSM

        Args:
            nodes: List of (lat, lon) coordinate tuples
            tags: Dictionary of OSM tags

        Returns:
            Way ID
        """
        if not self.authenticated:
            raise Exception("Not authenticated")

        if not self.changeset_id:
            self.create_changeset()

        try:
            # Create nodes
            node_ids = []

            for lat, lon in nodes:
                node = self.api.NodeCreate({
                    "lat": lat,
                    "lon": lon,
                    "tag": {}
                })
                node_ids.append(node['id'])
                time.sleep(0.1)  # Rate limiting

            # Create way
            way = self.api.WayCreate({
                "nd": node_ids,
                "tag": tags
            })

            print(f"✓ Created way: {way['id']} with {len(node_ids)} nodes")
            return way['id']

        except Exception as e:
            print(f"✗ Error creating way: {str(e)}")
            raise

    def create_building(self, coordinates: List[Tuple[float, float]]) -> int:
        """
        Create a building polygon

        Args:
            coordinates: List of (lat, lon) coordinate tuples

        Returns:
            Way ID
        """
        # Ensure polygon is closed
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        tags = {
            "building": "yes"
        }

        return self.create_way(coordinates, tags)

    def create_natural_feature(
        self,
        coordinates: List[Tuple[float, float]],
        feature_type: str
    ) -> int:
        """
        Create a natural feature (wood, water, etc.)

        Args:
            coordinates: List of (lat, lon) coordinate tuples
            feature_type: Type of natural feature

        Returns:
            Way ID
        """
        # Ensure polygon is closed
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        # Map feature types to OSM tags
        feature_tags = {
            "lawn": {"landuse": "grass", "grass": "lawn"},
            "natural_wood": {"natural": "wood"},
            "artificial_forest": {"landuse": "forest"},
            "water_body": {"natural": "water"}
        }

        if feature_type not in feature_tags:
            raise ValueError(f"Unknown feature type: {feature_type}")

        return self.create_way(coordinates, feature_tags[feature_type])

    def upload_features(
        self,
        features: List[Dict],
        changeset_comment: str = "AI-assisted feature mapping"
    ) -> List[int]:
        """
        Upload multiple features to OSM

        Args:
            features: List of feature dictionaries with 'type' and 'coordinates'
            changeset_comment: Comment for the changeset

        Returns:
            List of created way IDs
        """
        if not self.authenticated:
            raise Exception("Not authenticated")

        # Create changeset
        self.create_changeset(changeset_comment)

        way_ids = []

        try:
            for i, feature in enumerate(features):
                print(f"Uploading feature {i + 1}/{len(features)} ({feature['type']})...")

                if feature['type'] == 'building':
                    way_id = self.create_building(feature['coordinates'])
                else:
                    way_id = self.create_natural_feature(
                        feature['coordinates'],
                        feature['type']
                    )

                way_ids.append(way_id)
                time.sleep(0.5)  # Rate limiting

            print(f"\n✓ Successfully uploaded {len(way_ids)} features")

        except Exception as e:
            print(f"\n✗ Error during upload: {str(e)}")
            raise

        finally:
            # Close changeset
            self.close_changeset()

        return way_ids


if __name__ == "__main__":
    # Test client (requires valid credentials)
    client = OSMClient()

    # Note: Don't commit real credentials!
    # client.authenticate("your_email@example.com", "your_password")
