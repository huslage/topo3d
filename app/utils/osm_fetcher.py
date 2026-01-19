"""OpenStreetMap data fetching utilities using Overpass API."""

import requests
import time

# Multiple Overpass API servers for fallback
OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def fetch_osm_features(north, south, east, west, feature_types):
    """
    Fetch OpenStreetMap features for a bounding box using Overpass API.

    Args:
        north: Northern latitude bound
        south: Southern latitude bound
        east: Eastern longitude bound
        west: Western longitude bound
        feature_types: List of feature types to fetch (e.g., ['roads', 'water', 'buildings'])

    Returns:
        dict: OSM feature data organized by type
    """
    features = {
        'roads': [],
        'water': [],
        'buildings': [],
        'railways': [],
        'landuse': []
    }

    bbox = f"{south},{west},{north},{east}"

    try:
        # Fetch roads
        if 'roads' in feature_types:
            query = f"""
            [out:json][timeout:30];
            way["highway"]({bbox});
            out body geom;
            """
            roads_data = query_overpass(query)
            features['roads'] = parse_elements(roads_data, 'road')
            print(f"Fetched {len(features['roads'])} roads")

        # Fetch water features
        if 'water' in feature_types:
            query = f"""
            [out:json][timeout:30];
            (
              way["natural"="water"]({bbox});
              way["waterway"]({bbox});
              relation["natural"="water"]({bbox});
            );
            out body geom;
            """
            water_data = query_overpass(query)
            features['water'] = parse_elements(water_data, 'water')
            print(f"Fetched {len(features['water'])} water features")

        # Fetch buildings
        if 'buildings' in feature_types:
            query = f"""
            [out:json][timeout:30];
            way["building"]({bbox});
            out body geom;
            """
            buildings_data = query_overpass(query)
            features['buildings'] = parse_elements(buildings_data, 'building')
            print(f"Fetched {len(features['buildings'])} buildings")

        # Fetch railways
        if 'railways' in feature_types:
            query = f"""
            [out:json][timeout:30];
            way["railway"]({bbox});
            out body geom;
            """
            railways_data = query_overpass(query)
            features['railways'] = parse_elements(railways_data, 'railway')
            print(f"Fetched {len(features['railways'])} railways")

        return features

    except Exception as e:
        # Return empty features on error rather than failing completely
        print(f"Warning: OSM fetch error: {str(e)}")
        return features


def query_overpass(query):
    """
    Execute an Overpass API query with fallback servers.

    Args:
        query: Overpass QL query string

    Returns:
        list: Elements from the response
    """
    last_error = None

    for server in OVERPASS_SERVERS:
        try:
            print(f"Trying Overpass server: {server}")
            response = requests.post(
                server,
                data={'data': query},
                timeout=45
            )
            response.raise_for_status()
            data = response.json()
            elements = data.get('elements', [])
            print(f"Success! Got {len(elements)} elements")
            return elements

        except requests.exceptions.Timeout:
            print(f"Timeout on {server}")
            last_error = "timeout"
            continue
        except requests.exceptions.RequestException as e:
            print(f"Error on {server}: {e}")
            last_error = str(e)
            continue
        except Exception as e:
            print(f"Unexpected error on {server}: {e}")
            last_error = str(e)
            continue

    print(f"All Overpass servers failed. Last error: {last_error}")
    return []


def parse_elements(elements, feature_type):
    """
    Parse OSM elements into simplified feature format.

    Args:
        elements: List of OSM elements from Overpass API
        feature_type: Type of feature (for categorization)

    Returns:
        list: Simplified feature data
    """
    features = []

    for element in elements:
        elem_type = element.get('type')

        coordinates = []

        if elem_type == 'way':
            # Simple way - get geometry directly
            geometry = element.get('geometry', [])
            if not geometry:
                continue
            for point in geometry:
                coordinates.append({
                    'lat': point.get('lat'),
                    'lon': point.get('lon')
                })
        elif elem_type == 'relation':
            # Relation (multipolygon) - get geometry from members
            members = element.get('members', [])
            # Find the outer way(s) and extract their geometry
            for member in members:
                if member.get('role') == 'outer' and member.get('type') == 'way':
                    geometry = member.get('geometry', [])
                    for point in geometry:
                        coordinates.append({
                            'lat': point.get('lat'),
                            'lon': point.get('lon')
                        })
                    # For now, just use the first outer ring
                    break
        else:
            continue

        if len(coordinates) < 2:
            continue

        tags = element.get('tags', {})

        feature = {
            'id': element.get('id'),
            'type': feature_type,
            'coordinates': coordinates,
            'tags': tags
        }

        # Add specific attributes based on tags
        if 'name' in tags:
            feature['name'] = tags['name']

        if 'highway' in tags:
            feature['road_type'] = tags['highway']

        if 'building' in tags:
            feature['building_type'] = tags['building']

        if 'height' in tags:
            try:
                # Parse height (might be "10m" or just "10")
                height_str = tags['height'].replace('m', '').strip()
                feature['height'] = float(height_str)
            except (ValueError, AttributeError):
                feature['height'] = 10.0  # Default height

        elif 'building:levels' in tags:
            try:
                # Estimate height from levels (assume 3m per level)
                feature['height'] = float(tags['building:levels']) * 3.0
            except (ValueError, AttributeError):
                feature['height'] = 10.0
        else:
            # Default building height
            if feature_type == 'building':
                feature['height'] = 10.0

        features.append(feature)

    return features
