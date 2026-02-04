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


def generate_feature_name(tags, feature_type, feature_id):
    """
    Generate a descriptive name for a feature from OSM tags.

    Args:
        tags: Dictionary of OSM tags
        feature_type: Type of feature (road, building, water, etc.)
        feature_id: OSM feature ID (fallback if no name available)

    Returns:
        str: Descriptive feature name
    """
    # Road type label mapping
    ROAD_TYPE_LABELS = {
        'motorway': 'Motorway',
        'trunk': 'Trunk Road',
        'primary': 'Primary Road',
        'secondary': 'Secondary Road',
        'tertiary': 'Tertiary Road',
        'unclassified': 'Road',
        'residential': 'Residential Street',
        'service': 'Service Road',
        'motorway_link': 'Motorway Link',
        'trunk_link': 'Trunk Link',
        'primary_link': 'Primary Link',
        'secondary_link': 'Secondary Link',
        'tertiary_link': 'Tertiary Link',
        'living_street': 'Living Street',
        'pedestrian': 'Pedestrian Way',
        'track': 'Track',
        'bus_guideway': 'Bus Guideway',
        'escape': 'Emergency Escape',
        'raceway': 'Raceway',
        'road': 'Road',
        'footway': 'Footway',
        'bridleway': 'Bridleway',
        'steps': 'Steps',
        'path': 'Path',
        'cycleway': 'Cycle Way'
    }

    # Water type label mapping
    WATER_TYPE_LABELS = {
        'river': 'River',
        'stream': 'Stream',
        'canal': 'Canal',
        'drain': 'Drain',
        'ditch': 'Ditch',
        'lake': 'Lake',
        'pond': 'Pond',
        'reservoir': 'Reservoir',
        'basin': 'Basin'
    }

    name = tags.get('name')
    type_label = None

    # Generate type label based on feature type
    if feature_type == 'road':
        highway_type = tags.get('highway')
        if highway_type:
            type_label = ROAD_TYPE_LABELS.get(highway_type, highway_type.title())

    elif feature_type == 'building':
        # Check for building amenity/shop type
        if 'amenity' in tags:
            type_label = tags['amenity'].replace('_', ' ').title()
        elif 'shop' in tags:
            type_label = tags['shop'].replace('_', ' ').title() + ' Shop'
        elif 'building' in tags and tags['building'] not in ['yes', 'true']:
            type_label = tags['building'].replace('_', ' ').title()
        else:
            type_label = 'Building'

    elif feature_type == 'water':
        # Check for water type
        if 'waterway' in tags:
            waterway_type = tags['waterway']
            type_label = WATER_TYPE_LABELS.get(waterway_type, waterway_type.title())
        elif 'water' in tags and tags['water'] not in ['yes', 'true']:
            type_label = tags['water'].replace('_', ' ').title()
        else:
            type_label = 'Water Body'

    elif feature_type == 'railway':
        railway_type = tags.get('railway', 'railway')
        type_label = railway_type.replace('_', ' ').title()

    else:
        type_label = feature_type.title()

    # Format: "{name} ({type})" when both available, otherwise just what we have
    if name and type_label:
        return f"{name} ({type_label})"
    elif name:
        return name
    elif type_label:
        return type_label
    else:
        # Fallback to feature type and ID
        return f"{feature_type}_{feature_id}"


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
        feature_id = element.get('id')

        feature = {
            'id': feature_id,
            'type': feature_type,
            'coordinates': coordinates,
            'tags': tags,
            'name': generate_feature_name(tags, feature_type, feature_id)
        }

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
