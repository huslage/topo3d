"""Address geocoding utilities."""

from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
import requests
import re


def geocode_address(address, timeout=10):
    """
    Geocode an address to coordinates using OpenStreetMap Nominatim.
    Tries structured search first for better accuracy, then falls back to free-form.

    Args:
        address: Address string to geocode
        timeout: Request timeout in seconds

    Returns:
        dict: Location data with coordinates and display name
    """
    # Try to parse the address into structured components for better accuracy
    structured_result = try_structured_geocode(address, timeout)
    if structured_result:
        print(f"Structured geocode result: {structured_result['lat']}, {structured_result['lon']}")
        return structured_result

    # Fall back to free-form geocoding
    try:
        geolocator = Nominatim(user_agent="topo3d")
        location = geolocator.geocode(address, timeout=timeout, addressdetails=True)

        if not location:
            raise ValueError(f"Address not found: {address}")

        print(f"Free-form geocode result: {location.latitude}, {location.longitude}")

        return {
            'address': location.address,
            'lat': location.latitude,
            'lon': location.longitude,
            'raw': location.raw
        }

    except GeopyError as e:
        raise Exception(f"Geocoding error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error geocoding address: {str(e)}")


def try_structured_geocode(address, timeout=10):
    """
    Try to geocode using structured query for better accuracy.
    Parses common address formats and uses Nominatim's structured search.
    """
    try:
        # Try to extract house number and street from the address
        # Common patterns: "38 Brolga St", "38 Brolga Street, Mount Gambier"
        match = re.match(r'^(\d+)\s+(.+?)(?:,\s*(.+))?$', address.strip())
        if not match:
            return None

        housenumber = match.group(1)
        rest = match.group(2)

        # Split the rest into street and city/state
        parts = rest.split(',')
        street = parts[0].strip()

        # Build structured query
        params = {
            'street': f"{housenumber} {street}",
            'format': 'json',
            'addressdetails': '1',
            'limit': '1'
        }

        # Add city/state if available
        if len(parts) > 1:
            # Try to identify city and state
            remaining = ', '.join(parts[1:]).strip()
            if remaining:
                params['city'] = remaining.split(',')[0].strip()
                if len(remaining.split(',')) > 1:
                    state_country = remaining.split(',')[1:]
                    for part in state_country:
                        part = part.strip()
                        if part.lower() in ['australia', 'au', 'usa', 'uk', 'canada']:
                            params['country'] = part
                        elif len(part) <= 3:
                            params['state'] = part
        elif match.group(3):
            city_state = match.group(3)
            parts = city_state.split(',')
            params['city'] = parts[0].strip()

        # Make the request
        response = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params=params,
            headers={'User-Agent': 'topo3d'},
            timeout=timeout
        )

        if response.status_code == 200:
            results = response.json()
            if results:
                result = results[0]
                return {
                    'address': result.get('display_name', address),
                    'lat': float(result['lat']),
                    'lon': float(result['lon']),
                    'raw': result
                }

    except Exception as e:
        print(f"Structured geocode failed: {e}")

    return None
