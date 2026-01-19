"""GPX file parsing utilities."""

import gpxpy
import gpxpy.gpx


def parse_gpx_file(filepath):
    """
    Parse a GPX file and extract track points, waypoints, and bounds.

    Args:
        filepath: Path to GPX file

    Returns:
        dict: Parsed GPX data with tracks, waypoints, and bounds
    """
    with open(filepath, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    tracks = []
    waypoints = []

    # Extract tracks
    for track in gpx.tracks:
        track_points = []
        for segment in track.segments:
            for point in segment.points:
                track_points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'elevation': point.elevation if point.elevation else 0,
                    'time': point.time.isoformat() if point.time else None
                })

        tracks.append({
            'name': track.name,
            'points': track_points
        })

    # Extract waypoints
    for waypoint in gpx.waypoints:
        waypoints.append({
            'name': waypoint.name,
            'lat': waypoint.latitude,
            'lon': waypoint.longitude,
            'elevation': waypoint.elevation if waypoint.elevation else 0,
            'description': waypoint.description
        })

    # Calculate bounds
    bounds = gpx.get_bounds()

    return {
        'tracks': tracks,
        'waypoints': waypoints,
        'bounds': {
            'north': bounds.max_latitude,
            'south': bounds.min_latitude,
            'east': bounds.max_longitude,
            'west': bounds.min_longitude
        } if bounds else None,
        'metadata': {
            'name': gpx.name,
            'description': gpx.description,
            'author': gpx.author_name
        }
    }
