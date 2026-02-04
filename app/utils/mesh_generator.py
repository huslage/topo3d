"""3D mesh generation utilities."""

import numpy as np
import zipfile
import io
from stl import mesh
from scipy.spatial import Delaunay
from .shape_clipper import CircleClipper, SquareClipper, RectangleClipper, HexagonClipper
from .building_shapes import BuildingShapeGenerator


def generate_mesh(elevation_data, features, options):
    """
    Generate 3D mesh from elevation data and map features.

    Args:
        elevation_data: Dict with elevation grid and coordinates
        features: Dict with OSM features (roads, water, buildings)
        options: Dict with generation options (scale, extrusion, etc.)

    Returns:
        dict: Mesh data with vertices, faces, and metadata
    """
    try:
        # Extract parameters
        elevation_grid = np.array(elevation_data['grid'])
        lats = np.array(elevation_data['lats'])
        lons = np.array(elevation_data['lons'])
        bounds = elevation_data['bounds']

        # Options
        vertical_scale = options.get('vertical_scale', 1.5)
        base_height = options.get('base_height', 10.0)  # mm
        model_width = options.get('model_width', 200.0)  # mm
        include_base = options.get('include_base', True)

        # Shape selection (backward compatible with circular_model)
        model_shape = options.get('model_shape', 'square')
        if options.get('circular_model', False):
            model_shape = 'circle'

        # Calculate scaling factors
        lat_range = bounds['north'] - bounds['south']
        lon_range = bounds['east'] - bounds['west']

        # Aspect ratio correction (latitude vs longitude)
        avg_lat = (bounds['north'] + bounds['south']) / 2
        lon_scale = np.cos(np.radians(avg_lat))

        # Scale to model size
        scale_factor = model_width / max(lat_range, lon_range * lon_scale)

        # Compute elevation normalization (used by terrain and features)
        min_elev = np.min(elevation_grid)
        max_elev = np.max(elevation_grid)
        elev_range = max_elev - min_elev if max_elev > min_elev else 1.0

        # Instantiate shape clipper
        lat_scaled = lat_range * scale_factor
        lon_scaled = lon_range * lon_scale * scale_factor
        center_x = lon_scaled / 2
        center_z = lat_scaled / 2

        if model_shape == 'circle':
            radius = min(lon_scaled, lat_scaled) / 2
            shape_clipper = CircleClipper(center_x, center_z, radius)
        elif model_shape == 'rectangle':
            # Auto aspect ratio from bounds
            half_width = lon_scaled / 2
            half_height = lat_scaled / 2
            shape_clipper = RectangleClipper(center_x, center_z, half_width, half_height)
        elif model_shape == 'hexagon':
            radius = min(lon_scaled, lat_scaled) / 2
            shape_clipper = HexagonClipper(center_x, center_z, radius)
        else:  # 'square' or default
            half_width = min(lon_scaled, lat_scaled) / 2
            shape_clipper = SquareClipper(center_x, center_z, half_width)

        # Generate terrain mesh
        terrain_mesh = generate_terrain_mesh(
            elevation_grid,
            lats,
            lons,
            bounds,
            scale_factor,
            vertical_scale,
            base_height,
            include_base,
            shape_clipper
        )

        # Add features (roads, buildings, water)
        # Pass normalization params for consistent elevation mapping
        elev_params = {
            'min_elev': min_elev,
            'elev_range': elev_range,
            'avg_lat': avg_lat
        }

        feature_meshes = []

        if features.get('buildings'):
            # Get address location options
            address_location = options.get('address_location')
            show_only_address_building = options.get('show_only_address_building', False)
            custom_building_colors = options.get('custom_building_colors', {})

            building_meshes = generate_building_meshes(
                features['buildings'],
                elevation_data,
                bounds,
                scale_factor,
                vertical_scale,
                elev_params,
                options.get('building_height_scale', 1.0),
                address_location,
                show_only_address_building,
                shape_clipper,
                custom_building_colors
            )
            feature_meshes.extend(building_meshes)

        if features.get('roads'):
            road_meshes = generate_road_meshes(
                features['roads'],
                elevation_data,
                bounds,
                scale_factor,
                vertical_scale,
                elev_params,
                options.get('road_height', 0.2),  # Reduced offset to sit closer to terrain
                shape_clipper
            )
            feature_meshes.extend(road_meshes)

        if features.get('water'):
            water_meshes = generate_water_meshes(
                features['water'],
                elevation_data,
                bounds,
                scale_factor,
                vertical_scale,
                elev_params,
                shape_clipper
            )
            feature_meshes.extend(water_meshes)

        # Generate GPX track mesh if provided
        gpx_track_mesh = None
        if options.get('gpx_tracks'):
            gpx_track_mesh = generate_gpx_track_mesh(
                options['gpx_tracks'],
                elevation_data,
                bounds,
                scale_factor,
                vertical_scale,
                elev_params,
                shape_clipper
            )

        return {
            'terrain': terrain_mesh,
            'features': feature_meshes,
            'gpx_track': gpx_track_mesh,
            'bounds': bounds,
            'scale': {
                'factor': scale_factor,
                'vertical': vertical_scale,
                'width_mm': model_width
            },
            'metadata': {
                'vertices_count': len(terrain_mesh['vertices']),
                'faces_count': len(terrain_mesh['faces']),
                'features_count': len(feature_meshes)
            }
        }

    except Exception as e:
        raise Exception(f"Error generating mesh: {str(e)}")


def generate_terrain_mesh(elevation_grid, lats, lons, bounds, scale_factor, vertical_scale, base_height, include_base, shape_clipper=None):
    """
    Generate terrain mesh from elevation data.

    Args:
        elevation_grid: 2D numpy array of elevation values
        lats: Array of latitude values
        lons: Array of longitude values
        bounds: Dict with north, south, east, west bounds
        scale_factor: Scaling factor for coordinates
        vertical_scale: Vertical exaggeration factor
        base_height: Height of base platform in mm
        include_base: Whether to include base and walls
        shape_clipper: ShapeClipper instance for boundary clipping (None = no clipping)
    """
    rows, cols = elevation_grid.shape

    # Average latitude for longitude scaling
    avg_lat = (bounds['north'] + bounds['south']) / 2

    # Normalize elevations
    min_elev = np.min(elevation_grid)
    max_elev = np.max(elevation_grid)
    elev_range = max_elev - min_elev if max_elev > min_elev else 1.0

    # Generate vertices
    # Three.js coordinate system: X = east-west, Y = up (elevation), Z = south-north (flipped for map view)
    vertices = []
    for i in range(rows):
        for j in range(cols):
            lat = lats[i]
            lon = lons[j]

            # Convert to model coordinates
            # North at z=0, south at higher z for standard map orientation when viewed from above
            x = (lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
            z = (bounds['north'] - lat) * scale_factor

            # Elevation with vertical exaggeration (Y is up in Three.js)
            y = ((elevation_grid[i, j] - min_elev) / elev_range) * 20.0 * vertical_scale

            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Create mask for vertices inside the shape boundary (if shape clipper provided)
    if shape_clipper:
        x_coords = vertices[:, 0]
        z_coords = vertices[:, 2]
        inside_shape = shape_clipper.is_inside(x_coords, z_coords)
    else:
        inside_shape = np.ones(len(vertices), dtype=bool)

    # Generate faces (triangles)
    # Winding: CCW when viewed from above (+Y) so normals point up
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Two triangles per quad
            idx = i * cols + j
            idx_right = idx + 1
            idx_down = idx + cols
            idx_diag = idx + cols + 1

            if shape_clipper:
                # Only include faces where all vertices are inside the shape
                if inside_shape[idx] and inside_shape[idx_down] and inside_shape[idx_right]:
                    faces.append([idx, idx_down, idx_right])
                if inside_shape[idx_right] and inside_shape[idx_down] and inside_shape[idx_diag]:
                    faces.append([idx_right, idx_down, idx_diag])
            else:
                # Triangle 1: CCW from above
                faces.append([idx, idx_down, idx_right])
                # Triangle 2: CCW from above
                faces.append([idx_right, idx_down, idx_diag])

    faces = np.array(faces) if faces else np.array([]).reshape(0, 3)

    # Add base if requested
    if include_base:
        if shape_clipper:
            base_vertices, base_faces = generate_shape_base(vertices, base_height, shape_clipper)
        else:
            base_vertices, base_faces = generate_base(vertices, base_height, rows, cols)
        vertices = np.vstack([vertices, base_vertices])
        faces = np.vstack([faces, base_faces]) if len(faces) > 0 else base_faces

    return {
        'vertices': vertices.tolist(),
        'faces': faces.tolist(),
        'bounds': {
            'min': vertices.min(axis=0).tolist(),
            'max': vertices.max(axis=0).tolist()
        }
    }


def generate_base(vertices, base_height, rows, cols):
    """Generate watertight base for 3D printing.

    All faces use CCW winding when viewed from outside the mesh.
    The terrain surface uses CCW when viewed from above (+Y).
    """
    base_vertices = []
    base_faces = []

    # Bottom vertices (project top vertices down - Y is vertical in Three.js)
    for v in vertices:
        base_vertices.append([v[0], -base_height, v[2]])

    base_vertices = np.array(base_vertices)

    # Bottom vertices start at index len(vertices) in the combined array
    n = len(vertices)  # num_top_vertices

    # Side faces - connect terrain edges to base edges
    # Winding: CCW when viewed from outside (normals point outward)

    # West wall (left edge, j=0) - normal points -X
    # Looking from -X: top vertices go from low i to high i (back to front in Z)
    # CCW from -X view: top-high-i, top-low-i, bot-low-i, bot-high-i
    for i in range(rows - 1):
        t0 = i * cols           # top, current row
        t1 = (i + 1) * cols     # top, next row
        b0 = n + t0             # bottom, current row
        b1 = n + t1             # bottom, next row
        # Two triangles, CCW when viewed from -X
        base_faces.append([t1, t0, b0])
        base_faces.append([t1, b0, b1])

    # East wall (right edge, j=cols-1) - normal points +X
    # Looking from +X: top vertices go from high i to low i
    # CCW from +X view: top-low-i, top-high-i, bot-high-i, bot-low-i
    for i in range(rows - 1):
        t0 = i * cols + (cols - 1)
        t1 = (i + 1) * cols + (cols - 1)
        b0 = n + t0
        b1 = n + t1
        # Two triangles, CCW when viewed from +X
        base_faces.append([t0, t1, b1])
        base_faces.append([t0, b1, b0])

    # North wall (back edge, i=0, low Z) - normal points -Z
    # Looking from -Z: top vertices go from high j to low j (right to left in X)
    # CCW from -Z view: top-low-j, top-high-j, bot-high-j, bot-low-j
    for j in range(cols - 1):
        t0 = j                  # top, current col
        t1 = j + 1              # top, next col
        b0 = n + t0             # bottom, current col
        b1 = n + t1             # bottom, next col
        # Two triangles, CCW when viewed from -Z
        base_faces.append([t0, t1, b1])
        base_faces.append([t0, b1, b0])

    # South wall (front edge, i=rows-1, high Z) - normal points +Z
    # Looking from +Z: top vertices go from low j to high j
    # CCW from +Z view: top-high-j, top-low-j, bot-low-j, bot-high-j
    for j in range(cols - 1):
        t0 = (rows - 1) * cols + j
        t1 = (rows - 1) * cols + j + 1
        b0 = n + t0
        b1 = n + t1
        # Two triangles, CCW when viewed from +Z
        base_faces.append([t1, t0, b0])
        base_faces.append([t1, b0, b1])

    # Bottom face - normal points -Y (down)
    # CCW when viewed from below means CW when viewed from above
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = n + i * cols + j
            # CW from above = CCW from below
            base_faces.append([idx, idx + 1, idx + cols])
            base_faces.append([idx + 1, idx + cols + 1, idx + cols])

    return base_vertices, np.array(base_faces)


def generate_shape_base(vertices, base_height, shape_clipper):
    """
    Generate watertight base for any shape using shape clipper.

    Creates a smooth wall that follows the terrain contour and a bottom face.

    Args:
        vertices: Terrain vertices array
        base_height: Height of base platform
        shape_clipper: ShapeClipper instance for wall generation

    Returns:
        tuple: (base_vertices, base_faces) as numpy arrays
    """
    base_vertices = []
    base_faces = []

    n = len(vertices)  # Number of terrain vertices

    # Create elevation interpolation function for the wall
    def terrain_elevation_func(x, z):
        """Interpolate elevation at (x, z) from nearest terrain vertices."""
        # Find closest terrain vertex (simple nearest neighbor)
        min_dist = float('inf')
        closest_elev = -base_height

        for v in vertices:
            vx, vy, vz = v
            dist_sq = (vx - x)**2 + (vz - z)**2
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_elev = vy

        return closest_elev

    # Generate wall vertices using shape clipper
    wall_vertices, wall_faces = shape_clipper.generate_wall_vertices(
        terrain_elevation_func,
        base_height
    )

    # Offset wall face indices by number of terrain vertices
    wall_faces_offset = wall_faces + n

    # Add wall vertices to base
    base_vertices.extend(wall_vertices.tolist())

    # Add wall faces
    base_faces.extend(wall_faces_offset.tolist())

    # Create center point for bottom face
    center_x = shape_clipper.center_x
    center_z = shape_clipper.center_z
    center_bottom_idx = n + len(wall_vertices)
    base_vertices.append([center_x, -base_height, center_z])

    # Create bottom face using fan triangulation from center
    # Wall vertices alternate: top, bottom, top, bottom, ...
    # We need to connect bottom vertices in a circular pattern
    num_wall_segments = len(wall_vertices) // 2

    for i in range(num_wall_segments):
        next_i = (i + 1) % num_wall_segments

        # Bottom vertices are at odd indices: 1, 3, 5, ...
        bottom_i = n + i * 2 + 1
        bottom_next = n + next_i * 2 + 1

        # Fan triangle: center, next_bottom, current_bottom (CCW from below)
        base_faces.append([center_bottom_idx, bottom_next, bottom_i])

    return np.array(base_vertices), np.array(base_faces, dtype=np.int32)


def generate_circular_base(vertices, base_height, rows, cols, center_x, center_z, radius, inside_circle):
    """Generate watertight circular base for 3D printing.

    Creates a smooth circular wall that follows the terrain contour.
    Adds a hidden skirt below the terrain surface to fill the gap between
    the jagged terrain boundary and the smooth circular wall.
    """
    base_faces = []

    # Don't modify terrain vertices
    vertices = vertices.copy()

    n = len(vertices)  # Number of terrain vertices

    # Create smooth circle vertices for the wall
    num_circle_segments = 360

    # Find boundary vertices and their angles for elevation interpolation
    boundary_data = []  # List of (angle, elevation, x, z)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if not inside_circle[idx]:
                continue
            # Check if this vertex has any neighbor outside the circle
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * cols + j)
            if i < rows - 1:
                neighbors.append((i + 1) * cols + j)
            if j > 0:
                neighbors.append(i * cols + j - 1)
            if j < cols - 1:
                neighbors.append(i * cols + j + 1)
            for neighbor_idx in neighbors:
                if not inside_circle[neighbor_idx]:
                    # This is a boundary vertex
                    v = vertices[idx]
                    angle = np.arctan2(v[2] - center_z, v[0] - center_x)
                    boundary_data.append((angle, v[1], v[0], v[2], idx))
                    break

    # Sort boundary data by angle for interpolation
    boundary_data.sort(key=lambda x: x[0])

    # Vertex layout:
    # - Outer top circle (wall top edge): n to n + num_segments - 1
    # - Outer bottom circle (wall base): n + num_segments to n + 2*num_segments - 1
    # - Center bottom vertex: n + 2*num_segments
    outer_top_start = n
    outer_bottom_start = n + num_circle_segments
    center_bottom_idx = n + num_circle_segments * 2

    outer_top = []      # Wall top on exact circle
    outer_bottom = []   # Wall bottom

    for i in range(num_circle_segments):
        angle = 2 * np.pi * i / num_circle_segments - np.pi  # -pi to pi

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Outer circle at full radius (smooth wall edge)
        outer_x = center_x + radius * cos_a
        outer_z = center_z + radius * sin_a

        # Interpolate elevation from boundary vertices
        elev = interpolate_boundary_elevation(angle, boundary_data)

        # Outer top follows terrain contour
        outer_top.append([outer_x, elev, outer_z])

        # Outer bottom at base
        outer_bottom.append([outer_x, -base_height, outer_z])

    # Build base_vertices array
    base_vertices = outer_top + outer_bottom + [[center_x, -base_height, center_z]]

    # Create outer wall faces (from outer top to outer bottom)
    for i in range(num_circle_segments):
        next_i = (i + 1) % num_circle_segments

        ot0 = outer_top_start + i
        ot1 = outer_top_start + next_i
        ob0 = outer_bottom_start + i
        ob1 = outer_bottom_start + next_i

        # Two triangles per wall segment (CCW when viewed from outside)
        base_faces.append([ot0, ob0, ot1])
        base_faces.append([ot1, ob0, ob1])

    # Create bottom face using fan triangulation
    for i in range(num_circle_segments):
        next_i = (i + 1) % num_circle_segments
        ob0 = outer_bottom_start + i
        ob1 = outer_bottom_start + next_i

        # CCW when viewed from below
        base_faces.append([center_bottom_idx, ob1, ob0])

    # Move ALL boundary vertices outward to exactly match the wall circle
    # This stretches the terrain mesh to close the gap with the smooth wall
    for data in boundary_data:
        idx = data[4]  # Vertex index stored in boundary_data
        v = vertices[idx]
        dx = v[0] - center_x
        dz = v[2] - center_z
        dist = np.sqrt(dx * dx + dz * dz)
        if dist > 0:
            scale = radius / dist
            vertices[idx][0] = center_x + dx * scale
            vertices[idx][2] = center_z + dz * scale

    return np.array(base_vertices), np.array(base_faces), vertices


def interpolate_boundary_elevation(angle, boundary_with_angles):
    """Interpolate elevation at a given angle from sorted boundary vertices.

    boundary_with_angles is a list of tuples: (angle, elevation, ...) - only first two elements used.
    """
    n = len(boundary_with_angles)
    if n == 0:
        return 0

    # Find bracketing boundary vertices
    for i in range(n):
        if boundary_with_angles[i][0] >= angle:
            # Found upper bracket
            upper_idx = i
            lower_idx = (i - 1) % n
            break
    else:
        # Angle is past all boundary angles, wrap around
        upper_idx = 0
        lower_idx = n - 1

    lower_angle, lower_elev = boundary_with_angles[lower_idx][0], boundary_with_angles[lower_idx][1]
    upper_angle, upper_elev = boundary_with_angles[upper_idx][0], boundary_with_angles[upper_idx][1]

    # Handle wraparound
    if upper_angle < lower_angle:
        if angle < 0:
            upper_angle_adj = upper_angle
            lower_angle_adj = lower_angle - 2 * np.pi
        else:
            upper_angle_adj = upper_angle + 2 * np.pi
            lower_angle_adj = lower_angle
    else:
        upper_angle_adj = upper_angle
        lower_angle_adj = lower_angle

    # Linear interpolation
    if abs(upper_angle_adj - lower_angle_adj) < 1e-6:
        return lower_elev

    t = (angle - lower_angle_adj) / (upper_angle_adj - lower_angle_adj)
    t = max(0, min(1, t))

    return lower_elev + t * (upper_elev - lower_elev)


def create_box(x1, x2, y1, y2, z1, z2):
    """Create a simple box mesh from min/max coordinates."""
    # 8 vertices of the box
    vertices = [
        [x1, y1, z1],  # 0: bottom-front-left
        [x2, y1, z1],  # 1: bottom-front-right
        [x2, y1, z2],  # 2: bottom-back-right
        [x1, y1, z2],  # 3: bottom-back-left
        [x1, y2, z1],  # 4: top-front-left
        [x2, y2, z1],  # 5: top-front-right
        [x2, y2, z2],  # 6: top-back-right
        [x1, y2, z2],  # 7: top-back-left
    ]

    # 12 triangles (2 per face, 6 faces)
    faces = [
        # Bottom face
        [0, 2, 1], [0, 3, 2],
        # Top face
        [4, 5, 6], [4, 6, 7],
        # Front face
        [0, 1, 5], [0, 5, 4],
        # Back face
        [2, 3, 7], [2, 7, 6],
        # Left face
        [0, 4, 7], [0, 7, 3],
        # Right face
        [1, 2, 6], [1, 6, 5],
    ]

    return {'vertices': vertices, 'faces': faces}


def generate_building_meshes(buildings, elevation_data, bounds, scale_factor, vertical_scale, elev_params, height_scale, address_location=None, show_only_address_building=False, shape_clipper=None, custom_building_colors=None):
    """
    Generate 3D meshes for buildings with customizable shapes and colors.

    Args:
        buildings: List of building features
        elevation_data: Elevation grid data
        bounds: Geographic bounds
        scale_factor: Scale factor for model
        vertical_scale: Vertical exaggeration
        elev_params: Elevation normalization params
        height_scale: Building height multiplier
        address_location: Optional dict with 'lat' and 'lon' of highlighted address
        show_only_address_building: If True, only return the building at the address
        shape_clipper: ShapeClipper for boundary clipping (None = no clipping)
        custom_building_colors: Dict mapping building IDs to custom hex colors

    Returns:
        list: Building mesh dictionaries
    """
    if custom_building_colors is None:
        custom_building_colors = {}
    meshes = []
    min_elev = elev_params['min_elev']
    elev_range = elev_params['elev_range']
    avg_lat = elev_params['avg_lat']

    # Initialize building shape generator
    building_shape_gen = BuildingShapeGenerator()

    # Find the building closest to the address location
    address_building_id = None
    if address_location:
        addr_lat = address_location.get('lat')
        addr_lon = address_location.get('lon')
        if addr_lat is not None and addr_lon is not None:
            min_distance = float('inf')
            for building in buildings:
                coords = building['coordinates']
                if len(coords) < 3:
                    continue

                # Calculate center of building
                lats = [c['lat'] for c in coords]
                lons = [c['lon'] for c in coords]
                center_lat = (min(lats) + max(lats)) / 2
                center_lon = (min(lons) + max(lons)) / 2

                # Check if address is inside the building's bounding box
                if (min(lats) <= addr_lat <= max(lats) and
                    min(lons) <= addr_lon <= max(lons)):
                    # Address is inside this building - prefer this one
                    address_building_id = building['id']
                    break

                # Calculate distance to center
                distance = ((addr_lat - center_lat) ** 2 + (addr_lon - center_lon) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    address_building_id = building['id']

            print(f"Address building ID: {address_building_id}, distance: {min_distance if address_building_id else 'N/A'}")

    for building in buildings[:150]:  # Limit to avoid too many polygons
        coords = building['coordinates']
        if len(coords) < 3:
            continue

        is_address_building = (building['id'] == address_building_id)

        # If showing only address building, skip all others
        if show_only_address_building and not is_address_building:
            continue

        # Calculate bounding box of building
        lats = [c['lat'] for c in coords]
        lons = [c['lon'] for c in coords]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Check if building is within bounds
        if max_lat < bounds['south'] or min_lat > bounds['north']:
            continue
        if max_lon < bounds['west'] or min_lon > bounds['east']:
            continue

        # Clamp to bounds
        min_lat = max(bounds['south'], min_lat)
        max_lat = min(bounds['north'], max_lat)
        min_lon = max(bounds['west'], min_lon)
        max_lon = min(bounds['east'], max_lon)

        # Get building height (scale to model units)
        # Make address building taller to stand out more
        height = building.get('height', 8.0) * height_scale * 0.15
        if is_address_building:
            height = max(height, 3.0)  # Ensure minimum height for visibility

        # Convert bounding box corners to model space
        x1 = (min_lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
        x2 = (max_lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
        z1 = (bounds['north'] - max_lat) * scale_factor
        z2 = (bounds['north'] - min_lat) * scale_factor

        # Check if building is within shape boundary
        # Check all 4 corners - if any corner is outside, skip the building
        if shape_clipper:
            corners = [(x1, z1), (x1, z2), (x2, z1), (x2, z2)]
            outside = False
            for cx, cz in corners:
                if not shape_clipper.is_inside(cx, cz):
                    outside = True
                    break
            if outside:
                continue  # Skip buildings that extend outside the shape

        # Get base elevation at center of building
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        raw_elev = interpolate_elevation(center_lat, center_lon, elevation_data)
        base_y = ((raw_elev - min_elev) / elev_range) * 20.0 * vertical_scale

        # Determine building shape based on tags
        building_tags = building.get('tags', {})
        shape_type = building_shape_gen.determine_building_shape(building_tags)

        # Get custom color for this building (if any)
        building_id_str = str(building['id'])
        custom_color = custom_building_colors.get(building_id_str)

        # Generate building mesh with appropriate shape and color
        building_mesh = building_shape_gen.generate_building_mesh(
            x1, x2, base_y, base_y + height, z1, z2,
            shape_type=shape_type,
            custom_color=custom_color
        )

        meshes.append({
            'type': 'building',
            'id': building['id'],
            'name': building.get('name', ''),
            'building_type': building.get('building_type', 'yes'),
            'vertices': building_mesh['vertices'],
            'faces': building_mesh['faces'],
            'is_address_building': is_address_building,
            'custom_color': building_mesh.get('custom_color')
        })

    return meshes


def generate_road_meshes(roads, elevation_data, bounds, scale_factor, vertical_scale, elev_params, road_height, shape_clipper=None):
    """
    Generate 3D meshes for roads.

    Args:
        roads: List of road features
        elevation_data: Elevation grid data
        bounds: Geographic bounds
        scale_factor: Coordinate scaling factor
        vertical_scale: Vertical exaggeration
        elev_params: Elevation normalization parameters
        road_height: Height offset above terrain
        shape_clipper: ShapeClipper for boundary clipping (None = no clipping)

    Returns:
        list: Road mesh dictionaries
    """
    meshes = []
    min_elev = elev_params['min_elev']
    elev_range = elev_params['elev_range']
    avg_lat = elev_params['avg_lat']

    for road in roads[:200]:  # Limit to avoid too many
        coords = road['coordinates']
        if len(coords) < 2:
            continue

        # Convert ALL coordinates to model space first
        points_xz = []  # (x, z) pairs for clipping
        points_xyz = []  # Full (x, y, z) points

        for coord in coords:
            lat = coord['lat']
            lon = coord['lon']

            # Skip points outside bounds
            if lat < bounds['south'] or lat > bounds['north']:
                continue
            if lon < bounds['west'] or lon > bounds['east']:
                continue

            # Use avg_lat for consistent longitude scaling (same as terrain)
            x = (lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
            z = (bounds['north'] - lat) * scale_factor

            # Get normalized elevation and raise slightly above terrain
            raw_elev = interpolate_elevation(lat, lon, elevation_data)
            y = ((raw_elev - min_elev) / elev_range) * 20.0 * vertical_scale + road_height

            points_xz.append((x, z))
            points_xyz.append([x, y, z])

        if len(points_xz) < 2:
            continue

        # Clip path to shape boundary (preserving continuity)
        if shape_clipper:
            clipped_segments = shape_clipper.clip_linestring(points_xz)

            # Process each clipped segment
            for segment in clipped_segments:
                if len(segment) < 2:
                    continue

                # Find corresponding 3D points (with elevation)
                segment_points_3d = []
                for sx, sz in segment:
                    # Find closest original point or interpolate
                    # For simplicity, find nearest point
                    min_dist = float('inf')
                    closest_point = None
                    for px, py, pz in points_xyz:
                        dist_sq = (px - sx)**2 + (pz - sz)**2
                        if dist_sq < min_dist:
                            min_dist = dist_sq
                            closest_point = [px, py, pz]

                    if closest_point:
                        # Use the segment's x, z but preserve the elevation from nearest point
                        segment_points_3d.append([sx, closest_point[1], sz])

                if len(segment_points_3d) < 2:
                    continue

                segment_points_3d = np.array(segment_points_3d)

                # Create road strip for this segment
                road_mesh = create_road_strip(segment_points_3d, width=1.0)

                meshes.append({
                    'type': 'road',
                    'id': road['id'],
                    'name': road.get('name', f"road_{road['id']}"),
                    'road_type': road.get('road_type', 'unknown'),
                    'vertices': road_mesh['vertices'],
                    'faces': road_mesh['faces']
                })
        else:
            # No clipping - use all points
            points_xyz = np.array(points_xyz)

            # Create road as line with width
            road_mesh = create_road_strip(points_xyz, width=1.0)

            meshes.append({
                'type': 'road',
                'id': road['id'],
                'name': road.get('name', f"road_{road['id']}"),
                'road_type': road.get('road_type', 'unknown'),
                'vertices': road_mesh['vertices'],
                'faces': road_mesh['faces']
            })

    return meshes


def generate_water_meshes(water_features, elevation_data, bounds, scale_factor, vertical_scale, elev_params, shape_clipper=None):
    """
    Generate 3D meshes for water bodies.

    Args:
        water_features: List of water features
        elevation_data: Elevation grid data
        bounds: Geographic bounds
        scale_factor: Coordinate scaling factor
        vertical_scale: Vertical exaggeration
        elev_params: Elevation normalization parameters
        shape_clipper: ShapeClipper for boundary clipping (None = no clipping)

    Returns:
        list: Water mesh dictionaries
    """
    meshes = []
    min_elev = elev_params['min_elev']
    elev_range = elev_params['elev_range']
    avg_lat = elev_params['avg_lat']

    for water in water_features[:50]:  # Limit water features
        coords = water['coordinates']
        if len(coords) < 3:
            continue

        # Check if water is within bounds (check first coord)
        first_coord = coords[0]
        if first_coord['lat'] < bounds['south'] or first_coord['lat'] > bounds['north']:
            continue
        if first_coord['lon'] < bounds['west'] or first_coord['lon'] > bounds['east']:
            continue

        # Convert coordinates to model space (Three.js: X=east-west, Y=up, Z=north-south)
        # For water, use minimum elevation of all points to create flat surface
        min_water_elev = float('inf')
        for coord in coords:
            lat = max(bounds['south'], min(bounds['north'], coord['lat']))
            lon = max(bounds['west'], min(bounds['east'], coord['lon']))
            raw_elev = interpolate_elevation(lat, lon, elevation_data)
            min_water_elev = min(min_water_elev, raw_elev)

        # Normalize the water level - ensure it's always visible above terrain base (Y=0)
        # Don't subtract offset for lakes at minimum elevation (like crater lakes)
        water_y = ((min_water_elev - min_elev) / elev_range) * 20.0 * vertical_scale
        # Ensure water is at least at Y=0.2 so it's visible on the surface
        water_y = max(0.2, water_y - 0.3)  # Slight offset to sit in terrain, but never below Y=0.2

        # Convert all coordinates to model space
        points_xz = []  # (x, z) for clipping
        for coord in coords:
            lat = coord['lat']
            lon = coord['lon']

            # Clamp to bounds
            lat = max(bounds['south'], min(bounds['north'], lat))
            lon = max(bounds['west'], min(bounds['east'], lon))

            # Use avg_lat for consistent longitude scaling (same as terrain)
            x = (lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
            z = (bounds['north'] - lat) * scale_factor

            points_xz.append((x, z))

        # Clip polygon to shape boundary
        if shape_clipper:
            clipped_polygon = shape_clipper.clip_polygon(points_xz)
            if clipped_polygon is None or len(clipped_polygon) < 3:
                continue

            # Create 3D points with water elevation
            points_3d = []
            for x, z in clipped_polygon:
                points_3d.append([x, water_y, z])

            points = np.array(points_3d)
        else:
            # No clipping - use all points
            points = np.array([[x, water_y, z] for x, z in points_xz])

        # Skip if not enough points
        if len(points) < 3:
            continue

        # Create solid water body with thickness for 3D printing
        water_mesh = create_solid_polygon(points, thickness=0.5)

        meshes.append({
            'type': 'water',
            'id': water['id'],
            'name': water.get('name', ''),
            'vertices': water_mesh['vertices'],
            'faces': water_mesh['faces']
        })

    return meshes


def generate_gpx_track_mesh(gpx_tracks, elevation_data, bounds, scale_factor, vertical_scale, elev_params, shape_clipper=None):
    """
    Generate 3D mesh for GPX track as a tube/ribbon following the terrain.

    Args:
        gpx_tracks: List of tracks with points [{lat, lon, elevation}, ...]
        elevation_data: Elevation grid data
        bounds: Geographic bounds
        scale_factor: Scale factor for model
        vertical_scale: Vertical exaggeration
        elev_params: Elevation normalization params
        shape_clipper: ShapeClipper for boundary clipping (None = no clipping)

    Returns:
        dict: Mesh data with vertices and faces
    """
    min_elev = elev_params['min_elev']
    elev_range = elev_params['elev_range']
    avg_lat = elev_params['avg_lat']

    # Convert all track points to model space first
    all_points_xz = []
    all_points_xyz = []

    for track in gpx_tracks:
        track_points = track.get('points', [])

        # Sample points to reduce density (every Nth point for performance)
        sample_rate = max(1, len(track_points) // 500)  # Max ~500 points per track

        for i, point in enumerate(track_points):
            if i % sample_rate != 0 and i != len(track_points) - 1:
                continue

            lat = point['lat']
            lon = point['lon']

            # Skip if outside bounds
            if lat < bounds['south'] or lat > bounds['north']:
                continue
            if lon < bounds['west'] or lon > bounds['east']:
                continue

            # Get terrain elevation at this point
            raw_elev = interpolate_elevation(lat, lon, elevation_data)

            # Convert to model coordinates (Three.js: X=east-west, Y=up, Z=north-south)
            x = (lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
            z = (bounds['north'] - lat) * scale_factor

            # Elevation - sit above roads (roads are at +0.2, so GPX at +0.5)
            y = ((raw_elev - min_elev) / elev_range) * 20.0 * vertical_scale + 0.5  # Above roads

            all_points_xz.append((x, z))
            all_points_xyz.append([x, y, z])

    if len(all_points_xz) < 2:
        return None

    # Clip track to shape boundary (preserving continuity)
    if shape_clipper:
        clipped_segments = shape_clipper.clip_linestring(all_points_xz)

        # Combine all segments for GPX track (or process separately if needed)
        all_clipped_points = []
        for segment in clipped_segments:
            if len(segment) < 2:
                continue

            # Find corresponding 3D points
            for sx, sz in segment:
                min_dist = float('inf')
                closest_point = None
                for px, py, pz in all_points_xyz:
                    dist_sq = (px - sx)**2 + (pz - sz)**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        closest_point = [px, py, pz]

                if closest_point:
                    all_clipped_points.append([sx, closest_point[1], sz])

        if len(all_clipped_points) < 2:
            return None

        points = np.array(all_clipped_points)
    else:
        # No clipping - use all points
        points = np.array(all_points_xyz)

    # Create track as a 3D-printable strip (wider and thicker than roads)
    track_mesh = create_road_strip(points, width=2.5, thickness=0.5)

    return {
        'type': 'gpx_track',
        'id': 'gpx_track',
        'name': 'GPX Track',
        'vertices': track_mesh['vertices'],
        'faces': track_mesh['faces']
    }


def interpolate_elevation(lat, lon, elevation_data):
    """Interpolate elevation at a specific lat/lon point using bilinear interpolation."""
    lats = np.array(elevation_data['lats'])
    lons = np.array(elevation_data['lons'])
    grid = np.array(elevation_data['grid'])

    # Clamp to grid bounds
    lat = max(lats[0], min(lats[-1], lat))
    lon = max(lons[0], min(lons[-1], lon))

    # Find surrounding grid indices for bilinear interpolation
    lat_idx = np.searchsorted(lats, lat) - 1
    lon_idx = np.searchsorted(lons, lon) - 1

    # Clamp indices
    lat_idx = max(0, min(len(lats) - 2, lat_idx))
    lon_idx = max(0, min(len(lons) - 2, lon_idx))

    # Get the four surrounding points
    lat0, lat1 = lats[lat_idx], lats[lat_idx + 1]
    lon0, lon1 = lons[lon_idx], lons[lon_idx + 1]

    # Get elevations at four corners
    z00 = grid[lat_idx, lon_idx]
    z01 = grid[lat_idx, lon_idx + 1]
    z10 = grid[lat_idx + 1, lon_idx]
    z11 = grid[lat_idx + 1, lon_idx + 1]

    # Bilinear interpolation
    t = (lat - lat0) / (lat1 - lat0) if lat1 != lat0 else 0
    s = (lon - lon0) / (lon1 - lon0) if lon1 != lon0 else 0

    z0 = z00 * (1 - s) + z01 * s
    z1 = z10 * (1 - s) + z11 * s

    return z0 * (1 - t) + z1 * t


def extrude_polygon(base_points, height):
    """Extrude a 2D polygon to create a 3D building using simple box extrusion."""
    n = len(base_points)
    if n < 3:
        return {'vertices': [], 'faces': []}

    # Use average Y as the base height
    base_y = np.mean(base_points[:, 1])

    # Flatten base points to same Y level for cleaner buildings
    flat_base = base_points.copy()
    flat_base[:, 1] = base_y

    # Create top points
    top_points = flat_base.copy()
    top_points[:, 1] = base_y + height

    # Combine vertices: base first, then top
    vertices = np.vstack([flat_base, top_points])

    faces = []

    # Only create side faces (skip top/bottom for simpler geometry)
    # Side faces - walls of the building
    for i in range(n):
        next_i = (i + 1) % n

        # Two triangles per wall segment
        # Bottom-left, bottom-right, top-left
        faces.append([i, next_i, n + i])
        # Bottom-right, top-right, top-left
        faces.append([next_i, n + next_i, n + i])

    # Simple top face using fan triangulation from centroid
    # Calculate centroid of top face
    centroid = np.mean(top_points, axis=0)
    centroid_idx = len(vertices)
    vertices = np.vstack([vertices, [centroid]])

    for i in range(n):
        next_i = (i + 1) % n
        # Triangle from centroid to edge
        faces.append([centroid_idx, n + i, n + next_i])

    return {
        'vertices': vertices.tolist(),
        'faces': np.array(faces).tolist()
    }


def create_flat_polygon(points):
    """Create a flat polygon mesh."""
    n = len(points)
    faces = []

    # Triangulate polygon
    if n > 2:
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])

    return {
        'vertices': points.tolist(),
        'faces': np.array(faces).tolist()
    }


def create_solid_polygon(points, thickness=0.5):
    """Create a 3D-printable solid polygon with thickness.

    Creates a watertight mesh with top surface, bottom surface, and side walls.
    Uses ear-clipping style triangulation for proper polygon fill.

    Args:
        points: Numpy array of [x, y, z] coordinates forming the polygon outline
        thickness: Height of the extrusion (in model units)

    Returns:
        dict: Mesh with vertices and faces
    """
    n = len(points)
    if n < 3:
        return {'vertices': [], 'faces': []}

    points = np.array(points)

    # Remove duplicate consecutive points
    unique_mask = np.ones(n, dtype=bool)
    for i in range(n):
        next_i = (i + 1) % n
        if np.allclose(points[i], points[next_i], atol=1e-6):
            unique_mask[next_i] = False
    points = points[unique_mask]
    n = len(points)

    if n < 3:
        return {'vertices': [], 'faces': []}

    vertices = []
    faces = []

    # Create top vertices (original points)
    for p in points:
        vertices.append([float(p[0]), float(p[1]), float(p[2])])

    # Create bottom vertices (offset down by thickness)
    for p in points:
        vertices.append([float(p[0]), float(p[1]) - thickness, float(p[2])])

    # Triangulate top/bottom faces using ear-clipping algorithm
    top_faces = triangulate_polygon(points[:, [0, 2]])  # Project to XZ plane

    print(f"Water polygon: {n} points, {len(top_faces)} triangles generated")

    if len(top_faces) == 0:
        # Fallback to simple fan triangulation
        print("Ear-clipping produced no triangles, using fan triangulation")
        for i in range(1, n - 1):
            top_faces.append([0, i, i + 1])

    # Top face
    for tri in top_faces:
        faces.append([int(tri[0]), int(tri[1]), int(tri[2])])

    # Bottom face (reversed winding)
    for tri in top_faces:
        faces.append([int(n + tri[0]), int(n + tri[2]), int(n + tri[1])])

    # Side walls - connect top and bottom edges
    for i in range(n):
        next_i = (i + 1) % n
        top_curr = i
        top_next = next_i
        bot_curr = n + i
        bot_next = n + next_i

        # Two triangles per side segment (consistent winding for outward normals)
        faces.append([top_curr, bot_curr, top_next])
        faces.append([top_next, bot_curr, bot_next])

    print(f"Water mesh: {len(vertices)} vertices, {len(faces)} faces")

    return {
        'vertices': vertices,
        'faces': faces
    }


def triangulate_polygon(points_2d):
    """Triangulate a 2D polygon using ear-clipping algorithm.

    Args:
        points_2d: Nx2 numpy array of 2D points forming polygon boundary

    Returns:
        List of triangle index triplets
    """
    n = len(points_2d)
    if n < 3:
        return []

    # Work with indices
    indices = list(range(n))
    triangles = []

    # Determine polygon winding (clockwise or counter-clockwise)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points_2d[i, 0] * points_2d[j, 1]
        area -= points_2d[j, 0] * points_2d[i, 1]
    ccw = area > 0

    # Ear clipping
    max_iterations = n * n  # Prevent infinite loops
    iteration = 0

    while len(indices) > 3 and iteration < max_iterations:
        iteration += 1
        ear_found = False

        for i in range(len(indices)):
            prev_i = (i - 1) % len(indices)
            next_i = (i + 1) % len(indices)

            prev_idx = indices[prev_i]
            curr_idx = indices[i]
            next_idx = indices[next_i]

            # Check if this is a valid ear
            if is_ear(points_2d, indices, prev_i, i, next_i, ccw):
                triangles.append([prev_idx, curr_idx, next_idx])
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            # No ear found, polygon might be degenerate
            # Fall back to fan triangulation from first vertex
            print(f"Ear clipping stuck at {len(indices)} vertices, using fan fallback")
            first = indices[0]
            for i in range(1, len(indices) - 1):
                triangles.append([first, indices[i], indices[i + 1]])
            break

    # Handle remaining triangle
    if len(indices) == 3:
        triangles.append([indices[0], indices[1], indices[2]])

    return triangles


def is_ear(points_2d, indices, prev_i, curr_i, next_i, ccw):
    """Check if vertex at curr_i forms a valid ear."""
    prev_idx = indices[prev_i]
    curr_idx = indices[curr_i]
    next_idx = indices[next_i]

    p0 = points_2d[prev_idx]
    p1 = points_2d[curr_idx]
    p2 = points_2d[next_idx]

    # Check if triangle is convex (correct winding)
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if ccw and cross <= 0:
        return False
    if not ccw and cross >= 0:
        return False

    # Check if any other vertex is inside this triangle
    for i, idx in enumerate(indices):
        if idx in (prev_idx, curr_idx, next_idx):
            continue
        if point_in_triangle(points_2d[idx], p0, p1, p2):
            return False

    return True


def point_in_triangle(p, a, b, c):
    """Check if point p is inside triangle abc using barycentric coordinates."""
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-10:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def create_road_strip(centerline, width=2.0, thickness=0.3):
    """Create a 3D-printable road strip with thickness along a centerline.

    Creates a box-like extrusion that's watertight for 3D printing.
    Vertex layout per point: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    """
    if len(centerline) < 2:
        return {'vertices': [], 'faces': []}

    # Remove duplicate consecutive points
    clean_centerline = [centerline[0]]
    for i in range(1, len(centerline)):
        if not np.allclose(centerline[i], centerline[i-1], atol=1e-6):
            clean_centerline.append(centerline[i])
    centerline = np.array(clean_centerline)

    if len(centerline) < 2:
        return {'vertices': [], 'faces': []}

    vertices = []
    faces = []

    # Generate vertices for top and bottom surfaces
    for i, point in enumerate(centerline):
        if i == 0:
            direction = centerline[i + 1] - centerline[i]
        elif i == len(centerline) - 1:
            direction = centerline[i] - centerline[i - 1]
        else:
            direction = centerline[i + 1] - centerline[i - 1]

        # Normalize in XZ plane (Y is up)
        length = np.linalg.norm([direction[0], direction[2]])
        if length < 1e-6:
            # Degenerate direction, use previous or default
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / length

        # Perpendicular in XZ plane (horizontal) - pointing left when facing direction
        perpendicular = np.array([-direction[2], 0, direction[0]])

        # Create 4 vertices per point: top-left, top-right, bottom-left, bottom-right
        half_width = width / 2
        top_left = point + perpendicular * half_width
        top_right = point - perpendicular * half_width
        bottom_left = np.array([top_left[0], top_left[1] - thickness, top_left[2]])
        bottom_right = np.array([top_right[0], top_right[1] - thickness, top_right[2]])

        vertices.extend([
            top_left.tolist(),
            top_right.tolist(),
            bottom_left.tolist(),
            bottom_right.tolist()
        ])

    n_points = len(centerline)

    # Create faces for each segment with consistent outward-facing winding
    # Indices: TL=0, TR=1, BL=2, BR=3 (per point)
    for i in range(n_points - 1):
        curr = i * 4      # Current point base index
        next_pt = (i + 1) * 4  # Next point base index

        # Current: TL=curr+0, TR=curr+1, BL=curr+2, BR=curr+3
        # Next:    TL=next+0, TR=next+1, BL=next+2, BR=next+3

        # Top face (normal pointing up +Y) - CCW when viewed from above
        faces.append([curr + 0, curr + 1, next_pt + 1])
        faces.append([curr + 0, next_pt + 1, next_pt + 0])

        # Bottom face (normal pointing down -Y) - CCW when viewed from below
        faces.append([curr + 2, next_pt + 2, next_pt + 3])
        faces.append([curr + 2, next_pt + 3, curr + 3])

        # Left side (normal pointing left) - CCW when viewed from left
        faces.append([curr + 0, next_pt + 0, next_pt + 2])
        faces.append([curr + 0, next_pt + 2, curr + 2])

        # Right side (normal pointing right) - CCW when viewed from right
        faces.append([curr + 1, curr + 3, next_pt + 3])
        faces.append([curr + 1, next_pt + 3, next_pt + 1])

    # Start cap (normal pointing backward along road) - CCW when viewed from start
    faces.append([0, 2, 3])
    faces.append([0, 3, 1])

    # End cap (normal pointing forward along road) - CCW when viewed from end
    end = (n_points - 1) * 4
    faces.append([end + 0, end + 1, end + 3])
    faces.append([end + 0, end + 3, end + 2])

    return {
        'vertices': vertices,
        'faces': faces
    }


def export_to_stl(mesh_data, filepath):
    """
    Export mesh to STL format for 3D printing.

    Args:
        mesh_data: Dict with terrain and feature meshes
        filepath: Output STL file path
    """
    try:
        # Collect all vertices and faces
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Add terrain mesh
        terrain = mesh_data.get('terrain', {})
        if terrain:
            vertices = np.array(terrain['vertices'], dtype=np.float64)
            faces = np.array(terrain['faces'], dtype=np.int32)

            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)

        # Add feature meshes
        for feature in mesh_data.get('features', []):
            vertices = np.array(feature['vertices'], dtype=np.float64)
            faces = np.array(feature['faces'], dtype=np.int32)

            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)

        # Add GPX track mesh if present
        gpx_track = mesh_data.get('gpx_track', {})
        if gpx_track and gpx_track.get('vertices') and gpx_track.get('faces'):
            vertices = np.array(gpx_track['vertices'], dtype=np.float64)
            faces = np.array(gpx_track['faces'], dtype=np.int32)

            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)

        if not all_vertices:
            raise ValueError("No mesh data to export")

        # Combine all geometry
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)

        # Create STL mesh
        stl_mesh = mesh.Mesh(np.zeros(combined_faces.shape[0], dtype=mesh.Mesh.dtype))

        for i, face in enumerate(combined_faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = combined_vertices[face[j], :]

        # Save to file
        stl_mesh.save(filepath)

        return {
            'success': True,
            'filepath': filepath,
            'vertices': len(combined_vertices),
            'faces': len(combined_faces)
        }

    except Exception as e:
        raise Exception(f"Error exporting to STL: {str(e)}")


def hex_to_rgb(hex_color):
    """
    Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#aabbcc" or "aabbcc")

    Returns:
        tuple: (r, g, b) values 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def export_to_3mf(mesh_data, filepath):
    """
    Export mesh to 3MF format with separate objects for each feature type.

    3MF supports multiple objects, colors, and proper units (mm).
    Buildings with custom colors are exported as separate objects with their custom colors.

    Args:
        mesh_data: Dict with terrain and feature meshes
        filepath: Output 3MF file path
    """
    # Define colors for each feature type (sRGB values 0-255)
    COLORS = {
        'terrain': (139, 115, 85),      # Brown/tan
        'building': (170, 170, 170),    # Gray
        'address_building': (255, 0, 0), # Red
        'road': (68, 68, 68),           # Dark gray
        'water': (74, 144, 226),        # Blue
        'gpx_track': (255, 0, 0),       # Red
        'railway': (100, 100, 100),     # Medium gray
    }

    objects = []  # List of (name, vertices, faces, color)

    # Add terrain
    terrain = mesh_data.get('terrain', {})
    if terrain and terrain.get('vertices'):
        objects.append(('Terrain', terrain['vertices'], terrain['faces'], COLORS['terrain']))

    # Add features grouped by type (except buildings with custom colors)
    features_by_type = {}
    for feature in mesh_data.get('features', []):
        ftype = feature.get('type', 'unknown')

        # Check if this building has a custom color
        if ftype == 'building' and feature.get('custom_color'):
            # Export building with custom color as separate object
            hex_color = feature['custom_color']
            rgb_color = hex_to_rgb(hex_color)
            building_name = feature.get('name', f"Building {feature.get('id', 'Unknown')}")
            objects.append((building_name, feature['vertices'], feature['faces'], rgb_color))
            continue

        # Check if it's the address building
        if feature.get('is_address_building'):
            ftype = 'address_building'

        if ftype not in features_by_type:
            features_by_type[ftype] = {'vertices': [], 'faces': [], 'vertex_offset': 0}

        verts = feature['vertices']
        fcs = np.array(feature['faces'], dtype=np.int32)

        # Offset faces for combined mesh using numpy array arithmetic
        offset = features_by_type[ftype]['vertex_offset']
        offset_faces = fcs + offset

        features_by_type[ftype]['vertices'].extend(verts)
        features_by_type[ftype]['faces'].extend(offset_faces.tolist())
        features_by_type[ftype]['vertex_offset'] += len(verts)

    for ftype, data in features_by_type.items():
        if data['vertices']:
            color = COLORS.get(ftype, (128, 128, 128))
            name = ftype.replace('_', ' ').title()
            objects.append((name, data['vertices'], data['faces'], color))

    # Add GPX track
    gpx_track = mesh_data.get('gpx_track', {})
    if gpx_track and gpx_track.get('vertices'):
        objects.append(('GPX Track', gpx_track['vertices'], gpx_track['faces'], COLORS['gpx_track']))

    if not objects:
        raise ValueError("No mesh data to export")

    # Build 3MF XML
    model_xml = build_3mf_model(objects)

    # Create 3MF file (ZIP archive)
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Content Types
        content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
        zf.writestr('[Content_Types].xml', content_types)

        # Relationships
        rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
        zf.writestr('_rels/.rels', rels)

        # 3D Model
        zf.writestr('3D/3dmodel.model', model_xml)

    return {
        'success': True,
        'filepath': filepath,
        'objects': len(objects)
    }


def build_3mf_model(objects):
    """Build the 3MF model XML with multiple objects."""
    # XML header
    xml_parts = ['''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <metadata name="Application">Topo3D</metadata>
  <resources>''']

    # Add base materials for colors
    xml_parts.append('    <m:basematerials id="1">')
    for i, (name, _, _, color) in enumerate(objects):
        r, g, b = color
        xml_parts.append(f'      <m:base name="{name}" displaycolor="#{r:02X}{g:02X}{b:02X}"/>')
    xml_parts.append('    </m:basematerials>')

    # Add each object as a separate mesh
    for obj_id, (name, vertices, faces, _) in enumerate(objects, start=2):
        xml_parts.append(f'    <object id="{obj_id}" name="{name}" pid="1" pindex="{obj_id - 2}" type="model">')
        xml_parts.append('      <mesh>')

        # Vertices
        xml_parts.append('        <vertices>')
        for v in vertices:
            xml_parts.append(f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
        xml_parts.append('        </vertices>')

        # Triangles
        xml_parts.append('        <triangles>')
        for f in faces:
            xml_parts.append(f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>')
        xml_parts.append('        </triangles>')

        xml_parts.append('      </mesh>')
        xml_parts.append('    </object>')

    xml_parts.append('  </resources>')

    # Build section - place all objects
    xml_parts.append('  <build>')
    for obj_id in range(2, len(objects) + 2):
        xml_parts.append(f'    <item objectid="{obj_id}"/>')
    xml_parts.append('  </build>')

    xml_parts.append('</model>')

    return '\n'.join(xml_parts)
