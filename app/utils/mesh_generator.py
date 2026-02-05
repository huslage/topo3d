"""3D mesh generation utilities."""

import numpy as np
import zipfile
import io
from stl import mesh
from scipy.spatial import Delaunay
from .shape_clipper import CircleClipper, SquareClipper, RectangleClipper, HexagonClipper
from .building_shapes import BuildingShapeGenerator

_ELEVATION_PREP_CACHE = {}
_ELEVATION_SAMPLE_CACHE_LIMIT = 50000


def _enforce_min_footprint(x1, x2, z1, z2, min_dim_mm):
    """Expand a rectangular footprint to at least `min_dim_mm` in X and Z."""
    cx = (x1 + x2) / 2.0
    cz = (z1 + z2) / 2.0
    width = abs(x2 - x1)
    depth = abs(z2 - z1)
    half_w = max(width / 2.0, min_dim_mm / 2.0)
    half_d = max(depth / 2.0, min_dim_mm / 2.0)
    return cx - half_w, cx + half_w, cz - half_d, cz + half_d


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

        # Size scale factor: all hardcoded dimensions are calibrated for 200mm model
        size_scale = model_width / 200.0

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
            # Use max dimension to ensure all content (esp. GPX tracks) fits within square
            half_width = max(lon_scaled, lat_scaled) / 2
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
            shape_clipper,
            size_scale
        )

        # Add features (roads, buildings, water)
        # Pass normalization params for consistent elevation mapping
        elev_params = {
            'min_elev': min_elev,
            'elev_range': elev_range,
            'avg_lat': avg_lat,
            'size_scale': size_scale
        }

        feature_meshes = []

        address_location = options.get('address_location')
        show_only_address_building = options.get('show_only_address_building', False)

        if show_only_address_building and address_location:
            # Create a single synthetic building at the address location
            building_mesh = create_address_building(
                address_location,
                elevation_data,
                bounds,
                scale_factor,
                vertical_scale,
                elev_params,
                options.get('building_height_scale', 1.0),
                shape_clipper
            )
            if building_mesh:
                feature_meshes.append(building_mesh)
        elif features.get('buildings'):
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
                False,
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


def generate_terrain_mesh(elevation_grid, lats, lons, bounds, scale_factor, vertical_scale, base_height, include_base, shape_clipper=None, size_scale=1.0):
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
            y = ((elevation_grid[i, j] - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale

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
        if shape_clipper and not np.all(inside_shape):
            # Some vertices outside shape - use shape-aware base generation
            base_vertices, base_faces, vertices = generate_shape_base(vertices, base_height, shape_clipper, rows, cols, inside_shape, faces)
        else:
            # No shape clipping or all vertices inside shape - use simple rectangular base
            base_vertices, base_faces = generate_base(vertices, base_height, rows, cols)
        if len(base_vertices) > 0 and len(base_faces) > 0:
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


def generate_shape_base(vertices, base_height, shape_clipper, rows, cols, inside_shape, terrain_faces=None):
    """
    Generate watertight base using terrain boundary edges.

    Finds actual boundary edges from the terrain faces and builds wall faces
    that share those edges exactly, ensuring a manifold mesh.

    Args:
        vertices: Terrain vertices array
        base_height: Height of base platform
        shape_clipper: ShapeClipper instance
        rows: Number of rows in elevation grid
        cols: Number of columns in elevation grid
        inside_shape: Boolean mask of which vertices are inside shape
        terrain_faces: Terrain face array (used to find boundary edges)

    Returns:
        tuple: (base_vertices, base_faces, modified_vertices) as numpy arrays
    """
    from collections import defaultdict

    vertices = vertices.copy()
    n = len(vertices)
    center_x = shape_clipper.center_x
    center_z = shape_clipper.center_z

    all_inside = np.all(inside_shape)

    if all_inside:
        # Shape is larger than grid - boundary IS the grid perimeter
        boundary_indices = []
        for j in range(cols):
            boundary_indices.append(j)
        for i in range(1, rows):
            boundary_indices.append(i * cols + (cols - 1))
        for j in range(cols - 2, -1, -1):
            boundary_indices.append((rows - 1) * cols + j)
        for i in range(rows - 2, 0, -1):
            boundary_indices.append(i * cols)

        if len(boundary_indices) < 3:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), vertices

        # Project boundary vertices to shape edge
        for idx in boundary_indices:
            v = vertices[idx]
            projected = shape_clipper.project_to_boundary(v[0], v[2])
            if projected:
                vertices[idx][0] = projected[0]
                vertices[idx][2] = projected[1]
    else:
        # Find boundary edges from actual terrain faces
        edge_count = defaultdict(int)
        if terrain_faces is not None and len(terrain_faces) > 0:
            for face in terrain_faces:
                v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
                for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
                    edge = (min(a, b), max(a, b))
                    edge_count[edge] += 1

        # Boundary edges have exactly 1 face
        boundary_edge_list = [e for e, c in edge_count.items() if c == 1]

        if not boundary_edge_list:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), vertices

        # Build adjacency from boundary edges to walk the loop
        adjacency = defaultdict(list)
        for v1, v2 in boundary_edge_list:
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)

        # Walk the boundary loop starting from any vertex
        start = boundary_edge_list[0][0]
        boundary_indices = [start]
        visited = {start}

        current = start
        while True:
            neighbors = adjacency[current]
            next_v = None
            for nb in neighbors:
                if nb not in visited:
                    next_v = nb
                    break
            if next_v is None:
                break
            boundary_indices.append(next_v)
            visited.add(next_v)
            current = next_v

        if len(boundary_indices) < 3:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), vertices

        # Project boundary vertices to shape edge
        for idx in boundary_indices:
            v = vertices[idx]
            projected = shape_clipper.project_to_boundary(v[0], v[2])
            if projected:
                vertices[idx][0] = projected[0]
                vertices[idx][2] = projected[1]

    num_boundary = len(boundary_indices)
    print(f"[INFO] generate_shape_base: {num_boundary} boundary vertices in ordered loop")

    # Map boundary vertex index -> position in boundary list
    boundary_pos = {idx: i for i, idx in enumerate(boundary_indices)}

    # Create base vertices (one below each boundary vertex)
    base_vertices = []
    for idx in boundary_indices:
        v = vertices[idx]
        base_vertices.append([v[0], -base_height, v[2]])

    # Create wall faces connecting terrain boundary to base vertices
    wall_faces = []
    for i in range(num_boundary):
        next_i = (i + 1) % num_boundary

        terrain_top = boundary_indices[i]
        terrain_top_next = boundary_indices[next_i]
        base_bottom = n + i
        base_bottom_next = n + next_i

        # Two triangles per wall segment (CCW when viewed from outside)
        wall_faces.append([terrain_top, base_bottom, terrain_top_next])
        wall_faces.append([terrain_top_next, base_bottom, base_bottom_next])

    base_faces = list(wall_faces)

    # Create center point for bottom face
    center_bottom_idx = n + num_boundary
    base_vertices.append([center_x, -base_height, center_z])

    # Create bottom face using fan triangulation from center
    for i in range(num_boundary):
        next_i = (i + 1) % num_boundary
        base_bottom = n + i
        base_bottom_next = n + next_i
        base_faces.append([center_bottom_idx, base_bottom_next, base_bottom])

    return np.array(base_vertices), np.array(base_faces, dtype=np.int32), vertices


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


def create_address_building(address_location, elevation_data, bounds, scale_factor, vertical_scale, elev_params, height_scale, shape_clipper=None):
    """Create a single building at the address location without needing OSM data."""
    addr_lat = address_location.get('lat')
    addr_lon = address_location.get('lon')
    if addr_lat is None or addr_lon is None:
        return None

    min_elev = elev_params['min_elev']
    elev_range = elev_params['elev_range']
    avg_lat = elev_params['avg_lat']
    size_scale = elev_params.get('size_scale', 1.0)

    # Create a building footprint around the address point (~20m x 20m)
    building_size_deg = 0.0001  # ~11m in each direction
    min_lat = addr_lat - building_size_deg
    max_lat = addr_lat + building_size_deg
    min_lon = addr_lon - building_size_deg / np.cos(np.radians(avg_lat))
    max_lon = addr_lon + building_size_deg / np.cos(np.radians(avg_lat))

    # Clamp to bounds
    min_lat = max(bounds['south'], min_lat)
    max_lat = min(bounds['north'], max_lat)
    min_lon = max(bounds['west'], min_lon)
    max_lon = min(bounds['east'], max_lon)

    # Convert to model space
    x1 = (min_lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
    x2 = (max_lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
    z1 = (bounds['north'] - max_lat) * scale_factor
    z2 = (bounds['north'] - min_lat) * scale_factor

    # Clip to shape boundary if applicable
    if shape_clipper and not shape_clipper.is_inside(x1, z1):
        return None

    # Get base elevation
    raw_elev = interpolate_elevation(addr_lat, addr_lon, elevation_data)
    base_y = ((raw_elev - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale

    # Ensure address marker remains printable even on tiny (e.g., 50mm) models.
    min_address_footprint_mm = 2.0
    min_address_height_mm = 2.0
    x1, x2, z1, z2 = _enforce_min_footprint(x1, x2, z1, z2, min_address_footprint_mm)

    height = 8.0 * height_scale * 0.15 * size_scale
    height = max(height, min_address_height_mm)

    building_mesh = create_box(x1, x2, base_y, base_y + height, z1, z2)

    return {
        'type': 'building',
        'id': 'address_building',
        'name': 'Address Building',
        'building_type': 'yes',
        'vertices': building_mesh['vertices'],
        'faces': building_mesh['faces'],
        'is_address_building': True
    }


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
    size_scale = elev_params.get('size_scale', 1.0)

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

    # When showing only the address building, don't apply the 150-building limit
    # since the address building may be beyond that index
    building_list = buildings if show_only_address_building else buildings[:150]
    for building in building_list:
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
        height = building.get('height', 8.0) * height_scale * 0.15 * size_scale
        if is_address_building:
            height = max(height, 2.0)  # Ensure minimum height for visibility/printability
        else:
            height = max(height, 1.2)  # Prevent sub-mm buildings at tiny model widths

        # Convert bounding box corners to model space
        x1 = (min_lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
        x2 = (max_lon - bounds['west']) * scale_factor * np.cos(np.radians(avg_lat))
        z1 = (bounds['north'] - max_lat) * scale_factor
        z2 = (bounds['north'] - min_lat) * scale_factor

        # Enforce minimum printable footprint for very small structures.
        min_building_footprint_mm = 1.2
        x1, x2, z1, z2 = _enforce_min_footprint(x1, x2, z1, z2, min_building_footprint_mm)

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
        base_y = ((raw_elev - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale

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
    size_scale = elev_params.get('size_scale', 1.0)

    road_height = road_height * size_scale

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
            y = ((raw_elev - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale + road_height

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
                road_mesh = create_road_strip(segment_points_3d, width=1.0 * size_scale)

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
            road_mesh = create_road_strip(points_xyz, width=1.0 * size_scale)

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
    size_scale = elev_params.get('size_scale', 1.0)

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
        # For water, use AVERAGE elevation of perimeter points to represent water surface
        # Water bodies in real life have a flat surface, not following terrain beneath
        perimeter_elevations = []
        for coord in coords:
            lat = max(bounds['south'], min(bounds['north'], coord['lat']))
            lon = max(bounds['west'], min(bounds['east'], coord['lon']))
            raw_elev = interpolate_elevation(lat, lon, elevation_data)
            perimeter_elevations.append(raw_elev)

        # Use average of perimeter elevations as water surface level
        avg_water_elev = np.mean(perimeter_elevations) if perimeter_elevations else min_elev

        # Normalize the water level
        water_y = ((avg_water_elev - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale
        # Add offset to lift water above terrain surface for visibility
        water_y = max(0.5 * size_scale, water_y + 0.5 * size_scale)

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

        # Clip water polygon to shape boundary
        if shape_clipper:
            # Filter to only vertices inside the shape
            clipped_xz = [(x, z) for x, z in points_xz if shape_clipper.is_inside(x, z)]
            if len(clipped_xz) < 3:
                continue
            points_xz = clipped_xz

        points = np.array([[x, water_y, z] for x, z in points_xz])

        # Skip if not enough points
        if len(points) < 3:
            continue

        # Create solid water body with thickness for 3D printing
        water_mesh = create_solid_polygon(points, thickness=0.5 * size_scale)

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
    size_scale = elev_params.get('size_scale', 1.0)

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

            # Elevation - sit slightly above terrain to be visible (roads are at +0.2, GPX at +0.3)
            y = ((raw_elev - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale + 0.3 * size_scale

            all_points_xz.append((x, z))
            all_points_xyz.append([x, y, z])

    if len(all_points_xyz) < 2:
        return None

    # Don't clip GPX track to shape boundary - preserve natural track path
    # The GPX track should show the actual route, not be cut off at shape boundaries
    points = np.array(all_points_xyz)

    # Create track as a 3D-printable strip (wider and thicker than roads)
    track_mesh = create_road_strip(points, width=2.5 * size_scale, thickness=0.5 * size_scale)

    return {
        'type': 'gpx_track',
        'id': 'gpx_track',
        'name': 'GPX Track',
        'vertices': track_mesh['vertices'],
        'faces': track_mesh['faces']
    }


def interpolate_elevation(lat, lon, elevation_data):
    """Interpolate elevation at a specific lat/lon point using bilinear interpolation."""
    cache_key = (
        id(elevation_data.get('lats')),
        id(elevation_data.get('lons')),
        id(elevation_data.get('grid'))
    )
    prepared = _ELEVATION_PREP_CACHE.get(cache_key)
    if prepared is None:
        lats = np.asarray(elevation_data['lats'], dtype=np.float64)
        lons = np.asarray(elevation_data['lons'], dtype=np.float64)
        grid = np.asarray(elevation_data['grid'], dtype=np.float64)
        prepared = {
            'lats': lats,
            'lons': lons,
            'grid': grid,
            'samples': {}
        }
        _ELEVATION_PREP_CACHE[cache_key] = prepared

    lats = prepared['lats']
    lons = prepared['lons']
    grid = prepared['grid']
    sample_cache = prepared['samples']

    sample_key = (round(float(lat), 7), round(float(lon), 7))
    cached = sample_cache.get(sample_key)
    if cached is not None:
        return cached

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

    interpolated = z0 * (1 - t) + z1 * t
    if len(sample_cache) >= _ELEVATION_SAMPLE_CACHE_LIMIT:
        sample_cache.clear()
    sample_cache[sample_key] = interpolated
    return interpolated


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

    # Merge non-consecutive near-duplicate vertices (common in OSM water polygons
    # where the boundary visits the same point twice, e.g. at pinch points)
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, [0, 2]])  # Match in XZ plane
    pairs = tree.query_pairs(r=1e-4)  # Find near-duplicate pairs
    if pairs:
        # Build mapping: for each duplicate, map to the lowest index
        remap = list(range(n))
        for i, j in pairs:
            lo, hi = min(i, j), max(i, j)
            remap[hi] = lo
        # Resolve chains (if a→b→c, make a→c and b→c)
        for i in range(n):
            while remap[i] != remap[remap[i]]:
                remap[i] = remap[remap[i]]
        # Rebuild points list removing duplicates but preserving order
        keep = [i for i in range(n) if remap[i] == i]
        old_to_new = {}
        for new_idx, old_idx in enumerate(keep):
            old_to_new[old_idx] = new_idx
        # Map all remapped indices
        index_map = [old_to_new[remap[i]] for i in range(n)]
        # Rebuild polygon: walk original order but use remapped indices, skip consecutive dupes
        new_indices = []
        for i in range(n):
            mapped = index_map[i]
            if len(new_indices) == 0 or mapped != new_indices[-1]:
                new_indices.append(mapped)
        # Remove wrap-around duplicate
        if len(new_indices) > 1 and new_indices[0] == new_indices[-1]:
            new_indices.pop()
        points = points[keep]
        n = len(points)
        print(f"Water polygon: merged {len(pairs)} duplicate vertex pairs, {n} unique vertices")

    if n < 3:
        return {'vertices': [], 'faces': []}

    # Remove collinear vertices (common in OSM data where straight edges have many nodes).
    # Collinear vertices cause ear-clipping to fail because the cross product is zero.
    non_collinear = []
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        ax, az = points[prev_i, 0], points[prev_i, 2]
        bx, bz = points[i, 0], points[i, 2]
        cx, cz = points[next_i, 0], points[next_i, 2]
        cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax)
        if abs(cross) > 1e-10:
            non_collinear.append(i)
    if len(non_collinear) < n and len(non_collinear) >= 3:
        removed = n - len(non_collinear)
        points = points[non_collinear]
        n = len(points)
        print(f"Water polygon: removed {removed} collinear vertices, {n} remaining")

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

    # Diagnostic: count non-manifold edges in this water mesh
    from collections import defaultdict
    edge_count = defaultdict(int)
    for f in faces:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            edge = (min(a, b), max(a, b))
            edge_count[edge] += 1
    non_manifold = [(e, c) for e, c in edge_count.items() if c != 2]
    if non_manifold:
        count_1 = sum(1 for _, c in non_manifold if c == 1)
        count_3plus = sum(1 for _, c in non_manifold if c >= 3)
        print(f"[DIAG] Water non-manifold edges: {len(non_manifold)} total ({count_1} with 1 face, {count_3plus} with 3+ faces)")
        # Show a few examples
        for e, c in non_manifold[:5]:
            v0, v1 = e
            is_top = v0 < n and v1 < n
            is_bot = v0 >= n and v1 >= n
            is_wall = (v0 < n) != (v1 < n)
            loc = "top" if is_top else ("bottom" if is_bot else "wall-diagonal")
            print(f"[DIAG]   Edge ({v0},{v1}) count={c} location={loc}")

    return {
        'vertices': vertices,
        'faces': faces
    }


def triangulate_polygon(points_2d):
    """Triangulate a 2D polygon using ear-clipping.

    Ear-clipping preserves all polygon boundary edges, which is critical
    for manifold mesh generation (walls reference boundary edges).

    Args:
        points_2d: Nx2 numpy array of 2D points forming polygon boundary

    Returns:
        List of triangle index triplets (indices into original points_2d)
    """
    n = len(points_2d)
    if n < 3:
        return []
    if n == 3:
        return [[0, 1, 2]]

    points_2d = np.array(points_2d, dtype=np.float64)

    # Determine polygon winding (need CCW for ear-clipping)
    # Shoelace formula for signed area
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += points_2d[i, 0] * points_2d[j, 1]
        signed_area -= points_2d[j, 0] * points_2d[i, 1]
    ccw = signed_area > 0

    # Build index list (we remove vertices as we clip ears)
    indices = list(range(n))
    triangles = []

    def cross_2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def is_ear(idx_list, pos, pts, is_ccw):
        """Check if vertex at position pos in idx_list is an ear."""
        m = len(idx_list)
        prev_pos = (pos - 1) % m
        next_pos = (pos + 1) % m

        a = pts[idx_list[prev_pos]]
        b = pts[idx_list[pos]]
        c = pts[idx_list[next_pos]]

        # Check if this vertex is convex (cross product sign matches winding)
        cross = cross_2d(a, b, c)
        if is_ccw and cross <= 1e-12:
            return False
        if not is_ccw and cross >= -1e-12:
            return False

        # Check that no other polygon vertex is inside this triangle
        for j in range(m):
            if j == prev_pos or j == pos or j == next_pos:
                continue
            p = pts[idx_list[j]]
            # Point-in-triangle test using barycentric coordinates
            d1 = cross_2d(a, b, p)
            d2 = cross_2d(b, c, p)
            d3 = cross_2d(c, a, p)
            has_neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
            has_pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
            if not (has_neg and has_pos):
                # Point is inside or on edge of triangle
                return False
        return True

    max_iterations = n * n  # Safety limit
    iteration = 0
    while len(indices) > 3 and iteration < max_iterations:
        ear_found = False
        m = len(indices)
        for i in range(m):
            if is_ear(indices, i, points_2d, ccw):
                prev_pos = (i - 1) % m
                next_pos = (i + 1) % m
                triangles.append([indices[prev_pos], indices[i], indices[next_pos]])
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            # No ear found - polygon may be degenerate
            # Use remaining vertices as fan triangulation fallback
            for i in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[i], indices[i + 1]])
            break
        iteration += 1

    # Add last triangle
    if len(indices) == 3:
        triangles.append([indices[0], indices[1], indices[2]])

    return triangles


def _point_in_polygon(point, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
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
