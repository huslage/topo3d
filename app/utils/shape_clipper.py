"""Shape clipping utilities for different model boundary types."""

import numpy as np
from abc import ABC, abstractmethod


class ShapeClipper(ABC):
    """
    Abstract base class for shape clipping operations.

    Different shape types (circle, square, rectangle, hexagon) can clip
    features (roads, buildings, water) to their boundaries while preserving
    path continuity and generating appropriate walls.
    """

    def __init__(self, center_x, center_z, size):
        """
        Initialize shape clipper.

        Args:
            center_x: Center X coordinate in model space
            center_z: Center Z coordinate in model space
            size: Characteristic size (radius, half-width, etc.)
        """
        self.center_x = center_x
        self.center_z = center_z
        self.size = size

    @abstractmethod
    def is_inside(self, x, z):
        """
        Test if a point is inside the shape boundary.

        Args:
            x: X coordinate(s) - can be scalar or numpy array
            z: Z coordinate(s) - can be scalar or numpy array

        Returns:
            bool or numpy array of bools: True if inside
        """
        pass

    @abstractmethod
    def clip_linestring(self, points):
        """
        Clip a linestring (path) to shape boundary, preserving continuity.

        Uses line-shape intersection to keep paths continuous instead of
        just removing out-of-bounds points.

        Args:
            points: List of (x, z) tuples or numpy array of shape (N, 2)

        Returns:
            list: List of continuous path segments, each a numpy array of shape (M, 2)
        """
        pass

    @abstractmethod
    def clip_polygon(self, points):
        """
        Clip a polygon to shape boundary.

        Args:
            points: List of (x, z) tuples or numpy array of shape (N, 2)

        Returns:
            numpy array or None: Clipped polygon vertices, or None if fully outside
        """
        pass

    @abstractmethod
    def generate_wall_vertices(self, terrain_elevation_func, base_height, num_segments=None):
        """
        Generate vertices for the shape boundary wall.

        Args:
            terrain_elevation_func: Function that takes (x, z) and returns elevation (y)
            base_height: Height of the base platform
            num_segments: Number of segments for the wall (None = auto)

        Returns:
            tuple: (wall_vertices, wall_faces) as numpy arrays
        """
        pass

    @abstractmethod
    def project_to_boundary(self, x, z):
        """
        Project a point to the nearest point on the shape boundary.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            tuple: (projected_x, projected_z) on the boundary, or None if cannot project
        """
        pass


class CircleClipper(ShapeClipper):
    """Circular boundary clipper with line-circle intersection."""

    def __init__(self, center_x, center_z, radius):
        """
        Initialize circular clipper.

        Args:
            center_x: Center X coordinate
            center_z: Center Z coordinate
            radius: Circle radius
        """
        super().__init__(center_x, center_z, radius)
        self.radius = radius

    def is_inside(self, x, z):
        """Test if point(s) are inside the circle."""
        dx = x - self.center_x
        dz = z - self.center_z
        distance_sq = dx * dx + dz * dz
        return distance_sq <= (self.radius * self.radius)

    def _line_circle_intersection(self, p1, p2):
        """
        Find intersection points of line segment with circle.

        Args:
            p1: Start point (x, z)
            p2: End point (x, z)

        Returns:
            list: Intersection points (0, 1, or 2 points)
        """
        x1, z1 = p1
        x2, z2 = p2

        # Translate to circle-centered coordinates
        dx = x2 - x1
        dz = z2 - z1
        fx = x1 - self.center_x
        fz = z1 - self.center_z

        # Quadratic equation coefficients: a*t^2 + b*t + c = 0
        a = dx*dx + dz*dz
        b = 2*(fx*dx + fz*dz)
        c = fx*fx + fz*fz - self.radius*self.radius

        if a < 1e-10:  # Line segment is too short
            return []

        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return []  # No intersection

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        intersections = []
        for t in [t1, t2]:
            if 0 <= t <= 1:  # Intersection within segment
                ix = x1 + t * dx
                iz = z1 + t * dz
                intersections.append((ix, iz))

        return intersections

    def clip_linestring(self, points):
        """
        Clip linestring to circle boundary, preserving continuity.

        Returns list of continuous path segments.
        """
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))

            if i < len(points) - 1:
                x_next, z_next = points[i + 1]
                inside_next = self.is_inside(x_next, z_next)

                # Check for boundary crossing
                if inside != inside_next:
                    intersections = self._line_circle_intersection(
                        (x, z), (x_next, z_next)
                    )

                    if intersections:
                        # Use the first intersection point
                        ix, iz = intersections[0]

                        if inside:
                            # Exiting circle
                            current_segment.append((ix, iz))
                            if len(current_segment) >= 2:
                                segments.append(np.array(current_segment))
                            current_segment = []
                        else:
                            # Entering circle
                            current_segment = [(ix, iz)]

        # Add final segment if it has points
        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points):
        """
        Clip polygon to circle boundary using Sutherland-Hodgman-style algorithm.

        For simplicity, this returns the polygon if any vertex is inside,
        None if all vertices are outside. More sophisticated polygon clipping
        would require a full implementation of circle-polygon intersection.
        """
        if len(points) == 0:
            return None

        points = np.array(points)
        x_coords = points[:, 0]
        z_coords = points[:, 1]

        inside_mask = self.is_inside(x_coords, z_coords)

        if np.any(inside_mask):
            # At least one vertex inside - keep the polygon
            # (simplified approach; full clipping would intersect edges)
            return points
        else:
            return None

    def generate_wall_vertices(self, terrain_elevation_func, base_height, num_segments=360):
        """
        Generate circular wall following terrain contour.

        Args:
            terrain_elevation_func: Function (x, z) -> elevation
            base_height: Base platform height
            num_segments: Number of segments in circular wall (default 360)

        Returns:
            tuple: (wall_vertices, wall_faces)
        """
        angles = np.linspace(0, 2*np.pi, num_segments, endpoint=False)

        wall_vertices = []
        wall_faces = []

        for i, angle in enumerate(angles):
            x = self.center_x + self.radius * np.cos(angle)
            z = self.center_z + self.radius * np.sin(angle)

            # Get elevation at this boundary point
            try:
                elevation = terrain_elevation_func(x, z)
            except:
                elevation = base_height

            # Top vertex (at terrain)
            wall_vertices.append([x, elevation, z])
            # Bottom vertex (at base)
            wall_vertices.append([x, -base_height, z])

        wall_vertices = np.array(wall_vertices)

        # Generate wall faces (quads as two triangles)
        for i in range(num_segments):
            next_i = (i + 1) % num_segments

            top_i = i * 2
            bottom_i = i * 2 + 1
            top_next = next_i * 2
            bottom_next = next_i * 2 + 1

            # Triangle 1: bottom_i, top_i, top_next
            wall_faces.append([bottom_i, top_i, top_next])
            # Triangle 2: bottom_i, top_next, bottom_next
            wall_faces.append([bottom_i, top_next, bottom_next])

        return wall_vertices, np.array(wall_faces, dtype=np.int32)

    def project_to_boundary(self, x, z):
        """Project point to circle boundary."""
        dx = x - self.center_x
        dz = z - self.center_z
        dist = np.sqrt(dx * dx + dz * dz)

        if dist == 0:
            # Point at center - project to arbitrary point on circle
            return (self.center_x + self.radius, self.center_z)

        # Scale to radius
        scale = self.radius / dist
        return (self.center_x + dx * scale, self.center_z + dz * scale)


class SquareClipper(ShapeClipper):
    """Square boundary clipper."""

    def __init__(self, center_x, center_z, half_width):
        """
        Initialize square clipper.

        Args:
            center_x: Center X coordinate
            center_z: Center Z coordinate
            half_width: Half of square side length
        """
        super().__init__(center_x, center_z, half_width)
        self.half_width = half_width

    def is_inside(self, x, z):
        """Test if point(s) are inside the square."""
        return (np.abs(x - self.center_x) <= self.half_width) & \
               (np.abs(z - self.center_z) <= self.half_width)

    def _line_box_intersection(self, p1, p2):
        """
        Find intersection of line segment with square boundary.

        Returns list of intersection points.
        """
        x1, z1 = p1
        x2, z2 = p2

        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_width
        max_z = self.center_z + self.half_width

        intersections = []

        # Check intersection with four edges
        # Left edge (x = min_x)
        if x1 != x2:
            t = (min_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((min_x, z))

        # Right edge (x = max_x)
        if x1 != x2:
            t = (max_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((max_x, z))

        # Bottom edge (z = min_z)
        if z1 != z2:
            t = (min_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, min_z))

        # Top edge (z = max_z)
        if z1 != z2:
            t = (max_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, max_z))

        # Remove duplicate intersections (corner cases)
        unique_intersections = []
        for pt in intersections:
            is_duplicate = False
            for existing_pt in unique_intersections:
                if np.allclose(pt, existing_pt, atol=1e-6):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(pt)

        return unique_intersections

    def clip_linestring(self, points):
        """Clip linestring to square boundary."""
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))

            if i < len(points) - 1:
                x_next, z_next = points[i + 1]
                inside_next = self.is_inside(x_next, z_next)

                if inside != inside_next:
                    intersections = self._line_box_intersection(
                        (x, z), (x_next, z_next)
                    )

                    if intersections:
                        ix, iz = intersections[0]

                        if inside:
                            current_segment.append((ix, iz))
                            if len(current_segment) >= 2:
                                segments.append(np.array(current_segment))
                            current_segment = []
                        else:
                            current_segment = [(ix, iz)]

        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points):
        """Clip polygon to square boundary (simplified)."""
        if len(points) == 0:
            return None

        points = np.array(points)
        x_coords = points[:, 0]
        z_coords = points[:, 1]

        inside_mask = self.is_inside(x_coords, z_coords)

        if np.any(inside_mask):
            return points
        else:
            return None

    def generate_wall_vertices(self, terrain_elevation_func, base_height, num_segments=None):
        """Generate square wall following terrain contour."""
        # For square, use 4 corners and subdivide each edge
        segments_per_edge = num_segments // 4 if num_segments else 50

        wall_vertices = []
        wall_faces = []

        # Define four edges
        edges = [
            # Bottom edge (min_z)
            (np.linspace(self.center_x - self.half_width, self.center_x + self.half_width, segments_per_edge),
             np.full(segments_per_edge, self.center_z - self.half_width)),
            # Right edge (max_x)
            (np.full(segments_per_edge, self.center_x + self.half_width),
             np.linspace(self.center_z - self.half_width, self.center_z + self.half_width, segments_per_edge)),
            # Top edge (max_z)
            (np.linspace(self.center_x + self.half_width, self.center_x - self.half_width, segments_per_edge),
             np.full(segments_per_edge, self.center_z + self.half_width)),
            # Left edge (min_x)
            (np.full(segments_per_edge, self.center_x - self.half_width),
             np.linspace(self.center_z + self.half_width, self.center_z - self.half_width, segments_per_edge))
        ]

        for x_coords, z_coords in edges:
            for x, z in zip(x_coords, z_coords):
                try:
                    elevation = terrain_elevation_func(x, z)
                except:
                    elevation = base_height

                wall_vertices.append([x, elevation, z])
                wall_vertices.append([x, -base_height, z])

        wall_vertices = np.array(wall_vertices)

        # Generate wall faces
        total_vertices = len(wall_vertices) // 2
        for i in range(total_vertices):
            next_i = (i + 1) % total_vertices

            top_i = i * 2
            bottom_i = i * 2 + 1
            top_next = next_i * 2
            bottom_next = next_i * 2 + 1

            wall_faces.append([bottom_i, top_i, top_next])
            wall_faces.append([bottom_i, top_next, bottom_next])

        return wall_vertices, np.array(wall_faces, dtype=np.int32)

    def project_to_boundary(self, x, z):
        """Project point to square boundary."""
        dx = x - self.center_x
        dz = z - self.center_z

        # Clamp to square bounds
        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_width
        max_z = self.center_z + self.half_width

        # If already on or outside boundary, find nearest edge point
        if abs(dx) >= abs(dz):
            # Closer to left/right edge
            if dx >= 0:
                # Right edge
                return (max_x, z)
            else:
                # Left edge
                return (min_x, z)
        else:
            # Closer to top/bottom edge
            if dz >= 0:
                # Bottom edge
                return (x, max_z)
            else:
                # Top edge
                return (x, min_z)


class RectangleClipper(ShapeClipper):
    """Rectangle boundary clipper with automatic aspect ratio."""

    def __init__(self, center_x, center_z, half_width, half_height):
        """
        Initialize rectangle clipper.

        Args:
            center_x: Center X coordinate
            center_z: Center Z coordinate
            half_width: Half of rectangle width (X direction)
            half_height: Half of rectangle height (Z direction)
        """
        super().__init__(center_x, center_z, max(half_width, half_height))
        self.half_width = half_width
        self.half_height = half_height

    def is_inside(self, x, z):
        """Test if point(s) are inside the rectangle."""
        return (np.abs(x - self.center_x) <= self.half_width) & \
               (np.abs(z - self.center_z) <= self.half_height)

    def _line_box_intersection(self, p1, p2):
        """Find intersection of line segment with rectangle boundary."""
        x1, z1 = p1
        x2, z2 = p2

        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_height
        max_z = self.center_z + self.half_height

        intersections = []

        # Check intersection with four edges (same logic as SquareClipper)
        if x1 != x2:
            t = (min_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((min_x, z))

        if x1 != x2:
            t = (max_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((max_x, z))

        if z1 != z2:
            t = (min_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, min_z))

        if z1 != z2:
            t = (max_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, max_z))

        unique_intersections = []
        for pt in intersections:
            is_duplicate = False
            for existing_pt in unique_intersections:
                if np.allclose(pt, existing_pt, atol=1e-6):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(pt)

        return unique_intersections

    def clip_linestring(self, points):
        """Clip linestring to rectangle boundary."""
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))

            if i < len(points) - 1:
                x_next, z_next = points[i + 1]
                inside_next = self.is_inside(x_next, z_next)

                if inside != inside_next:
                    intersections = self._line_box_intersection(
                        (x, z), (x_next, z_next)
                    )

                    if intersections:
                        ix, iz = intersections[0]

                        if inside:
                            current_segment.append((ix, iz))
                            if len(current_segment) >= 2:
                                segments.append(np.array(current_segment))
                            current_segment = []
                        else:
                            current_segment = [(ix, iz)]

        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points):
        """Clip polygon to rectangle boundary (simplified)."""
        if len(points) == 0:
            return None

        points = np.array(points)
        x_coords = points[:, 0]
        z_coords = points[:, 1]

        inside_mask = self.is_inside(x_coords, z_coords)

        if np.any(inside_mask):
            return points
        else:
            return None

    def generate_wall_vertices(self, terrain_elevation_func, base_height, num_segments=None):
        """Generate rectangular wall following terrain contour."""
        # Distribute segments proportionally to edge length
        perimeter = 2 * (self.half_width + self.half_height)
        total_segments = num_segments if num_segments else 200

        segments_width = int(total_segments * self.half_width / perimeter)
        segments_height = int(total_segments * self.half_height / perimeter)

        segments_width = max(segments_width, 10)
        segments_height = max(segments_height, 10)

        wall_vertices = []
        wall_faces = []

        edges = [
            (np.linspace(self.center_x - self.half_width, self.center_x + self.half_width, segments_width),
             np.full(segments_width, self.center_z - self.half_height)),
            (np.full(segments_height, self.center_x + self.half_width),
             np.linspace(self.center_z - self.half_height, self.center_z + self.half_height, segments_height)),
            (np.linspace(self.center_x + self.half_width, self.center_x - self.half_width, segments_width),
             np.full(segments_width, self.center_z + self.half_height)),
            (np.full(segments_height, self.center_x - self.half_width),
             np.linspace(self.center_z + self.half_height, self.center_z - self.half_height, segments_height))
        ]

        for x_coords, z_coords in edges:
            for x, z in zip(x_coords, z_coords):
                try:
                    elevation = terrain_elevation_func(x, z)
                except:
                    elevation = base_height

                wall_vertices.append([x, elevation, z])
                wall_vertices.append([x, -base_height, z])

        wall_vertices = np.array(wall_vertices)

        total_vertices = len(wall_vertices) // 2
        for i in range(total_vertices):
            next_i = (i + 1) % total_vertices

            top_i = i * 2
            bottom_i = i * 2 + 1
            top_next = next_i * 2
            bottom_next = next_i * 2 + 1

            wall_faces.append([bottom_i, top_i, top_next])
            wall_faces.append([bottom_i, top_next, bottom_next])

        return wall_vertices, np.array(wall_faces, dtype=np.int32)

    def project_to_boundary(self, x, z):
        """Project point to rectangle boundary."""
        dx = x - self.center_x
        dz = z - self.center_z

        # Rectangle bounds
        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_height
        max_z = self.center_z + self.half_height

        # Determine which edge is closest
        # Compare ratio of distance to boundary vs half-dimension
        ratio_x = abs(dx) / self.half_width if self.half_width > 0 else 0
        ratio_z = abs(dz) / self.half_height if self.half_height > 0 else 0

        if ratio_x >= ratio_z:
            # Closer to left/right edge
            if dx >= 0:
                return (max_x, z)
            else:
                return (min_x, z)
        else:
            # Closer to top/bottom edge
            if dz >= 0:
                return (x, max_z)
            else:
                return (x, min_z)


class HexagonClipper(ShapeClipper):
    """Hexagon boundary clipper (flat-top orientation)."""

    def __init__(self, center_x, center_z, radius):
        """
        Initialize hexagon clipper.

        Args:
            center_x: Center X coordinate
            center_z: Center Z coordinate
            radius: Distance from center to vertex (circumradius)
        """
        super().__init__(center_x, center_z, radius)
        self.radius = radius

        # Compute hexagon vertices (flat-top orientation)
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        self.vertices = np.array([
            [center_x + radius * np.cos(a), center_z + radius * np.sin(a)]
            for a in angles
        ])

    def is_inside(self, x, z):
        """Test if point(s) are inside the hexagon using point-in-polygon test."""
        # Ensure inputs are arrays
        x = np.atleast_1d(x)
        z = np.atleast_1d(z)
        scalar_input = x.shape == (1,)

        result = np.zeros(x.shape, dtype=bool)

        # Ray casting algorithm for each point
        for i in range(len(x)):
            px, pz = x[i], z[i]
            inside = False

            for j in range(6):
                k = (j + 1) % 6
                vx1, vz1 = self.vertices[j]
                vx2, vz2 = self.vertices[k]

                if ((vz1 > pz) != (vz2 > pz)) and \
                   (px < (vx2 - vx1) * (pz - vz1) / (vz2 - vz1) + vx1):
                    inside = not inside

            result[i] = inside

        return result[0] if scalar_input else result

    def clip_linestring(self, points):
        """
        Clip linestring to hexagon boundary using Sutherland-Hodgman algorithm.
        """
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        # Simple approach: check if points are inside/outside
        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))
            elif len(current_segment) > 0:
                # Just exited hexagon
                if len(current_segment) >= 2:
                    segments.append(np.array(current_segment))
                current_segment = []

        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points):
        """Clip polygon to hexagon boundary (simplified)."""
        if len(points) == 0:
            return None

        points = np.array(points)
        x_coords = points[:, 0]
        z_coords = points[:, 1]

        inside_mask = self.is_inside(x_coords, z_coords)

        if np.any(inside_mask):
            return points
        else:
            return None

    def generate_wall_vertices(self, terrain_elevation_func, base_height, num_segments=None):
        """Generate hexagonal wall following terrain contour."""
        segments_per_edge = (num_segments // 6) if num_segments else 30

        wall_vertices = []
        wall_faces = []

        # Generate vertices along each of the 6 edges
        for i in range(6):
            j = (i + 1) % 6
            start = self.vertices[i]
            end = self.vertices[j]

            # Interpolate along edge
            for t in np.linspace(0, 1, segments_per_edge, endpoint=(i == 5)):
                x = start[0] + t * (end[0] - start[0])
                z = start[1] + t * (end[1] - start[1])

                try:
                    elevation = terrain_elevation_func(x, z)
                except:
                    elevation = base_height

                wall_vertices.append([x, elevation, z])
                wall_vertices.append([x, -base_height, z])

        wall_vertices = np.array(wall_vertices)

        total_vertices = len(wall_vertices) // 2
        for i in range(total_vertices):
            next_i = (i + 1) % total_vertices

            top_i = i * 2
            bottom_i = i * 2 + 1
            top_next = next_i * 2
            bottom_next = next_i * 2 + 1

            wall_faces.append([bottom_i, top_i, top_next])
            wall_faces.append([bottom_i, top_next, bottom_next])

        return wall_vertices, np.array(wall_faces, dtype=np.int32)

    def project_to_boundary(self, x, z):
        """Project point to hexagon boundary."""
        # Find nearest point on each of the 6 edges
        min_dist_sq = float('inf')
        nearest_point = None

        for i in range(6):
            j = (i + 1) % 6
            v1 = self.vertices[i]
            v2 = self.vertices[j]

            # Project point onto line segment
            edge = v2 - v1
            point_vec = np.array([x, z]) - v1
            edge_length_sq = np.dot(edge, edge)

            if edge_length_sq == 0:
                # Degenerate edge
                projected = v1
            else:
                # Parameter t along edge (clamped to [0, 1] for segment)
                t = max(0, min(1, np.dot(point_vec, edge) / edge_length_sq))
                projected = v1 + t * edge

            # Distance to this edge
            dist_sq = (projected[0] - x)**2 + (projected[1] - z)**2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_point = projected

        return tuple(nearest_point) if nearest_point is not None else None
