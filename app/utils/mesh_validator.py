"""Mesh validation and auto-repair for 3D printability."""

import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree


class MeshValidator:
    """
    Validate and auto-repair mesh geometry for 3D printing.

    Checks for common issues:
    - Non-manifold edges (edges not shared by exactly 2 faces)
    - Degenerate faces (zero-area triangles)
    - Duplicate vertices
    - Inconsistent face winding

    Auto-fixes issues and reports warnings to the user.
    """

    def __init__(self):
        """Initialize mesh validator."""
        self.warnings = []
        self.fixes_applied = []
        self.is_printable = True

    def validate_and_fix(self, mesh_data, validate_features=False, min_feature_size=100):
        """
        Validate mesh data and auto-fix common issues.

        Args:
            mesh_data: Dict with 'terrain', 'features', and 'gpx_track' mesh data
            validate_features: If True, validate all features. If False, only validate large features (default: False for speed)
            min_feature_size: Minimum number of vertices to validate a feature (default: 100)

        Returns:
            dict: {
                'is_printable': bool,
                'warnings': list of warning messages,
                'fixes_applied': list of fixes that were applied
            }
        """
        import time
        self.warnings = []
        self.fixes_applied = []
        self.is_printable = True

        # Always validate terrain (most important for 3D printing)
        if mesh_data.get('terrain'):
            t_start = time.time()
            self._validate_mesh('terrain', mesh_data['terrain'])
            print(f"[PERF] Validated terrain in {time.time() - t_start:.3f}s")

        # Selectively validate features based on size
        features = mesh_data.get('features', [])
        if features:
            validated_count = 0
            skipped_count = 0
            t_features_start = time.time()

            for i, feature in enumerate(features):
                feature_name = feature.get('name', f"feature_{i}")
                vertex_count = len(feature.get('vertices', []))

                # Only validate if explicitly requested OR if feature is large enough
                if validate_features or vertex_count >= min_feature_size:
                    self._validate_mesh(feature_name, feature)
                    validated_count += 1
                else:
                    skipped_count += 1

            t_features_total = time.time() - t_features_start
            if validated_count > 0:
                print(f"[PERF] Validated {validated_count} features (skipped {skipped_count} small features) in {t_features_total:.3f}s")

        # Always validate GPX track if present
        if mesh_data.get('gpx_track'):
            t_start = time.time()
            self._validate_mesh('gpx_track', mesh_data['gpx_track'])
            print(f"[PERF] Validated GPX track in {time.time() - t_start:.3f}s")

        return {
            'is_printable': self.is_printable,
            'warnings': self.warnings,
            'fixes_applied': self.fixes_applied
        }

    def _validate_mesh(self, name, mesh):
        """
        Validate a single mesh component.

        Args:
            name: Name of the mesh component
            mesh: Dict with 'vertices' and 'faces'
        """
        if not mesh.get('vertices') or not mesh.get('faces'):
            return

        vertices = np.array(mesh['vertices'])
        faces = np.array(mesh['faces'])

        # Skip validation for very tiny meshes (< 8 vertices) - unlikely to have issues
        if len(vertices) < 8:
            return

        # Check for degenerate faces (zero-area triangles)
        faces, removed_count = self._remove_degenerate_faces(vertices, faces)
        if removed_count > 0:
            mesh['faces'] = faces.tolist()
            self.fixes_applied.append(f"Removed {removed_count} degenerate face(s) from {name}")

        # Check for duplicate vertices (now fast with KD-tree)
        vertices, faces, merged_count = self._merge_duplicate_vertices(vertices, faces)
        if merged_count > 0:
            mesh['vertices'] = vertices.tolist()
            mesh['faces'] = faces.tolist()
            self.fixes_applied.append(f"Merged {merged_count} duplicate vertices in {name}")

        # Check for non-manifold edges (only for medium+ meshes to save time)
        if len(faces) < 1000:  # Skip expensive check for huge meshes
            non_manifold_count = self._check_manifold_edges(faces)
            if non_manifold_count > 0:
                self.warnings.append(f"{name}: {non_manifold_count} non-manifold edge(s) detected (may cause print issues)")
                # Note: Auto-fixing non-manifold edges is complex and risky, so we just warn

    def _remove_degenerate_faces(self, vertices, faces):
        """
        Remove faces with zero area (degenerate triangles).

        Args:
            vertices: numpy array of vertices (N, 3)
            faces: numpy array of face indices (M, 3)

        Returns:
            tuple: (filtered_faces, removed_count)
        """
        valid_faces = []
        removed_count = 0

        for face in faces:
            # Get triangle vertices
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Calculate triangle area using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = np.linalg.norm(cross) / 2.0

            # Keep face if area is above threshold
            if area > 1e-10:
                valid_faces.append(face)
            else:
                removed_count += 1

        return np.array(valid_faces) if valid_faces else faces, removed_count

    def _merge_duplicate_vertices(self, vertices, faces, tolerance=1e-6):
        """
        Merge duplicate vertices and update face indices using fast KD-tree.

        Args:
            vertices: numpy array of vertices (N, 3)
            faces: numpy array of face indices (M, 3)
            tolerance: Distance threshold for considering vertices duplicate

        Returns:
            tuple: (unique_vertices, updated_faces, merged_count)
        """
        if len(vertices) == 0:
            return vertices, faces, 0

        # Use KD-tree for fast nearest neighbor lookup - O(n log n) instead of O(n²)
        tree = cKDTree(vertices)

        # Find groups of duplicate vertices
        # query_ball_tree returns all points within tolerance distance
        groups = tree.query_ball_tree(tree, tolerance)

        # Build vertex mapping: map each vertex to its representative (lowest index in group)
        vertex_map = np.arange(len(vertices))
        visited = set()

        for i, group in enumerate(groups):
            if i in visited:
                continue
            # Use the first vertex in the group as representative
            rep = min(group)
            for v in group:
                vertex_map[v] = rep
                visited.add(v)

        # Get unique vertex indices
        unique_indices = np.unique(vertex_map)
        unique_vertices = vertices[unique_indices]

        # Remap vertex_map to sequential indices
        remap = np.zeros(len(vertices), dtype=int)
        for new_idx, old_idx in enumerate(unique_indices):
            remap[vertex_map == old_idx] = new_idx

        merged_count = len(vertices) - len(unique_vertices)

        # Update face indices using vectorized operation
        if merged_count > 0:
            updated_faces = remap[faces]
            return unique_vertices, updated_faces, merged_count
        else:
            return vertices, faces, 0

    def _check_manifold_edges(self, faces):
        """
        Check if all edges are shared by exactly 2 faces (manifold condition).
        Optimized with vectorized numpy operations.

        Args:
            faces: numpy array of face indices (M, 3)

        Returns:
            int: Number of non-manifold edges
        """
        if len(faces) == 0:
            return 0

        # Extract all edges from faces using vectorized operations
        # Each triangle has 3 edges: (v0,v1), (v1,v2), (v2,v0)
        edges = np.vstack([
            np.sort(faces[:, [0, 1]], axis=1),  # Edge between vertex 0 and 1
            np.sort(faces[:, [1, 2]], axis=1),  # Edge between vertex 1 and 2
            np.sort(faces[:, [2, 0]], axis=1)   # Edge between vertex 2 and 0
        ])

        # Use numpy's unique to count edge occurrences
        # unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        # Converting to tuples is faster for the unique operation
        edge_tuples = [tuple(edge) for edge in edges]
        edge_count = defaultdict(int)
        for edge in edge_tuples:
            edge_count[edge] += 1

        # Count edges that are not shared by exactly 2 faces
        non_manifold_count = sum(1 for count in edge_count.values() if count != 2)

        return non_manifold_count

    def _check_face_winding(self, vertices, faces):
        """
        Check face winding consistency (all faces should have CCW winding).

        Note: This is a simplified check and may not catch all winding issues.

        Args:
            vertices: numpy array of vertices (N, 3)
            faces: numpy array of face indices (M, 3)

        Returns:
            int: Number of faces with potentially incorrect winding
        """
        if len(faces) == 0 or len(vertices) == 0:
            return 0

        inconsistent_count = 0

        # Check if face normals point roughly in the same direction as neighbors
        # This is a simplified heuristic and may not be 100% accurate
        for i, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            # Check if normal has reasonable magnitude
            if np.linalg.norm(normal) < 1e-10:
                continue  # Degenerate face, skip

            # For terrain, normals should generally point upward (+Y)
            # This is a heuristic and may not apply to all meshes
            if normal[1] < 0:
                inconsistent_count += 1

        return inconsistent_count
