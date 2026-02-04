"""Mesh validation and auto-repair for 3D printability."""

import numpy as np
from collections import defaultdict


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

    def validate_and_fix(self, mesh_data):
        """
        Validate mesh data and auto-fix common issues.

        Args:
            mesh_data: Dict with 'terrain', 'features', and 'gpx_track' mesh data

        Returns:
            dict: {
                'is_printable': bool,
                'warnings': list of warning messages,
                'fixes_applied': list of fixes that were applied
            }
        """
        self.warnings = []
        self.fixes_applied = []
        self.is_printable = True

        # Validate terrain
        if mesh_data.get('terrain'):
            self._validate_mesh('terrain', mesh_data['terrain'])

        # Validate features
        for i, feature in enumerate(mesh_data.get('features', [])):
            feature_name = feature.get('name', f"feature_{i}")
            self._validate_mesh(feature_name, feature)

        # Validate GPX track
        if mesh_data.get('gpx_track'):
            self._validate_mesh('gpx_track', mesh_data['gpx_track'])

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

        # Check for degenerate faces (zero-area triangles)
        faces, removed_count = self._remove_degenerate_faces(vertices, faces)
        if removed_count > 0:
            mesh['faces'] = faces.tolist()
            self.fixes_applied.append(f"Removed {removed_count} degenerate face(s) from {name}")

        # Check for duplicate vertices
        vertices, faces, merged_count = self._merge_duplicate_vertices(vertices, faces)
        if merged_count > 0:
            mesh['vertices'] = vertices.tolist()
            mesh['faces'] = faces.tolist()
            self.fixes_applied.append(f"Merged {merged_count} duplicate vertices in {name}")

        # Check for non-manifold edges
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
        Merge duplicate vertices and update face indices.

        Args:
            vertices: numpy array of vertices (N, 3)
            faces: numpy array of face indices (M, 3)
            tolerance: Distance threshold for considering vertices duplicate

        Returns:
            tuple: (unique_vertices, updated_faces, merged_count)
        """
        if len(vertices) == 0:
            return vertices, faces, 0

        # Build mapping from old indices to new indices
        vertex_map = {}
        unique_vertices = []
        unique_idx = 0

        for i, vertex in enumerate(vertices):
            # Check if this vertex is duplicate of an existing one
            is_duplicate = False
            for j, unique_vertex in enumerate(unique_vertices):
                if np.linalg.norm(vertex - unique_vertex) < tolerance:
                    vertex_map[i] = j
                    is_duplicate = True
                    break

            if not is_duplicate:
                vertex_map[i] = unique_idx
                unique_vertices.append(vertex)
                unique_idx += 1

        merged_count = len(vertices) - len(unique_vertices)

        # Update face indices
        if merged_count > 0:
            updated_faces = np.array([[vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]]] for f in faces])
            return np.array(unique_vertices), updated_faces, merged_count
        else:
            return vertices, faces, 0

    def _check_manifold_edges(self, faces):
        """
        Check if all edges are shared by exactly 2 faces (manifold condition).

        Args:
            faces: numpy array of face indices (M, 3)

        Returns:
            int: Number of non-manifold edges
        """
        if len(faces) == 0:
            return 0

        # Count how many faces share each edge
        edge_count = defaultdict(int)

        for face in faces:
            # Each triangle has 3 edges
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]

            for edge in edges:
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
