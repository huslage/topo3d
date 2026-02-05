import unittest

import numpy as np

from app.utils.shape_clipper import CircleClipper, HexagonClipper, RectangleClipper, SquareClipper


class ClipPolygonAllOrNothingTests(unittest.TestCase):
    def test_circle_rejects_partially_outside_polygon(self):
        clipper = CircleClipper(center_x=0.0, center_z=0.0, radius=1.0)
        polygon = np.array([[0.0, 0.0], [0.6, 0.0], [1.5, 0.0]])
        self.assertIsNone(clipper.clip_polygon(polygon))

    def test_square_rejects_partially_outside_polygon(self):
        clipper = SquareClipper(center_x=0.0, center_z=0.0, half_width=1.0)
        polygon = np.array([[0.0, 0.0], [0.5, 0.0], [1.2, 0.0]])
        self.assertIsNone(clipper.clip_polygon(polygon))

    def test_rectangle_rejects_partially_outside_polygon(self):
        clipper = RectangleClipper(center_x=0.0, center_z=0.0, half_width=2.0, half_height=1.0)
        polygon = np.array([[0.0, 0.0], [1.0, 0.5], [2.5, 0.0]])
        self.assertIsNone(clipper.clip_polygon(polygon))

    def test_hexagon_rejects_partially_outside_polygon(self):
        clipper = HexagonClipper(center_x=0.0, center_z=0.0, radius=1.0)
        polygon = np.array([[0.0, 0.0], [0.2, 0.2], [2.0, 2.0]])
        self.assertIsNone(clipper.clip_polygon(polygon))

    def test_all_clippers_accept_fully_inside_polygon(self):
        polygon = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0]])
        clippers = [
            CircleClipper(center_x=0.0, center_z=0.0, radius=1.0),
            SquareClipper(center_x=0.0, center_z=0.0, half_width=1.0),
            RectangleClipper(center_x=0.0, center_z=0.0, half_width=1.0, half_height=1.0),
            HexagonClipper(center_x=0.0, center_z=0.0, radius=1.0),
        ]

        for clipper in clippers:
            result = clipper.clip_polygon(polygon)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
