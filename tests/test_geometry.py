import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tvd.calibration import Calibration
from tvd.geometry import (point_in_polygon, iou, crossed_line,
                          dist_point_to_segment, cosine)


class TestGeometry(unittest.TestCase):
    def test_point_in_polygon(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon((5, 5), sq))
        self.assertFalse(point_in_polygon((15, 5), sq))

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_crossed_line(self):
        a, b = (600, 200), (600, 270)
        self.assertTrue(crossed_line((590, 240), (610, 240), a, b))
        self.assertFalse(crossed_line((590, 240), (598, 240), a, b))
        self.assertFalse(crossed_line((590, 300), (610, 300), a, b))   # beyond segment end

    def test_crossing_exactly_on_line_counted_once(self):
        a, b = (600, 200), (600, 270)
        self.assertTrue(crossed_line((593, 240), (600, 240), a, b))
        self.assertFalse(crossed_line((600, 240), (607, 240), a, b))

    def test_distance_and_cosine(self):
        self.assertAlmostEqual(dist_point_to_segment((5, 5), (0, 0), (10, 0)), 5.0)
        self.assertAlmostEqual(cosine((1, 0), (-1, 0)), -1.0)
        self.assertEqual(cosine((0, 0), (1, 0)), 0.0)


class TestCalibration(unittest.TestCase):
    def test_scale(self):
        self.assertEqual(Calibration(meters_per_pixel=0.05).to_world((100, 200)), (5.0, 10.0))

    def test_homography_maps_control_points(self):
        img = [(300, 400), (660, 400), (820, 700), (140, 700)]
        world = [(0, 0), (3.5, 0), (3.5, 10), (0, 10)]
        c = Calibration(image_points=img, world_points=world)
        for p, w in zip(img, world):
            x, y = c.to_world(p)
            self.assertAlmostEqual(x, w[0], places=4)
            self.assertAlmostEqual(y, w[1], places=4)

    def test_requires_input(self):
        with self.assertRaises(ValueError):
            Calibration()


if __name__ == "__main__":
    unittest.main()
