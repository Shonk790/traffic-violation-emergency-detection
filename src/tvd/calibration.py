"""Pixel -> ground-plane (metres) conversion, needed for real speed estimates."""
import numpy as np


def _fit_homography(img_pts, world_pts):
    rows = []
    for (x, y), (X, Y) in zip(img_pts, world_pts):
        rows.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        rows.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    _, _, vt = np.linalg.svd(np.array(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


class Calibration:
    """Either a homography (>= 4 image/world point pairs) or a flat scale.

    A homography is strongly preferred for real cameras because the metres-per-pixel
    ratio changes with distance from the camera. A flat scale is only valid for a
    top-down view or a small region of the image.
    """

    def __init__(self, meters_per_pixel=None, image_points=None, world_points=None):
        self.H = None
        self.mpp = None
        if image_points and world_points:
            if len(image_points) < 4 or len(image_points) != len(world_points):
                raise ValueError("Homography needs >= 4 matching image/world points")
            self.H = _fit_homography(np.array(image_points, float), np.array(world_points, float))
        elif meters_per_pixel:
            self.mpp = float(meters_per_pixel)
        else:
            raise ValueError("Provide meters_per_pixel or image_points + world_points")

    def to_world(self, pt):
        x, y = pt
        if self.H is None:
            return x * self.mpp, y * self.mpp
        v = self.H @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    @classmethod
    def from_config(cls, cfg):
        c = cfg.get("calibration", {}) or {}
        h = c.get("homography")
        if h:
            return cls(image_points=h["image_points"], world_points=h["world_points"])
        return cls(meters_per_pixel=c.get("meters_per_pixel"))
