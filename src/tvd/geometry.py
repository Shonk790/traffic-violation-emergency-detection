"""Small geometry helpers (pure Python, no OpenCV needed)."""
import math


def point_in_polygon(pt, poly):
    """Ray-casting point-in-polygon test. `poly` is a list of (x, y)."""
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def crossed_line(p_prev, p_cur, a, b):
    """True if the movement p_prev -> p_cur crosses segment a-b.

    A point landing exactly on the line counts as a crossing on that frame only,
    so a vehicle is never missed (or counted twice) when it hits the line exactly.
    """
    d1, d2 = _orient(a, b, p_prev), _orient(a, b, p_cur)
    if not ((d1 < 0 <= d2) or (d1 > 0 >= d2)):
        return False
    d3, d4 = _orient(p_prev, p_cur, a), _orient(p_prev, p_cur, b)
    return d3 * d4 <= 0


def dist_point_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def cosine(u, v):
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu == 0 or nv == 0:
        return 0.0
    return (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
