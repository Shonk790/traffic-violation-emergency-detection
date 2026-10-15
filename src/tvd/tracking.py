"""Track bookkeeping: ground-plane positions, speed, motion state."""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    id: str
    cls: str
    bbox: tuple
    first_frame: int
    last_frame: int
    hist: deque = field(default_factory=lambda: deque(maxlen=400))        # (t, gx, gy) pixels
    world: deque = field(default_factory=lambda: deque(maxlen=400))       # (t, X, Y) metres
    speed_hist: deque = field(default_factory=lambda: deque(maxlen=400))  # (t, km/h)
    speed_kmh: float = 0.0
    speed_valid: bool = False
    was_moving: bool = False
    stationary_since: Optional[float] = None
    last_decel_t: float = -1e9

    @property
    def ground(self):
        """Bottom-centre of the box: the point touching the road."""
        return self.hist[-1][1], self.hist[-1][2]

    def update(self, bbox, t, frame_idx, calib, window_s, moving_kmh):
        self.bbox = bbox
        self.last_frame = frame_idx
        gx, gy = (bbox[0] + bbox[2]) / 2.0, bbox[3]
        self.hist.append((t, gx, gy))
        X, Y = calib.to_world((gx, gy))
        self.world.append((t, X, Y))

        ref = next((w for w in self.world if w[0] >= t - window_s), None)
        if ref is not None and t - ref[0] >= 0.6 * window_s:
            self.speed_kmh = math.hypot(X - ref[1], Y - ref[2]) / (t - ref[0]) * 3.6
            self.speed_valid = True
            self.speed_hist.append((t, self.speed_kmh))
            if self.speed_kmh > moving_kmh:
                self.was_moving = True
        else:
            self.speed_valid = False

    def recent_max_speed(self, t, window_s):
        return max((v for tt, v in self.speed_hist if tt >= t - window_s), default=0.0)

    def motion_vector(self, t, window_s):
        """Pixel displacement over the last `window_s` seconds."""
        ref = next((h for h in self.hist if h[0] >= t - window_s), None)
        if ref is None:
            return 0.0, 0.0
        return self.hist[-1][1] - ref[1], self.hist[-1][2] - ref[2]


class TrackManager:
    def __init__(self, calib, fps, max_age_s=1.0, match_dist_px=80,
                 speed_window_s=0.5, moving_kmh=8):
        self.calib = calib
        self.max_age_frames = max(1, int(max_age_s * fps))
        self.match_dist = match_dist_px
        self.window_s = speed_window_s
        self.moving_kmh = moving_kmh
        self.tracks = {}
        self._next = 1

    def _associate(self, det, matched, frame_idx):
        cx, cy = (det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2
        best, best_d = None, self.match_dist
        for tid, tr in self.tracks.items():
            if tid in matched or tr.cls != det.cls or frame_idx - tr.last_frame > self.max_age_frames:
                continue
            tx, ty = (tr.bbox[0] + tr.bbox[2]) / 2, (tr.bbox[1] + tr.bbox[3]) / 2
            d = math.hypot(cx - tx, cy - ty)
            if d < best_d:
                best, best_d = tid, d
        if best is None:
            best = "t%d" % self._next
            self._next += 1
        return best

    def update(self, detections, t, frame_idx):
        matched = set()
        for det in detections:
            tid = det.track_id if det.track_id is not None else self._associate(det, matched, frame_idx)
            tid = str(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = Track(id=tid, cls=det.cls, bbox=det.bbox,
                           first_frame=frame_idx, last_frame=frame_idx)
                self.tracks[tid] = tr
            tr.cls = det.cls
            tr.update(det.bbox, t, frame_idx, self.calib, self.window_s, self.moving_kmh)
            matched.add(tid)
        for tid in [k for k, v in self.tracks.items() if frame_idx - v.last_frame > self.max_age_frames]:
            del self.tracks[tid]
        return [self.tracks[i] for i in matched]
