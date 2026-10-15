"""Traffic-signal state: read from the image (HSV) or from a fixed timing plan."""
from collections import Counter, deque


class SignalMonitor:
    def __init__(self, cfg):
        s = cfg["signal"]
        self.mode = s["mode"]
        self.roi = s.get("roi")
        self.min_pixels = s.get("min_pixels", 40)
        self.cycle = s.get("cycle") or []
        self.offset = s.get("offset_s", 0)
        self._recent = deque(maxlen=s.get("smoothing_frames", 5))
        if self.mode == "roi" and not self.roi:
            raise ValueError("signal.mode=roi requires signal.roi = [x, y, w, h]")

    def _from_roi(self, frame):
        import cv2
        x, y, w, h = self.roi
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        bright = (s_ > 100) & (v_ > 100)
        counts = {
            "red": int((bright & ((h_ < 10) | (h_ > 170))).sum()),
            "yellow": int((bright & (h_ >= 15) & (h_ <= 35)).sum()),
            "green": int((bright & (h_ >= 40) & (h_ <= 90)).sum()),
        }
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_pixels else "unknown"

    def _from_cycle(self, t):
        total = sum(d for _, d in self.cycle)
        tt = (t + self.offset) % total
        for state, dur in self.cycle:
            if tt < dur:
                return state
            tt -= dur
        return "unknown"

    def state(self, frame, t):
        if self.mode == "manual":
            return self._from_cycle(t)
        raw = self._from_roi(frame)
        if raw != "unknown":
            self._recent.append(raw)
        if not self._recent:
            return "unknown"
        return Counter(self._recent).most_common(1)[0][0]
