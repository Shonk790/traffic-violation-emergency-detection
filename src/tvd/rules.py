"""Traffic-violation rules: speeding, red-light running, wrong-way, no-stopping."""
from .events import Event
from .geometry import point_in_polygon, crossed_line, cosine


class ViolationRules:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vehicles = set(cfg["classes"]["vehicles"])
        self.lanes = cfg["zones"].get("lanes") or []
        self.no_stop = cfg["zones"].get("no_stopping") or []
        sl = cfg.get("stop_line") or {}
        self.stop_pts = sl.get("points")
        self.stop_dir = sl.get("violation_direction", [1, 0])
        self.window_s = cfg["tracking"]["speed_window_s"]
        self._fired = set()
        self._speed_cnt, self._wrong_cnt, self._dwell = {}, {}, {}

    def _once(self, key):
        if key in self._fired:
            return False
        self._fired.add(key)
        return True

    def update(self, frame_idx, t, tracks, signal_state):
        events = []
        for tr in tracks:
            if tr.cls not in self.vehicles:
                continue
            events += self._speeding(frame_idx, t, tr)
            events += self._red_light(frame_idx, t, tr, signal_state)
            events += self._wrong_way(frame_idx, t, tr)
            events += self._no_stopping(frame_idx, t, tr)
        return events

    def _speeding(self, f, t, tr):
        c = self.cfg["speeding"]
        limit = c["limit_kmh"] + c.get("tolerance_kmh", 0)
        if tr.speed_valid and tr.speed_kmh > limit:
            self._speed_cnt[tr.id] = self._speed_cnt.get(tr.id, 0) + 1
        else:
            self._speed_cnt[tr.id] = 0
        if self._speed_cnt[tr.id] >= c["min_frames"] and self._once(("SPEEDING", tr.id)):
            return [Event("SPEEDING", "warning", f, t, [tr.id],
                          {"speed_kmh": round(tr.speed_kmh, 1), "limit_kmh": c["limit_kmh"]})]
        return []

    def _red_light(self, f, t, tr, signal_state):
        if signal_state != "red" or not self.stop_pts or len(tr.hist) < 2:
            return []
        prev, cur = tr.hist[-2], tr.hist[-1]
        p0, p1 = (prev[1], prev[2]), (cur[1], cur[2])
        move = (p1[0] - p0[0], p1[1] - p0[1])
        if (move[0] * self.stop_dir[0] + move[1] * self.stop_dir[1] > 0
                and crossed_line(p0, p1, self.stop_pts[0], self.stop_pts[1])
                and self._once(("RED_LIGHT_VIOLATION", tr.id))):
            return [Event("RED_LIGHT_VIOLATION", "critical", f, t, [tr.id],
                          {"signal": signal_state, "speed_kmh": round(tr.speed_kmh, 1)})]
        return []

    def _wrong_way(self, f, t, tr):
        c = self.cfg["wrong_way"]
        lane = next((l for l in self.lanes if point_in_polygon(tr.ground, l["polygon"])), None)
        if lane is None or not tr.speed_valid or tr.speed_kmh < c["min_speed_kmh"]:
            self._wrong_cnt[tr.id] = 0
            return []
        move = tr.motion_vector(t, self.window_s)
        if cosine(move, lane["direction"]) < c["cos_threshold"]:
            self._wrong_cnt[tr.id] = self._wrong_cnt.get(tr.id, 0) + 1
        else:
            self._wrong_cnt[tr.id] = 0
        if self._wrong_cnt[tr.id] >= c["min_frames"] and self._once(("WRONG_WAY", tr.id)):
            return [Event("WRONG_WAY", "critical", f, t, [tr.id],
                          {"lane": lane["name"], "speed_kmh": round(tr.speed_kmh, 1)})]
        return []

    def _no_stopping(self, f, t, tr):
        events = []
        for zone in self.no_stop:
            key = (zone["name"], tr.id)
            if tr.speed_valid and tr.speed_kmh < 2 and point_in_polygon(tr.ground, zone["polygon"]):
                start = self._dwell.setdefault(key, t)
                if t - start >= zone["max_seconds"] and self._once(("ILLEGAL_STOPPING", key)):
                    events.append(Event("ILLEGAL_STOPPING", "warning", f, t, [tr.id],
                                        {"zone": zone["name"], "stopped_s": round(t - start, 1)}))
            else:
                self._dwell.pop(key, None)
        return events
