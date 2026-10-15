"""Emergency / incident heuristics: collisions, stalled vehicles, pedestrians on road."""
from .events import Event
from .geometry import iou, point_in_polygon, dist_point_to_segment


class EmergencyRules:
    def __init__(self, cfg):
        self.cfg = cfg
        self.e = cfg["emergency"]
        cl = cfg["classes"]
        self.vehicles = set(cl["vehicles"])
        self.pedestrians = set(cl["pedestrians"])
        self.emergency_cls = set(cl.get("emergency_vehicles") or [])
        self.road = cfg["zones"].get("road") or []
        self.crosswalks = cfg["zones"].get("crosswalks") or []
        sl = cfg.get("stop_line") or {}
        self.stop_pts = sl.get("points")
        self._fired = set()
        self._ped_since = {}

    def _once(self, key):
        if key in self._fired:
            return False
        self._fired.add(key)
        return True

    def update(self, frame_idx, t, tracks, signal_state):
        events = []
        vehicles = [tr for tr in tracks if tr.cls in self.vehicles]
        self._mark_decel(t, vehicles)
        events += self._collisions(frame_idx, t, vehicles)
        for tr in vehicles:
            events += self._stalled(frame_idx, t, tr, signal_state)
        for tr in tracks:
            if tr.cls in self.pedestrians:
                events += self._pedestrian(frame_idx, t, tr)
            if tr.cls in self.emergency_cls and self._once(("EMERGENCY_VEHICLE", tr.id)):
                events.append(Event("EMERGENCY_VEHICLE_DETECTED", "info", frame_idx, t, [tr.id],
                                    {"class": tr.cls}))
        return events

    def _mark_decel(self, t, vehicles):
        c = self.e["collision"]
        for tr in vehicles:
            if not tr.speed_valid:
                continue
            vmax = tr.recent_max_speed(t, 1.0)
            if vmax >= c["min_prior_speed_kmh"] and (vmax - tr.speed_kmh) / vmax >= c["decel_drop_ratio"]:
                tr.last_decel_t = t

    def _collisions(self, f, t, vehicles):
        c = self.e["collision"]
        events = []
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                if iou(a.bbox, b.bbox) < c["iou_threshold"]:
                    continue
                if t - max(a.last_decel_t, b.last_decel_t) > c["window_s"]:
                    continue
                ids = sorted([a.id, b.id])
                if self._once(("COLLISION", tuple(ids))):
                    events.append(Event("COLLISION_SUSPECTED", "critical", f, t, ids,
                                        {"iou": round(iou(a.bbox, b.bbox), 2)}))
        return events

    def _stalled(self, f, t, tr, signal_state):
        c = self.e["stalled"]
        if not tr.speed_valid:
            return []
        near_red = (signal_state == "red" and self.stop_pts and
                    dist_point_to_segment(tr.ground, self.stop_pts[0], self.stop_pts[1])
                    < c["ignore_red_within_px"])
        if tr.speed_kmh >= c["speed_kmh"] or near_red:
            tr.stationary_since = None
            return []
        if tr.stationary_since is None:
            tr.stationary_since = t
        if (tr.was_moving and t - tr.stationary_since >= c["seconds"]
                and self._once(("STALLED", tr.id))):
            return [Event("STALLED_VEHICLE", "warning", f, t, [tr.id],
                          {"stationary_s": round(t - tr.stationary_since, 1)})]
        return []

    def _pedestrian(self, f, t, tr):
        c = self.e["pedestrian"]
        g = tr.ground
        on_road = (self.road and point_in_polygon(g, self.road)
                   and not any(point_in_polygon(g, cw) for cw in self.crosswalks))
        if not on_road:
            self._ped_since.pop(tr.id, None)
            return []
        start = self._ped_since.setdefault(tr.id, t)
        if t - start >= c["seconds"] and self._once(("PEDESTRIAN", tr.id)):
            return [Event("PEDESTRIAN_ON_ROAD", "warning", f, t, [tr.id],
                          {"seconds_on_road": round(t - start, 1)})]
        return []
