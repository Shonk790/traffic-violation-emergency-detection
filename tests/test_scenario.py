"""Runs the rule engine on the synthetic scene (no video / OpenCV needed)."""
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from tvd.calibration import Calibration
from tvd.config import load_config
from tvd.demo import FPS, N_FRAMES, build_scenario, CYCLE
from tvd.detectors import Detection
from tvd.emergency import EmergencyRules
from tvd.rules import ViolationRules
from tvd.signal import SignalMonitor
from tvd.tracking import TrackManager

EXPECTED = {
    ("SPEEDING", ("car_2",)),
    ("RED_LIGHT_VIOLATION", ("car_3",)),
    ("WRONG_WAY", ("car_4",)),
    ("STALLED_VEHICLE", ("car_5",)),
    ("COLLISION_SUSPECTED", ("car_6", "car_7")),
    ("PEDESTRIAN_ON_ROAD", ("ped_1",)),
}


def run_scene(strip_ids=False):
    cfg = load_config(os.path.join(ROOT, "config", "demo.yaml"))
    cfg["signal"] = {"mode": "manual", "cycle": [list(c) for c in CYCLE]}
    tm = TrackManager(Calibration.from_config(cfg), FPS)
    sig, vr, er = SignalMonitor(cfg), ViolationRules(cfg), EmergencyRules(cfg)
    scenario, events = build_scenario(), []
    for f in range(N_FRAMES):
        t = f / FPS
        dets = [Detection(tuple(d["bbox"]), d["cls"], 1.0, None if strip_ids else d["track_id"])
                for d in scenario[f]]
        tracks = tm.update(dets, t, f)
        s = sig.state(None, t)
        events += vr.update(f, t, tracks, s) + er.update(f, t, tracks, s)
    return events


class TestScenario(unittest.TestCase):
    def test_all_events_detected_with_no_false_positives(self):
        got = {(e.type, tuple(sorted(e.track_ids))) for e in run_scene()}
        self.assertEqual(got, EXPECTED)

    def test_normal_vehicle_never_flagged(self):
        flagged = {i for e in run_scene() for i in e.track_ids}
        self.assertNotIn("car_1", flagged)

    def test_speed_estimate(self):
        e = next(e for e in run_scene() if e.type == "SPEEDING")
        self.assertAlmostEqual(e.details["speed_kmh"], 49.5, delta=0.5)

    def test_works_without_detector_track_ids(self):
        """Built-in centroid association should recover the same event types."""
        types = {e.type for e in run_scene(strip_ids=True)}
        self.assertTrue({"SPEEDING", "RED_LIGHT_VIOLATION", "WRONG_WAY"} <= types)


if __name__ == "__main__":
    unittest.main()
