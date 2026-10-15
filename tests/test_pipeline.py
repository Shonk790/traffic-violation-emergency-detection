"""Full pipeline on the rendered demo video (reads the traffic light from pixels)."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


@unittest.skipUnless(HAVE_CV2, "OpenCV not installed")
class TestPipeline(unittest.TestCase):
    def test_demo_video_end_to_end(self):
        from tvd.config import load_config
        from tvd.demo import generate
        from tvd.detectors import ReplayDetector
        from tvd.pipeline import run

        with tempfile.TemporaryDirectory() as tmp:
            video, dets = generate(os.path.join(tmp, "demo"))
            cfg = load_config(os.path.join(ROOT, "config", "demo.yaml"))
            out = os.path.join(tmp, "out")
            events = run(video, cfg, ReplayDetector(dets), out, save_video=True)
            types = sorted(e.type for e in events)
            self.assertEqual(types, sorted([
                "SPEEDING", "RED_LIGHT_VIOLATION", "WRONG_WAY",
                "STALLED_VEHICLE", "COLLISION_SUSPECTED", "PEDESTRIAN_ON_ROAD"]))
            self.assertTrue(os.path.exists(os.path.join(out, "events.csv")))
            self.assertTrue(os.path.exists(os.path.join(out, "annotated.mp4")))
            self.assertEqual(len(os.listdir(os.path.join(out, "snapshots"))), 6)


class TestConfig(unittest.TestCase):
    def test_template_loads(self):
        from tvd.config import load_config
        cfg = load_config(os.path.join(ROOT, "config", "template.yaml"))
        self.assertEqual(cfg["detector"]["type"], "yolo")
        self.assertIsNotNone(cfg["calibration"]["homography"])


if __name__ == "__main__":
    unittest.main()
