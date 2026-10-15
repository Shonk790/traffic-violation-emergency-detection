#!/usr/bin/env python3
"""Create demo/demo.mp4 and demo/demo_detections.json (synthetic test scene)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tvd.demo import generate  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "demo")
    video, dets = generate(out)
    print("Wrote", video)
    print("Wrote", dets)
