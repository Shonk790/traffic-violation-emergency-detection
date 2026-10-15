#!/usr/bin/env python3
"""Traffic violation & emergency detection - command line entry point."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tvd.config import load_config  # noqa: E402
from tvd.detectors import ReplayDetector, YoloDetector  # noqa: E402
from tvd.pipeline import run  # noqa: E402


def build_detector(cfg, args):
    d = cfg["detector"]
    kind = args.detector or d["type"]
    if kind == "replay":
        path = args.replay or d.get("replay_file")
        if not path:
            sys.exit("Replay detector needs --replay or detector.replay_file")
        return ReplayDetector(path)
    c = cfg["classes"]
    keep = c["vehicles"] + c["pedestrians"] + (c.get("emergency_vehicles") or [])
    return YoloDetector(args.weights or d["weights"], d["conf"], d["tracker"], keep)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="video file path or camera index/RTSP URL")
    p.add_argument("--config", default=os.path.join(ROOT, "config", "demo.yaml"))
    p.add_argument("--detector", choices=["yolo", "replay"], help="override detector.type")
    p.add_argument("--weights", help="YOLO weights (default from config)")
    p.add_argument("--replay", help="detections JSON for the replay detector")
    p.add_argument("--output", help="output directory (default from config)")
    p.add_argument("--show", action="store_true", help="display live annotated video")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    a = p.parse_args()

    cfg = load_config(a.config)
    source = int(a.source) if a.source.isdigit() else a.source
    run(source, cfg, build_detector(cfg, a), a.output, a.show, a.max_frames,
        save_video=False if a.no_video else None)


if __name__ == "__main__":
    main()
