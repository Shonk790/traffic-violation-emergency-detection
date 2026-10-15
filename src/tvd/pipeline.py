"""End-to-end video processing loop."""
import os
from collections import Counter

import cv2

from .calibration import Calibration
from .emergency import EmergencyRules
from .events import EventLogger
from .rules import ViolationRules
from .signal import SignalMonitor
from .tracking import TrackManager
from .visualize import draw_overlay


def run(source, cfg, detector, output_dir=None, show=False, max_frames=None, save_video=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError("Cannot open video source: %s" % source)
    fps = cfg["video"].get("fps_override") or cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_dir = output_dir or cfg["output"]["dir"]
    logger = EventLogger(out_dir, cfg["output"]["save_snapshots"])
    save_video = cfg["output"]["save_video"] if save_video is None else save_video

    tc = cfg["tracking"]
    tracker = TrackManager(Calibration.from_config(cfg), fps, tc["max_age_s"], tc["match_dist_px"],
                           tc["speed_window_s"], tc["moving_kmh"])
    signal = SignalMonitor(cfg)
    violations, emergencies = ViolationRules(cfg), EmergencyRules(cfg)

    writer, flagged, banners, idx = None, set(), [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break
        t = idx / fps
        tracks = tracker.update(detector.detect(frame, idx), t, idx)
        sig = signal.state(frame, t)
        events = violations.update(idx, t, tracks, sig) + emergencies.update(idx, t, tracks, sig)

        for e in events:
            flagged.update(e.track_ids)
            banners.append(("%s %s" % (e.type, ",".join(e.track_ids)), e.severity, t + 3.0))
        banners = [b for b in banners if b[2] > t]
        annotated = draw_overlay(frame, cfg, tracks, sig, [(b[0], b[1]) for b in banners], flagged)
        for e in events:
            logger.log(e, annotated)

        if save_video:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)
        if show:
            cv2.imshow("Traffic monitoring (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    logger.close()
    summary = Counter(e.type for e in logger.events)
    print("Processed %d frames @ %.1f fps | %d events" % (idx, fps, len(logger.events)))
    for k, v in sorted(summary.items()):
        print("  %-24s %d" % (k, v))
    print("Output: %s" % os.path.abspath(out_dir))
    return logger.events
