"""Drawing helpers for the annotated output video."""
import cv2
import numpy as np

SIGNAL_COLORS = {"red": (0, 0, 255), "yellow": (0, 215, 255), "green": (0, 200, 0),
                 "unknown": (180, 180, 180)}
SEV_COLORS = {"info": (255, 200, 0), "warning": (0, 165, 255), "critical": (0, 0, 255)}


def draw_overlay(frame, cfg, tracks, signal_state, banners, flagged):
    out = frame.copy()
    for lane in cfg["zones"].get("lanes") or []:
        cv2.polylines(out, [np.array(lane["polygon"], np.int32)], True, (255, 200, 0), 1)
    sl = (cfg.get("stop_line") or {}).get("points")
    if sl:
        cv2.line(out, tuple(int(v) for v in sl[0]), tuple(int(v) for v in sl[1]),
                 SIGNAL_COLORS.get(signal_state, (255, 255, 255)), 3)
    vehicles = set(cfg["classes"]["vehicles"])
    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        color = (0, 0, 255) if tr.id in flagged else (0, 220, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = "%s %s" % (tr.id, tr.cls)
        if tr.cls in vehicles and tr.speed_valid:
            label += " %d km/h" % round(tr.speed_kmh)
        cv2.putText(out, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(out, "Signal: %s" % signal_state.upper(), (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, SIGNAL_COLORS.get(signal_state, (255, 255, 255)), 2, cv2.LINE_AA)
    for i, (text, sev) in enumerate(banners[-4:]):
        cv2.putText(out, text, (10, 50 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    SEV_COLORS.get(sev, (255, 255, 255)), 2, cv2.LINE_AA)
    return out
