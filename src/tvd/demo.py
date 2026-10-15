"""Synthetic demo scene: renders a video plus matching ground-truth detections.

The scene contains one example of every event type the system detects, so the full
pipeline (HSV signal reading, tracking, speed estimation, all rules) can be exercised
without a real camera or a trained model. Scene scale: 0.05 m/pixel, 25 fps.
"""
import json
import os

import numpy as np

FPS, W, H, N_FRAMES = 25, 960, 540, 700
CAR_W, CAR_H = 90, 45
LANE_A_Y, LANE_B_Y = 242, 300           # lane A moves right, lane B moves left
CYCLE = [("green", 10), ("yellow", 2), ("red", 10), ("green", 100)]
LIGHT = dict(x=640, y=100, w=40, h=100)


def light_state(t):
    tt = t % sum(d for _, d in CYCLE)
    for s, d in CYCLE:
        if tt < d:
            return s
        tt -= d


def _brake(f, f_brake, x_brake, base, steps=(4, 2, 0)):
    if f <= f_brake:
        return base(f)
    return x_brake + sum(steps[:f - f_brake])


# id, class, y-centre, x(frame), colour (BGR)
VEHICLES = [
    ("car_1", "car", LANE_A_Y, lambda f: 600 + 7 * (f - 75), (200, 120, 40)),      # normal
    ("car_2", "car", LANE_A_Y, lambda f: 600 + 11 * (f - 200), (40, 40, 200)),     # speeding (49.5 km/h)
    ("car_3", "car", LANE_A_Y, lambda f: 600 + 7 * (f - 375), (30, 160, 220)),     # runs the red light
    ("car_4", "car", LANE_B_Y, lambda f: -50 + 7 * (f - 60), (180, 60, 160)),      # wrong way
    ("car_5", "truck", LANE_B_Y, lambda f: max(300, 1000 - 7 * (f - 260)), (60, 160, 60)),  # stalls
    ("car_6", "car", LANE_A_Y,                                                      # brakes hard...
     lambda f: _brake(f, 610, 810, lambda g: 600 + 7 * (g - 580)), (200, 200, 50)),
    ("car_7", "car", LANE_A_Y,                                                      # ...and is hit
     lambda f: _brake(f, 620, 740, lambda g: 600 + 7 * (g - 600)), (150, 90, 220)),
]


def build_scenario():
    """Return {frame_idx: [detection dicts]} (ground-truth boxes with track ids)."""
    frames = {}
    for f in range(N_FRAMES):
        dets = []
        for vid, cls, yc, xf, _ in VEHICLES:
            x = xf(f)
            if -60 < x < 1020:
                dets.append({"track_id": vid, "cls": cls, "conf": 1.0,
                             "bbox": [x - CAR_W / 2, yc - CAR_H / 2, x + CAR_W / 2, yc + CAR_H / 2]})
        if 400 <= f <= 570:                                   # pedestrian crossing at x=420
            bottom = 180 + 1.2 * (f - 420)
            dets.append({"track_id": "ped_1", "cls": "person", "conf": 1.0,
                         "bbox": [410, bottom - 45, 430, bottom]})
        frames[f] = dets
    return frames


def render_frame(f, dets):
    import cv2
    img = np.full((H, W, 3), (70, 110, 70), np.uint8)
    cv2.rectangle(img, (0, 205), (W, 335), (80, 80, 80), -1)
    for x in range(0, W, 40):
        cv2.line(img, (x, 270), (x + 20, 270), (0, 220, 255), 2)
    cv2.line(img, (600, 205), (600, 270), (255, 255, 255), 4)           # stop line
    L = LIGHT
    cv2.rectangle(img, (L["x"], L["y"]), (L["x"] + L["w"], L["y"] + L["h"]), (10, 10, 10), -1)
    state = light_state(f / FPS)
    lamps = {"red": ((0, 0, 255), 125), "yellow": ((0, 215, 255), 150), "green": ((0, 200, 0), 175)}
    for name, (col, cy) in lamps.items():
        cv2.circle(img, (L["x"] + L["w"] // 2, cy), 9, col if name == state else (50, 50, 50), -1)
    colors = {v[0]: v[4] for v in VEHICLES}
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        if d["cls"] == "person":
            cv2.rectangle(img, (x1, y1 + 10), (x2, y2), (200, 200, 200), -1)
            cv2.circle(img, ((x1 + x2) // 2, y1 + 6), 6, (180, 200, 230), -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[d["track_id"]], -1)
            cv2.rectangle(img, (x1 + 55, y1 + 6), (x2 - 8, y2 - 6), (230, 230, 200), -1)
    return img


def generate(out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    scenario = build_scenario()
    video_path = os.path.join(out_dir, "demo.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N_FRAMES):
        writer.write(render_frame(f, scenario[f]))
    writer.release()
    det_path = os.path.join(out_dir, "demo_detections.json")
    with open(det_path, "w") as fh:
        json.dump({"fps": FPS, "frames": {str(k): v for k, v in scenario.items()}}, fh)
    return video_path, det_path
