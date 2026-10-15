"""YAML configuration loading with defaults."""
import copy

import yaml

DEFAULTS = {
    "detector": {"type": "yolo", "weights": "yolov8n.pt", "conf": 0.35,
                 "tracker": "bytetrack.yaml", "replay_file": None},
    "classes": {"vehicles": ["car", "motorcycle", "bus", "truck"],
                "pedestrians": ["person"],
                "emergency_vehicles": ["ambulance", "fire truck", "police car"]},
    "calibration": {"meters_per_pixel": 0.05, "homography": None},
    "tracking": {"max_age_s": 1.0, "match_dist_px": 80, "speed_window_s": 0.5, "moving_kmh": 8},
    "zones": {"road": [], "lanes": [], "crosswalks": [], "no_stopping": []},
    "signal": {"mode": "manual", "roi": None, "min_pixels": 40, "smoothing_frames": 5,
               "cycle": [["green", 30], ["yellow", 3], ["red", 30]], "offset_s": 0},
    "stop_line": {"points": None, "violation_direction": [1, 0]},
    "speeding": {"limit_kmh": 50, "tolerance_kmh": 0, "min_frames": 8},
    "wrong_way": {"min_frames": 15, "min_speed_kmh": 8, "cos_threshold": -0.5},
    "emergency": {
        "collision": {"iou_threshold": 0.1, "decel_drop_ratio": 0.7,
                      "min_prior_speed_kmh": 15, "window_s": 2.0},
        "stalled": {"seconds": 8, "speed_kmh": 2, "ignore_red_within_px": 250},
        "pedestrian": {"seconds": 1.0},
    },
    "output": {"dir": "outputs", "save_snapshots": True, "save_video": True},
    "video": {"fps_override": None},
}


def _merge(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg
