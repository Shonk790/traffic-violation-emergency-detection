"""Detector back-ends: YOLO (Ultralytics) and replay of saved detections."""
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    bbox: tuple                      # (x1, y1, x2, y2) pixels
    cls: str
    conf: float = 1.0
    track_id: Optional[str] = None   # None -> the TrackManager associates by position


class YoloDetector:
    """Ultralytics YOLO detection + built-in multi-object tracking (ByteTrack by default).

    Weights are downloaded automatically on first use (e.g. yolov8n.pt). COCO-trained
    weights do NOT contain ambulance / fire-truck classes: train a custom model and list
    its class names under `classes.emergency_vehicles` in the config to use that feature.
    """

    def __init__(self, weights, conf, tracker, keep_classes):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Install ultralytics: pip install ultralytics") from e
        self.model = YOLO(weights)
        self.conf = conf
        self.tracker = tracker
        self.keep = set(keep_classes)

    def detect(self, frame, frame_idx=None):
        res = self.model.track(frame, persist=True, conf=self.conf, tracker=self.tracker,
                               verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        out = []
        for box, c, k, tid in zip(xyxy, confs, cls_ids, ids):
            name = self.model.names[int(k)]
            if name not in self.keep:
                continue
            out.append(Detection(tuple(float(v) for v in box), name, float(c),
                                 None if tid is None else str(int(tid))))
        return out


class ReplayDetector:
    """Replays detections from JSON: {"frames": {"0": [{"bbox": [..], "cls": "car", "track_id": "a"}]}}.

    Useful for deterministic tests, debugging rules, and re-running the rule engine on
    detections saved from a heavier model.
    """

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.frames = data["frames"]

    def detect(self, frame, frame_idx=None):
        out = []
        for d in self.frames.get(str(frame_idx), []):
            tid = d.get("track_id")
            out.append(Detection(tuple(d["bbox"]), d["cls"], d.get("conf", 1.0),
                                 None if tid is None else str(tid)))
        return out
