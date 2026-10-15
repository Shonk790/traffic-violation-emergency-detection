"""Event record and logger (CSV + JSONL + annotated snapshots)."""
import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Event:
    type: str
    severity: str          # info | warning | critical
    frame: int
    time_s: float
    track_ids: List[str]
    details: dict = field(default_factory=dict)


class EventLogger:
    FIELDS = ["frame", "time_s", "type", "severity", "track_ids", "details"]

    def __init__(self, out_dir, save_snapshots=True):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.snap_dir = os.path.join(out_dir, "snapshots")
        self.save_snapshots = save_snapshots
        if save_snapshots:
            os.makedirs(self.snap_dir, exist_ok=True)
        self._csv_f = open(os.path.join(out_dir, "events.csv"), "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(self.FIELDS)
        self._jsonl = open(os.path.join(out_dir, "events.jsonl"), "w")
        self.events = []

    def log(self, event, annotated_frame=None):
        self.events.append(event)
        self._csv.writerow([event.frame, "%.2f" % event.time_s, event.type, event.severity,
                            "|".join(event.track_ids), json.dumps(event.details)])
        self._jsonl.write(json.dumps(asdict(event)) + "\n")
        self._csv_f.flush()
        self._jsonl.flush()
        if self.save_snapshots and annotated_frame is not None:
            import cv2
            ids = "-".join(event.track_ids)
            cv2.imwrite(os.path.join(self.snap_dir, "%06d_%s_%s.jpg" % (event.frame, event.type, ids)),
                        annotated_frame)

    def close(self):
        self._csv_f.close()
        self._jsonl.close()
