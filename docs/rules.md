# Rule reference

All thresholds live in the YAML config. Speeds are km/h from ground-plane positions,
averaged over `tracking.speed_window_s` (default 0.5 s).

## Violations (`src/tvd/rules.py`)
| Rule | Fires when | Key parameters |
|------|-----------|----------------|
| SPEEDING | speed > `limit_kmh + tolerance_kmh` for `min_frames` consecutive frames | `speeding.*` |
| RED_LIGHT_VIOLATION | signal is `red`, the track's ground point crosses `stop_line.points`, and motion agrees with `violation_direction` | `stop_line.*`, `signal.*` |
| WRONG_WAY | in a lane, speed >= `min_speed_kmh`, cosine(motion, lane direction) < `cos_threshold` for `min_frames` | `wrong_way.*`, `zones.lanes[].direction` |
| ILLEGAL_STOPPING | speed < 2 km/h inside a `zones.no_stopping` polygon for `max_seconds` | `zones.no_stopping[]` |

## Emergencies (`src/tvd/emergency.py`)
| Rule | Fires when | Key parameters |
|------|-----------|----------------|
| COLLISION_SUSPECTED | two vehicle boxes overlap (IoU >= threshold) and one of them had a sudden speed drop (>= `decel_drop_ratio` from >= `min_prior_speed_kmh`) within `window_s` | `emergency.collision.*` |
| STALLED_VEHICLE | a vehicle that was previously moving stays below `speed_kmh` for `seconds`; ignored when waiting at a red light within `ignore_red_within_px` of the stop line | `emergency.stalled.*` |
| PEDESTRIAN_ON_ROAD | a `pedestrians` class track stays inside `zones.road` (outside `zones.crosswalks`) for `seconds` | `emergency.pedestrian.*` |
| EMERGENCY_VEHICLE_DETECTED | a track's class is in `classes.emergency_vehicles` | `classes.emergency_vehicles` |

Each event fires once per track (or track pair) to avoid duplicate alerts.

## Signal state (`src/tvd/signal.py`)
- `mode: roi`: counts bright red/yellow/green pixels (HSV) inside `signal.roi`, with a
  majority vote over `smoothing_frames` to avoid flicker.
- `mode: manual`: a fixed timing plan `cycle: [[state, seconds], ...]`.

## Adding a rule
Add a method to `ViolationRules` or `EmergencyRules` that inspects the `Track`
objects (`tr.ground`, `tr.speed_kmh`, `tr.hist`, ...) and returns `Event`s, then add a
scenario to `src/tvd/demo.py` and an assertion in `tests/test_scenario.py`.
