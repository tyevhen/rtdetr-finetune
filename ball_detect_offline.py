"""Headless dense ball detection over a fraction of a video.

No live preview. Runs BallTracker with detect_every=1 (no frame skipping) for
maximum trajectory fidelity, writes an annotated MP4, and dumps the detected
ball positions to CSV.

Usage: python ball_detect_offline.py [FRACTION] [START_FRAME]
  FRACTION    portion of the video to process from START_FRAME (default 0.25)
  START_FRAME first frame (default 0)
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/yevhent/Projects/tennis-coach")

import cv2
from tennis_coach.ball_tracker import BallTracker

VIDEO = Path("/Users/yevhent/Projects/tennis-coach/IMG_3999.mov")
FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = Path("offline_out")
OUT.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
end = min(total, START + int(total * FRACTION))
n = end - START
print(f"video frames={total} fps={fps:.2f}  processing {START}..{end} ({n} frames)", flush=True)

bt = BallTracker(detect_every=1)        # no skipping = max fidelity
print("device:", bt.det.device, flush=True)

mp4 = OUT / f"ball_{START}_{end}.mp4"
csv_path = OUT / f"ball_{START}_{end}.csv"
writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

cap.set(cv2.CAP_PROP_POS_FRAMES, START)
rows: list[tuple[int, float, float]] = []
t0 = time.monotonic()
for fi in range(START, end):
    ok, frame = cap.read()
    if not ok:
        break
    bt.process(frame, fi)
    if fi in bt.real:                   # a real (non-interpolated) detection this frame
        x, y = bt.real[fi]
        rows.append((fi, round(x, 1), round(y, 1)))
    writer.write(frame)
    done = fi - START + 1
    if done % 1000 == 0:
        el = time.monotonic() - t0
        eta = el / done * (n - done)
        print(f"  {done}/{n}  detections={len(rows)}  "
              f"elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)

cap.release()
writer.release()
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "x", "y"])
    w.writerows(rows)

el = time.monotonic() - t0
print(f"\nDONE in {el/60:.1f}m  frames={n}  detections={len(rows)}", flush=True)
print(f"  video: {mp4}", flush=True)
print(f"  csv  : {csv_path}", flush=True)
