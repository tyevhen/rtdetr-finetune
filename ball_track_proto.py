"""Prototype (a): motion-gated ball detection diagnostic.

Static-camera footage → MOG2 background subtraction yields a motion mask.
We overlay, per frame:
  - cyan  : RT-DETR ball candidates (low threshold), with score
  - red   : ball-sized motion blobs (MOG2 contours, area-filtered)
  - green : RT-DETR ball candidate whose center falls in a motion blob (the
            "moving ball" we'd keep)
  - magenta: top racket
Goal: see whether the in-flight ball is (1) in RT-DETR's candidate set at all,
and (2) recoverable by intersecting with motion.

Usage: python ball_track_proto.py START [N]
"""

import sys
from pathlib import Path

sys.path.insert(0, "/Users/yevhent/Projects/tennis-coach")

import cv2
import numpy as np
from tennis_coach.detector import RacketBallDetector

VIDEO = Path("/Users/yevhent/Projects/tennis-coach/IMG_3999.mov")
START = int(sys.argv[1]) if len(sys.argv) > 1 else 30600
N = int(sys.argv[2]) if len(sys.argv) > 2 else 360
WARMUP = 60                      # frames before START to seed the background model
BALL_AREA = (30, 2500)           # px^2 range for a ball-sized motion blob
OUT = Path("ball_proto_out")
OUT.mkdir(exist_ok=True)
N_SAMPLE = 16

det = RacketBallDetector(threshold=0.003, max_per_class={"ball": 40, "racket": 1})
print("device:", det.device)

cap = cv2.VideoCapture(str(VIDEO))
mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, START - WARMUP))
sample_idxs = {START + int(N * i / N_SAMPLE) for i in range(N_SAMPLE)}

writer = None
print(f"\n{'frame':>7} {'ballcand':>9} {'mblobs':>7} {'matched':>8}  matched_pts")
fi = max(0, START - WARMUP)
while fi < START + N:
    ok, frame = cap.read()
    if not ok:
        break
    fg = mog.apply(frame)
    if fi < START:
        fi += 1
        continue

    # motion blobs, ball-sized
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mblobs = []
    for c in contours:
        a = cv2.contourArea(c)
        if BALL_AREA[0] <= a <= BALL_AREA[1]:
            x, y, w, h = cv2.boundingRect(c)
            mblobs.append((x, y, x + w, y + h))

    dets = det.detect(frame)
    ball_cands = [d for d in dets if d.label == "ball"]
    rackets = [d for d in dets if d.label == "racket"]

    def in_blob(box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for bx1, by1, bx2, by2 in mblobs:
            if bx1 - 6 <= cx <= bx2 + 6 and by1 - 6 <= cy <= by2 + 6:
                return True
        return False

    matched = [d for d in ball_cands if in_blob(d.box)]

    vis = frame.copy()
    for bx in mblobs:
        cv2.rectangle(vis, bx[:2], bx[2:], (0, 0, 255), 1)
    for d in ball_cands:
        cv2.rectangle(vis, d.box[:2], d.box[2:], (255, 255, 0), 1)
    for d in rackets:
        cv2.rectangle(vis, d.box[:2], d.box[2:], (255, 0, 255), 2)
    for d in matched:
        cv2.rectangle(vis, d.box[:2], d.box[2:], (0, 255, 0), 2)
        cv2.putText(vis, f"{d.score:.3f}", (d.box[0], max(0, d.box[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if writer is None:
        h, w = vis.shape[:2]
        writer = cv2.VideoWriter(str(OUT / "track.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    writer.write(vis)

    pts = [((d.box[0] + d.box[2]) // 2, (d.box[1] + d.box[3]) // 2) for d in matched]
    if fi in sample_idxs or matched:
        print(f"{fi:>7} {len(ball_cands):>9} {len(mblobs):>7} {len(matched):>8}  {pts}")
    if fi in sample_idxs:
        cv2.imwrite(str(OUT / f"f{fi:07d}.jpg"), vis)
    fi += 1

cap.release()
if writer:
    writer.release()
print(f"\nSaved overlay video + {N_SAMPLE} samples to {OUT}/")
