"""Prototype (a) v3: motion-gated ball anchors -> locally-parabolic trajectory.

Improvements over v2:
  - Gaps between real detections are filled with a LOCAL parametric quadratic
    (x(t), y(t) each deg<=2, fit to nearby real anchors) instead of a straight
    line. Local fitting stays valid under any camera angle (a global image-space
    parabola does not, due to perspective/depth foreshortening). Falls back to
    linear when too few anchors are available.
  - Ball drawn as a circle, not a square.
  - Real detections vs interpolated points are visually separated so the
    synthetic path can't be mistaken for data: real = bright filled circle +
    solid trail; interpolated = small dim dot + thin dim trail. Never
    extrapolates beyond the outermost real anchors.

Usage: python ball_traj_proto.py START [N]
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
WARMUP = 60
BALL_AREA = (30, 2500)
EDGE = 20
MAXGAP = 6                # interpolate across gaps up to this many frames
MAXJUMP = 180             # px/frame; reject anchor jumping more than this
FIT_WINDOW = 4            # real anchors on each side used for the local quad fit
TRAIL = 25
OUT = Path("ball_traj_out")
OUT.mkdir(exist_ok=True)
N_SAMPLE = 16

det = RacketBallDetector(threshold=0.003, max_per_class={"ball": 40, "racket": 1})
print("device:", det.device)

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ── Pass 1: one motion-gated ball anchor per frame ───────────────────────────
cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, START - WARMUP))
anchors: dict[int, tuple[float, float]] = {}
rackets: dict[int, tuple[int, int, int, int]] = {}
frames: dict[int, np.ndarray] = {}
fi = max(0, START - WARMUP)
while fi < START + N:
    ok, frame = cap.read()
    if not ok:
        break
    fg = mog.apply(frame)
    if fi < START:
        fi += 1
        continue
    frames[fi] = frame

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mblobs = []
    for c in contours:
        if BALL_AREA[0] <= cv2.contourArea(c) <= BALL_AREA[1]:
            x, y, w, h = cv2.boundingRect(c)
            mblobs.append((x, y, x + w, y + h))

    dets = det.detect(frame)
    rk = [d for d in dets if d.label == "racket"]
    if rk:
        rackets[fi] = rk[0].box

    pts = []
    for d in dets:
        if d.label != "ball":
            continue
        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
        if cx < EDGE or cx > W - EDGE or cy < EDGE or cy > H - EDGE:
            continue
        if any(bx1 - 6 <= cx <= bx2 + 6 and by1 - 6 <= cy <= by2 + 6
               for bx1, by1, bx2, by2 in mblobs):
            pts.append((cx, cy))
    if pts:
        anchors[fi] = (float(np.mean([p[0] for p in pts])),
                       float(np.mean([p[1] for p in pts])))
    fi += 1

# ── Velocity gate: drop outlier anchors that jump implausibly ────────────────
real: dict[int, tuple[float, float]] = {}
prev_f = None
for f in sorted(anchors):
    p = anchors[f]
    if prev_f is not None and f - prev_f <= MAXGAP:
        pp = real[prev_f]
        if np.hypot(p[0] - pp[0], p[1] - pp[1]) > MAXJUMP * (f - prev_f):
            continue
    real[f] = p
    prev_f = f
print(f"real anchors: {len(real)} / {len(frames)} frames")

# ── Fill gaps with a LOCAL parametric quadratic ──────────────────────────────
real_frames = sorted(real)
interp: dict[int, tuple[float, float]] = {}


def local_fit(gap_a: int, gap_b: int) -> None:
    """Fill frames strictly between consecutive real anchors gap_a, gap_b."""
    lo = real_frames.index(gap_a)
    win = real_frames[max(0, lo - FIT_WINDOW + 1): lo + 1 + FIT_WINDOW]
    fs = np.array(win, dtype=float)
    xs = np.array([real[k][0] for k in win])
    ys = np.array([real[k][1] for k in win])
    deg = 2 if len(win) >= 3 else 1
    px, py = np.polyfit(fs, xs, deg), np.polyfit(fs, ys, deg)
    for f in range(gap_a + 1, gap_b):
        interp[f] = (float(np.polyval(px, f)), float(np.polyval(py, f)))


for a, b in zip(real_frames, real_frames[1:]):
    if 1 < b - a <= MAXGAP:
        local_fit(a, b)

traj = {**interp, **real}   # real overrides on shared frames (none here)
print(f"trajectory points: {len(traj)} ({len(real)} real + {len(interp)} interp)")

# ── Render: circle for ball, real vs interp visually separated ───────────────
sample_idxs = {START + int(N * i / N_SAMPLE) for i in range(N_SAMPLE)}
writer = cv2.VideoWriter(str(OUT / "traj.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
YELLOW, DIM = (0, 255, 255), (60, 140, 140)
for f in range(START, START + N):
    if f not in frames:
        continue
    vis = frames[f].copy()
    if f in rackets:
        b = rackets[f]
        cv2.rectangle(vis, b[:2], b[2:], (255, 0, 255), 2)

    trail_pts = [(k, traj[k]) for k in range(f - TRAIL, f + 1) if k in traj]
    for i in range(1, len(trail_pts)):
        (k0, p0), (k1, p1) = trail_pts[i - 1], trail_pts[i]
        seg_real = (k0 in real) and (k1 in real) and (k1 - k0 == 1)
        cv2.line(vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 YELLOW if seg_real else DIM, 2 if seg_real else 1)
    # dim dots for interpolated samples in the trail
    for k, p in trail_pts:
        if k in interp:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, DIM, -1)

    if f in traj:
        p = traj[f]
        if f in real:
            cv2.circle(vis, (int(p[0]), int(p[1])), 7, (0, 255, 0), -1)   # solid = real
        else:
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 255, 0), 1)    # hollow = interp
    writer.write(vis)
    if f in sample_idxs:
        cv2.imwrite(str(OUT / f"f{f:07d}.jpg"), vis)
writer.release()
print(f"Saved trajectory video + samples to {OUT}/")
