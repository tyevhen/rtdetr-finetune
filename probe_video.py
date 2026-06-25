"""Probe the model's confidence on real video frames (domain-shift check).

Samples N frames spread across the video, runs the detector on CPU (stock),
and reports the top ball/racket confidence per frame plus saved overlays.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

VIDEO = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/yevhent/Projects/tennis-coach/IMG_3999.mov")
MODEL_DIR = Path(__file__).parent / "models" / "final"
N = 12
OUT = Path("video_probe_out")
OUT.mkdir(exist_ok=True)

device = torch.device("cpu")
processor = RTDetrImageProcessor.from_pretrained(str(MODEL_DIR))
model = AutoModelForObjectDetection.from_pretrained(str(MODEL_DIR)).to(device).eval()
id2label = model.config.id2label
colors = {"ball": (255, 255, 0), "racket": (255, 0, 255)}

cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = [int(total * (i + 0.5) / N) for i in range(N)]
print(f"video frames: {total}  sampling: {idxs}")

print(f"\n{'frame':>8}  {'ball':>6}  {'racket':>6}")
for fi in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = frame.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    raw = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(H, W)])[0]
    top = {0: 0.0, 1: 0.0}
    for s, l in zip(raw["scores"], raw["labels"]):
        top[int(l)] = max(top[int(l)], float(s))
    print(f"{fi:>8}  {top[0]:>6.3f}  {top[1]:>6.3f}")

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    res = processor.post_process_object_detection(outputs, threshold=0.02, target_sizes=[(H, W)])[0]
    for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
        name = id2label[int(l)]
        x1, y1, x2, y2 = b.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=colors.get(name, (0, 255, 0)), width=3)
        draw.text((x1, max(0, y1 - 12)), f"{name} {float(s):.2f}", fill=colors.get(name, (0, 255, 0)))
    img.save(OUT / f"frame_{fi:07d}.jpg")

cap.release()
print(f"\nSaved overlays (threshold 0.02) to {OUT}/")
