"""Visual eval: draw GT vs top predictions (with confidences) per test image.

For each test image we draw ground-truth boxes (green) and the model's top
predictions per class (regardless of threshold, since the head is poorly
calibrated) with confidence labels. We also print a per-image summary sorted
worst-first (lowest top confidence on a class that actually has GT) so the
hardest / lowest-confidence cases are easy to find.

Run:  uv run python eval_visualize.py
Output: eval_vis/ (annotated images) + console table.
"""

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

MODEL_DIR = Path("models/final")
TEST_DIR = Path("data/test")
ANN = TEST_DIR / "_annotations.coco.json"
CATEGORY_REMAP = {0: 0, 1: 1}  # data is canonical (0=ball, 1=racket) via prepare_data.py
CLASS_NAMES = {0: "ball", 1: "racket"}
TOP_K_PER_CLASS = 2   # how many predictions per class to draw
OUT_DIR = Path("eval_vis")

GT_COLOR = (0, 255, 0)
PRED_COLORS = {0: (255, 255, 0), 1: (255, 0, 255)}  # ball=yellow, racket=magenta

device = torch.device("cpu")
processor = RTDetrImageProcessor.from_pretrained(str(MODEL_DIR))
model = AutoModelForObjectDetection.from_pretrained(str(MODEL_DIR)).to(device).eval()

coco = json.load(open(ANN))
images = {im["id"]: im for im in coco["images"]}
ann_by_img: dict[int, list] = {}
for a in coco["annotations"]:
    if a["category_id"] in CATEGORY_REMAP:
        ann_by_img.setdefault(a["image_id"], []).append(a)

OUT_DIR.mkdir(exist_ok=True)
rows = []  # (sort_key, filename, line)

for img_id, info in images.items():
    path = TEST_DIR / info["file_name"]
    if not path.exists():
        continue
    img = Image.open(path).convert("RGB")
    W, H = img.size
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    raw = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(H, W)])[0]

    # Top-k predictions per class.
    preds_by_class: dict[int, list] = {0: [], 1: []}
    order = torch.argsort(raw["scores"], descending=True)
    for idx in order.tolist():
        cls = int(raw["labels"][idx])
        if cls in preds_by_class and len(preds_by_class[cls]) < TOP_K_PER_CLASS:
            preds_by_class[cls].append((float(raw["scores"][idx]), raw["boxes"][idx].tolist()))

    # Ground truth per class.
    gt_by_class: dict[int, list] = {0: [], 1: []}
    for a in ann_by_img.get(img_id, []):
        cls = CATEGORY_REMAP[a["category_id"]]
        x, y, w, h = a["bbox"]
        gt_by_class[cls].append([x, y, x + w, y + h])

    # Draw.
    draw = ImageDraw.Draw(img)
    for cls, boxes in gt_by_class.items():
        for b in boxes:
            draw.rectangle(b, outline=GT_COLOR, width=2)
            draw.text((b[0], max(0, b[1] - 11)), f"GT {CLASS_NAMES[cls]}", fill=GT_COLOR)
    for cls, preds in preds_by_class.items():
        for s, b in preds:
            draw.rectangle(b, outline=PRED_COLORS[cls], width=2)
            draw.text((b[0], b[3] + 1), f"{CLASS_NAMES[cls]} {s:.3f}", fill=PRED_COLORS[cls])
    img.save(OUT_DIR / f"vis_{path.stem}.jpg")

    # Summary line. Sort key = lowest top-confidence among classes that have GT
    # (so missed / under-confident real objects float to the top).
    parts, sort_key = [], 1.0
    for cls in (0, 1):
        n_gt = len(gt_by_class[cls])
        top = preds_by_class[cls][0][0] if preds_by_class[cls] else 0.0
        parts.append(f"{CLASS_NAMES[cls]}: gt={n_gt} topconf={top:.3f}")
        if n_gt > 0:
            sort_key = min(sort_key, top)
    rows.append((sort_key, path.name, "  |  ".join(parts)))

rows.sort(key=lambda r: r[0])
print(f"\nSaved {len(rows)} annotated images to {OUT_DIR}/")
print("Green=GT, Yellow=ball pred, Magenta=racket pred. Sorted worst-first")
print("(lowest top-confidence on a class that has ground truth):\n")
print(f"{'top':>6}  {'file':<55}  detail")
for sort_key, name, line in rows:
    print(f"{sort_key:6.3f}  {name[:55]:<55}  {line}")
