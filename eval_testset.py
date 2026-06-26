"""Quick eval of models/final on the dataset test split.

Runs on CPU with the stock RT-DETR position embedding (no MPS patch), which
matches how the model was trained on Colab. Reports per-image top confidences,
detection counts at a threshold, mAP, and saves a few annotated images.
"""

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

MODEL_DIR = Path("models/final")
TEST_DIR = Path("data/test")
ANN = TEST_DIR / "_annotations.coco.json"
CATEGORY_REMAP = {0: 0, 1: 1}  # data is canonical (0=ball, 1=racket) via prepare_data.py
THRESHOLD = 0.5
OUT_DIR = Path("eval_out")
N_SAVE = 8

device = torch.device("cpu")
processor = RTDetrImageProcessor.from_pretrained(str(MODEL_DIR))
model = AutoModelForObjectDetection.from_pretrained(str(MODEL_DIR)).to(device).eval()
id2label = model.config.id2label
print("id2label:", id2label)

coco = json.load(open(ANN))
images = {im["id"]: im for im in coco["images"]}
ann_by_img: dict[int, list] = {}
for a in coco["annotations"]:
    if a["category_id"] in CATEGORY_REMAP:
        ann_by_img.setdefault(a["image_id"], []).append(a)

metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
OUT_DIR.mkdir(exist_ok=True)
colors = {"ball": (255, 255, 0), "racket": (255, 0, 255)}

n_with_det = 0
saved = 0
maxconf_ball, maxconf_racket = [], []

for i, (img_id, info) in enumerate(images.items()):
    path = TEST_DIR / info["file_name"]
    if not path.exists():
        continue
    img = Image.open(path).convert("RGB")
    W, H = img.size
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # threshold=0 to inspect raw top confidences
    raw = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(H, W)])[0]
    by_label = {0: 0.0, 1: 0.0}
    for s, l in zip(raw["scores"], raw["labels"]):
        by_label[int(l)] = max(by_label[int(l)], float(s))
    maxconf_ball.append(by_label[0])
    maxconf_racket.append(by_label[1])

    res = processor.post_process_object_detection(outputs, threshold=THRESHOLD, target_sizes=[(H, W)])[0]
    if len(res["scores"]) > 0:
        n_with_det += 1

    # mAP against GT (in absolute xyxy, original size). Feed ALL predictions
    # (threshold 0) so mAP is comparable to the training-time number.
    gt_boxes, gt_labels = [], []
    for a in ann_by_img.get(img_id, []):
        x, y, w, h = a["bbox"]
        gt_boxes.append([x, y, x + w, y + h])
        gt_labels.append(CATEGORY_REMAP[a["category_id"]])
    metric.update(
        [{"boxes": raw["boxes"].cpu(), "scores": raw["scores"].cpu(), "labels": raw["labels"].cpu()}],
        [{"boxes": torch.tensor(gt_boxes).reshape(-1, 4).float(), "labels": torch.tensor(gt_labels).long()}],
    )

    if saved < N_SAVE and len(res["scores"]) > 0:
        draw = ImageDraw.Draw(img)
        for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
            name = id2label[int(l)]
            x1, y1, x2, y2 = b.tolist()
            draw.rectangle([x1, y1, x2, y2], outline=colors.get(name, (0, 255, 0)), width=3)
            draw.text((x1, max(0, y1 - 12)), f"{name} {float(s):.2f}", fill=colors.get(name, (0, 255, 0)))
        img.save(OUT_DIR / f"det_{path.stem}.jpg")
        saved += 1

n = len(maxconf_ball)
print(f"\nImages: {n}")
print(f"Images with >= 1 detection @ {THRESHOLD}: {n_with_det}/{n}")
print(f"Mean top-confidence  ball  : {sum(maxconf_ball)/n:.3f}   max: {max(maxconf_ball):.3f}")
print(f"Mean top-confidence  racket: {sum(maxconf_racket)/n:.3f}   max: {max(maxconf_racket):.3f}")
m = metric.compute()
print(f"\nmAP@[.5:.95]: {m['map'].item():.4f}   mAP@50: {m['map_50'].item():.4f}   mAP@75: {m['map_75'].item():.4f}")
print(f"Saved {saved} annotated images to {OUT_DIR}/")
