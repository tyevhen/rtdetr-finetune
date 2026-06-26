"""Pre-training overview of the COCO detection data.

Prints, per split (train/valid/test) and overall: image counts, per-class
instance totals (after the train.py category remap), objects-per-image, and a
COCO-style box-size breakdown (small/medium/large). The size breakdown matters
here because tennis balls are tiny — a head that sees mostly small balls explains
the low ball confidence, and it flags class imbalance (ball vs racket) up front.

Run:  uv run python dataset_stats.py            # defaults to data/
      uv run python dataset_stats.py path/to/data
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Keep in sync with train.py.
CATEGORY_REMAP = {1: 0, 5: 1}            # ap-tennis v2: ball=1->0, tennis racquet=5->1
ID2LABEL = {0: "ball", 1: "racket"}
SPLITS = ["train", "valid", "test"]

# COCO area thresholds (pixels²): small < 32², medium < 96², else large.
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96
SIZE_ORDER = ["small", "medium", "large"]


def size_bucket(area: float) -> str:
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def analyze(ann_file: Path) -> dict:
    coco = json.load(open(ann_file))
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    n_images = len(coco["images"])

    raw_counts: Counter[int] = Counter(a["category_id"] for a in coco["annotations"])
    class_counts: Counter[int] = Counter()
    size_by_class: dict[int, Counter[str]] = {c: Counter() for c in ID2LABEL}
    objs_per_img: Counter[int] = Counter()
    imgs_with_ann: set[int] = set()

    for a in coco["annotations"]:
        cid = a["category_id"]
        if cid not in CATEGORY_REMAP:
            continue
        cls = CATEGORY_REMAP[cid]
        class_counts[cls] += 1
        w, h = a["bbox"][2], a["bbox"][3]
        size_by_class[cls][size_bucket(w * h)] += 1
        objs_per_img[a["image_id"]] += 1
        imgs_with_ann.add(a["image_id"])

    return {
        "n_images": n_images,
        "cat_names": cat_names,
        "raw_counts": raw_counts,
        "class_counts": class_counts,
        "size_by_class": size_by_class,
        "imgs_with_ann": len(imgs_with_ann),
        "total_objs": sum(class_counts.values()),
    }


def print_split(name: str, s: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  images                  : {s['n_images']}  "
          f"({s['imgs_with_ann']} with ≥1 mapped object, "
          f"{s['n_images'] - s['imgs_with_ann']} background-only)")
    print(f"  mapped objects          : {s['total_objs']}  "
          f"(avg {s['total_objs'] / max(s['imgs_with_ann'], 1):.2f} / annotated image)")

    # Raw categories present, flagging which are dropped by the remap.
    raw = ", ".join(
        f"{s['cat_names'].get(cid, '?')}({cid})={n}"
        f"{'' if cid in CATEGORY_REMAP else ' [dropped]'}"
        for cid, n in sorted(s["raw_counts"].items())
    )
    print(f"  raw categories in JSON  : {raw}")

    print(f"  {'class':<8}{'count':>8}{'small':>9}{'medium':>9}{'large':>9}")
    for cls, label in ID2LABEL.items():
        c = s["class_counts"].get(cls, 0)
        sz = s["size_by_class"][cls]
        print(f"  {label:<8}{c:>8}"
              f"{sz['small']:>9}{sz['medium']:>9}{sz['large']:>9}")
    counts = [s["class_counts"].get(c, 0) for c in ID2LABEL]
    if min(counts) > 0:
        hi, lo = max(counts), min(counts)
        print(f"  class imbalance (max:min): {hi / lo:.1f}:1")


def main() -> None:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(f"Dataset overview — {data_dir.resolve()}")
    print(f"category_remap = {CATEGORY_REMAP}  ->  {ID2LABEL}")

    totals = {
        "n_images": 0, "imgs_with_ann": 0, "total_objs": 0,
        "class_counts": Counter(),
        "size_by_class": {c: Counter() for c in ID2LABEL},
    }
    any_split = False
    for split in SPLITS:
        ann = data_dir / split / "_annotations.coco.json"
        if not ann.exists():
            print(f"\n=== {split} ===\n  (no _annotations.coco.json — skipped)")
            continue
        any_split = True
        s = analyze(ann)
        print_split(split, s)
        totals["n_images"] += s["n_images"]
        totals["imgs_with_ann"] += s["imgs_with_ann"]
        totals["total_objs"] += s["total_objs"]
        totals["class_counts"].update(s["class_counts"])
        for c in ID2LABEL:
            totals["size_by_class"][c].update(s["size_by_class"][c])

    if not any_split:
        print("\nNo splits found. Expected data/<split>/_annotations.coco.json.")
        return

    print("\n=== TOTAL (all splits) ===")
    print(f"  images                  : {totals['n_images']}  "
          f"({totals['imgs_with_ann']} annotated)")
    print(f"  mapped objects          : {totals['total_objs']}")
    print(f"  {'class':<8}{'count':>8}{'small':>9}{'medium':>9}{'large':>9}")
    for cls, label in ID2LABEL.items():
        c = totals["class_counts"].get(cls, 0)
        sz = totals["size_by_class"][cls]
        print(f"  {label:<8}{c:>8}"
              f"{sz['small']:>9}{sz['medium']:>9}{sz['large']:>9}")


if __name__ == "__main__":
    main()
