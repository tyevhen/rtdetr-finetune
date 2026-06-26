"""Build the unified train/valid/test detection dataset from one or more COCO exports.

We train on two classes: ball (0) and racket (1). Roboflow exports disagree on
category ids and even on split layout, so this script normalizes every source to
the canonical model classes and merges them into data/{train,valid,test}/.

Each source declares its own remap (source category id -> canonical id); any
category not in the remap (players, persons, ball boys, supercategory rows) is
dropped. This is why a single global remap won't work: the SAME numeric id means
different things across datasets (e.g. id 5 is "racket" in ap-tennis v2 but
"person" in the Tennis_ball export), so normalization has to happen per source.

Each source points at its .zip. The script auto-extracts it (only if not already
extracted) and auto-locates the folder(s) holding _annotations.coco.json, so you
don't have to unzip by hand or match exact folder names. Single-folder sources are
carved into train/valid/test with a seeded random image split; pre-split sources
keep their own partition. Image files are copied with a per-source prefix so names
never collide, and image/annotation ids are reassigned to stay unique across the
merged file. Output categories are always 0=ball, 1=racket, so downstream
(train.py, eval, stats) uses an identity remap.

Usage (Windows / any OS), with the .zip files sitting in the repo folder:
  uv run python prepare_data.py --clean
  uv run python prepare_data.py --clean --out data --val 0.10 --test 0.05 --seed 42

Edit SOURCES to add/point at your exports. A source whose zip and extract dir are
both missing is skipped with a warning, so you can run with just what you have.
"""

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ANN = "_annotations.coco.json"
SPLITS = ["train", "valid", "test"]

# Canonical model classes (must match train.py ID2LABEL and models/final config).
CANON = {0: "ball", 1: "racket"}
CANON_CATEGORIES = [{"id": i, "name": n, "supercategory": "none"} for i, n in CANON.items()]

# Each source: its zip, where to extract it, whether it ships its own splits, and
# how its category ids map onto the canonical classes (unlisted ids are dropped).
SOURCES: list[dict[str, Any]] = [
    {
        # ap-tennis v2 — single folder (everything under train/), ball + racket + people.
        "name": "apt",
        "zip": Path("ap-tennis.v2i.coco.zip"),
        "extract": Path("export_apt"),
        "presplit": False,
        "remap": {1: 0, 5: 1},  # ball -> ball, tennis racquet -> racket
    },
    {
        # Tennis_ball — ball-only, already split; ball label fragmented across ids.
        "name": "ball",
        "zip": Path("Tennis_ball.coco.zip"),
        "extract": Path("export_ball"),
        "presplit": True,
        "remap": {1: 0, 2: 0, 4: 0, 6: 0, 3: 1},  # all "tennis ball" variants -> ball
    },
]


def ensure_extracted(zip_path: Path, extract_dir: Path) -> Path | None:
    """Return a directory holding the export, extracting the zip if needed."""
    already = extract_dir.exists() and any(extract_dir.rglob(ANN))
    if already:
        return extract_dir
    if not zip_path.exists():
        return None
    print(f"       extracting {zip_path} -> {extract_dir}/")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Map split name -> dir containing its _annotations.coco.json (robust to nesting)."""
    out: dict[str, Path] = {}
    for ann in root.rglob(ANN):
        if ann.parent.name in SPLITS:
            out.setdefault(ann.parent.name, ann.parent)
    return out


def find_single_dir(root: Path) -> Path | None:
    """The one dir holding _annotations.coco.json for a single-folder export."""
    hits = sorted({p.parent for p in root.rglob(ANN)}, key=lambda p: len(p.parts))
    return hits[0] if hits else None


def remap_anns(coco: dict, remap: dict[int, int]) -> dict[int, list[dict]]:
    """Group annotations by image, dropping/relabeling categories via remap."""
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        cid = a["category_id"]
        if cid not in remap:
            continue
        b = dict(a)
        b["category_id"] = remap[cid]
        by_img.setdefault(a["image_id"], []).append(b)
    return by_img


def split_ids(images: list[dict], seed: int, val: float, test: float) -> dict[str, list[int]]:
    ids = [im["id"] for im in images]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test, n_val = int(n * test), int(n * val)
    return {
        "test": ids[:n_test],
        "valid": ids[n_test:n_test + n_val],
        "train": ids[n_test + n_val:],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--val", type=float, default=0.10, help="val fraction for single-folder sources")
    ap.add_argument("--test", type=float, default=0.05, help="test fraction for single-folder sources")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="wipe OUT before building")
    args = ap.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    acc = {s: {"images": [], "annotations": []} for s in SPLITS}
    next_img = {s: 0 for s in SPLITS}
    next_ann = {s: 0 for s in SPLITS}
    cls_counts = {s: Counter() for s in SPLITS}
    missing = 0

    for src in SOURCES:
        name, remap = src["name"], src["remap"]
        root = ensure_extracted(src["zip"], src["extract"])
        if root is None:
            print(f"[skip] source '{name}': neither {src['zip']} nor extracted "
                  f"{src['extract']}/ found")
            continue

        # Resolve, per split, the (image-folder, image-records, anns-by-img) to add.
        jobs: list[tuple[str, Path, list[dict], dict[int, list[dict]]]] = []
        if src["presplit"]:
            split_dirs = find_split_dirs(root)
            if not split_dirs:
                print(f"[skip] source '{name}': no train/valid/test {ANN} under {root}/")
                continue
            for split, d in split_dirs.items():
                coco = json.loads((d / ANN).read_text())
                jobs.append((split, d, coco["images"], remap_anns(coco, remap)))
        else:
            d = find_single_dir(root)
            if d is None:
                print(f"[skip] source '{name}': no {ANN} found under {root}/")
                continue
            coco = json.loads((d / ANN).read_text())
            by_id = {im["id"]: im for im in coco["images"]}
            anns_by_img = remap_anns(coco, remap)
            for split, ids in split_ids(coco["images"], args.seed, args.val, args.test).items():
                jobs.append((split, d, [by_id[i] for i in ids], anns_by_img))

        src_total = 0
        for split, img_dir, images, anns_by_img in jobs:
            out_dir = args.out / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for im in images:
                src_f = img_dir / im["file_name"]
                if not src_f.exists():
                    missing += 1
                    continue
                new_fn = f"{name}__{im['file_name']}"
                shutil.copy2(src_f, out_dir / new_fn)
                new_id = next_img[split]
                next_img[split] += 1
                rec = dict(im)
                rec["id"], rec["file_name"] = new_id, new_fn
                acc[split]["images"].append(rec)
                for a in anns_by_img.get(im["id"], []):
                    na = dict(a)
                    na["id"], na["image_id"] = next_ann[split], new_id
                    next_ann[split] += 1
                    acc[split]["annotations"].append(na)
                    cls_counts[split][a["category_id"]] += 1
                src_total += 1
        print(f"[ok]   source '{name}': {src_total} images added from {root}/")

    wrote_any = False
    for split in SPLITS:
        if not acc[split]["images"]:
            continue
        wrote_any = True
        (args.out / split / ANN).write_text(json.dumps({
            "info": {"description": "merged tennis ball/racket dataset"},
            "licenses": [],
            "categories": CANON_CATEGORIES,
            "images": acc[split]["images"],
            "annotations": acc[split]["annotations"],
        }))

    if not wrote_any:
        print("\nNothing written — no sources found. Put the .zip exports in this "
              "folder (see SOURCES) and re-run.")
        return

    print("\n=== merged dataset ===")
    print(f"{'split':6}{'images':>9}{'ball':>9}{'racket':>9}")
    tot_i = tot_b = tot_r = 0
    for split in SPLITS:
        i = len(acc[split]["images"])
        b, r = cls_counts[split][0], cls_counts[split][1]
        tot_i, tot_b, tot_r = tot_i + i, tot_b + b, tot_r + r
        print(f"{split:6}{i:>9}{b:>9}{r:>9}")
    print(f"{'TOTAL':6}{tot_i:>9}{tot_b:>9}{tot_r:>9}")
    if missing:
        print(f"[!] {missing} referenced image files were missing and skipped")
    print(f"\nDone -> {args.out}/  (categories: {CANON})")


if __name__ == "__main__":
    main()
