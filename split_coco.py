"""Split a single-folder COCO export into train/valid/test.

Roboflow can export everything into one folder with no validation split (the
ap-tennis v2 export puts all 6,692 images under train/). This carves it into
train/valid/test by a seeded random split over images, copies each image into its
split folder, and writes a per-split _annotations.coco.json. Category ids are
preserved as-is — the remap to model classes happens at training time (train.py).

Usage:
  uv run python split_coco.py SRC_DIR [--out data] [--val 0.1] [--test 0.05] [--seed 42]

SRC_DIR is the folder holding the images + _annotations.coco.json (e.g. the
`train/` folder of the unzipped export). Output goes to OUT/{train,valid,test}/.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="folder with images + _annotations.coco.json")
    ap.add_argument("--out", type=Path, default=Path("data"), help="output data dir")
    ap.add_argument("--val", type=float, default=0.10, help="validation fraction")
    ap.add_argument("--test", type=float, default=0.05, help="test fraction (0 to skip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ann", default="_annotations.coco.json")
    args = ap.parse_args()

    coco = json.loads((args.src / args.ann).read_text())
    images = coco["images"]
    img_by_id = {im["id"]: im for im in images}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    ids = [im["id"] for im in images]
    random.Random(args.seed).shuffle(ids)
    n = len(ids)
    n_test = int(n * args.test)
    n_val = int(n * args.val)
    split_ids = {
        "test":  ids[:n_test],
        "valid": ids[n_test:n_test + n_val],
        "train": ids[n_test + n_val:],
    }

    base = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    print(f"Source: {args.src}  ({n} images, {len(coco['annotations'])} annotations)")
    for split, id_list in split_ids.items():
        if not id_list:
            print(f"{split:6}: 0 images (skipped)")
            continue
        out_dir = args.out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_images = [img_by_id[i] for i in id_list]
        sub_anns = [a for i in id_list for a in anns_by_img.get(i, [])]
        copied = missing = 0
        for im in sub_images:
            src_f = args.src / im["file_name"]
            if src_f.exists():
                shutil.copy2(src_f, out_dir / im["file_name"])
                copied += 1
            else:
                missing += 1
        (out_dir / args.ann).write_text(
            json.dumps({**base, "images": sub_images, "annotations": sub_anns})
        )
        note = f"  [!] {missing} image files missing" if missing else ""
        print(f"{split:6}: {len(sub_images):5} images (copied {copied}), "
              f"{len(sub_anns):5} annotations{note}")
    print(f"Done → {args.out}/")


if __name__ == "__main__":
    main()
