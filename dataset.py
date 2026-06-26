import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import RTDetrImageProcessor

logger = logging.getLogger(__name__)

# Tracks which splits have already logged their first batch so we only log once each.
_LOGGED_SPLITS: set[str] = set()


# ── COCO JSON schema ──────────────────────────────────────────────────────────

class _CocoAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]


class _CocoImage(TypedDict):
    id: int
    file_name: str


class _CocoCategory(TypedDict):
    id: int
    name: str


class _CocoData(TypedDict):
    images: list[_CocoImage]
    annotations: list[_CocoAnnotation]
    categories: list[_CocoCategory]


# ── Dataset ───────────────────────────────────────────────────────────────────

class TennisRacketDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        image_dir: str | Path,
        ann_file: str | Path,
        category_remap: dict[int, int],
    ) -> None:
        """
        Args:
            image_dir:       directory that contains the image files
            ann_file:        COCO-format annotations JSON
            category_remap:  maps COCO category_id (1-indexed) → model class index (0-indexed)
        """
        self.image_dir = Path(image_dir)
        self.category_remap = category_remap

        with open(ann_file) as f:
            coco = cast(_CocoData, json.load(f))

        self.images: dict[int, _CocoImage] = {img["id"]: img for img in coco["images"]}
        self.categories: dict[int, str] = {cat["id"]: cat["name"] for cat in coco["categories"]}

        self.ann_by_image: dict[int, list[_CocoAnnotation]] = {}
        for ann in coco["annotations"]:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Only keep images that have at least one mapped annotation.
        self.image_ids: list[int] = [
            img["id"] for img in coco["images"]
            if img["id"] in self.ann_by_image
        ]

        total_anns = sum(len(v) for v in self.ann_by_image.values())
        logger.info("Dataset — %s", ann_file)
        logger.info("  images with annotations : %d / %d",
                    len(self.image_ids), len(coco["images"]))
        logger.info("  total annotations       : %d  (avg %.1f / img)",
                    total_anns, total_anns / max(len(self.image_ids), 1))
        logger.info("  categories in JSON      : %s", self.categories)
        logger.info("  category_remap applied  : %s", category_remap)

        # Validate image files + bounding boxes up-front so problems surface early.
        missing: int = 0
        bad_boxes: int = 0
        unknown_cats: set[int] = set()
        for img_id in self.image_ids:
            if not (self.image_dir / self.images[img_id]["file_name"]).exists():
                missing += 1
            for ann in self.ann_by_image[img_id]:
                _, _, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    bad_boxes += 1
                if ann["category_id"] not in category_remap:
                    unknown_cats.add(ann["category_id"])

        if missing:
            logger.warning("  [!] missing image files  : %d", missing)
        if bad_boxes:
            logger.warning("  [!] bad bbox (w/h ≤ 0)  : %d  (will be passed through)", bad_boxes)
        if unknown_cats:
            logger.warning("  [!] unmapped category_ids: %s  (annotations skipped)", unknown_cats)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        image = Image.open(self.image_dir / img_info["file_name"]).convert("RGB")

        remapped_anns: list[dict[str, Any]] = [
            {**ann, "category_id": self.category_remap[ann["category_id"]]}
            for ann in self.ann_by_image[image_id]
            if ann["category_id"] in self.category_remap
        ]

        return {
            "image": image,
            "image_id": image_id,
            "annotations": remapped_anns,
            "orig_size": (image.height, image.width),
        }


# ── Collation ─────────────────────────────────────────────────────────────────

class BatchCollator:
    """Encodes a batch using the HF image processor (resize, normalize, bbox
    conversion COCO xywh absolute → cxcywh normalized).

    Implemented as a top-level class (not a closure) so it is picklable: on
    Windows ('spawn') and Python 3.14+ POSIX ('forkserver') the DataLoader sends
    the collate_fn to worker processes, which requires it to pickle cleanly.
    """

    def __init__(self, processor: RTDetrImageProcessor, split_name: str = "") -> None:
        self.processor = processor
        self.split_name = split_name

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        images  = [s["image"] for s in samples]
        targets = [
            {"image_id": s["image_id"], "annotations": s["annotations"]}
            for s in samples
        ]
        orig_sizes = [s["orig_size"] for s in samples]

        # Use processor.preprocess() directly: __call__ stubs only accept *args/**kwargs
        # and don't expose the `annotations` parameter, causing a false type error.
        # cast: preprocess returns BatchFeature (UserDict subclass), not dict.
        encoding: dict[str, Any] = cast(
            dict[str, Any],
            self.processor.preprocess(images=images, annotations=targets, return_tensors="pt"),
        )

        # Each worker process has its own _LOGGED_SPLITS, so this may log once per
        # worker — a one-time, harmless diagnostic.
        tag = f"collate:{self.split_name}"
        if tag not in _LOGGED_SPLITS:
            _LOGGED_SPLITS.add(tag)
            lbl = encoding["labels"][0]
            logger.info("[%s] first-batch diagnostics:", self.split_name or "collate")
            logger.info("  pixel_values shape : %s  dtype=%s",
                        tuple(encoding["pixel_values"].shape),
                        encoding["pixel_values"].dtype)
            logger.info("  batch size         : %d", len(images))
            logger.info("  orig_sizes (H,W)   : %s", orig_sizes)
            logger.info("  label keys         : %s", list(lbl.keys()))
            logger.info("  class_labels       : %s", lbl["class_labels"].tolist())
            logger.info("  boxes shape        : %s  (normalized cx,cy,w,h)",
                        tuple(lbl["boxes"].shape))
            if lbl["boxes"].numel() > 0:
                logger.info("  boxes[0]           : %s", lbl["boxes"][0].tolist())

        return encoding


def make_collate_fn(
    processor: RTDetrImageProcessor,
    split_name: str = "",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Returns a picklable collate callable (see BatchCollator)."""
    return BatchCollator(processor, split_name)


def make_val_loader(
    val_dataset: TennisRacketDataset,
    processor: RTDetrImageProcessor,
    batch_size: int,
) -> DataLoader[dict[str, Any]]:
    collate = make_collate_fn(processor, split_name="map_eval")
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=0,
    )
