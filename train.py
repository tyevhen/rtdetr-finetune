import json
import logging
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

# Must be set before torch is imported so MPS fallback is active from the start.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForObjectDetection,
    PreTrainedModel,
    RTDetrImageProcessor,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import TennisRacketDataset, make_collate_fn, make_val_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT    = "PekingU/rtdetr_r18vd"   # fast; switch to rtdetr_r50vd for final weights
DATA_DIR      = Path("data")

# Override OUTPUT_DIR via env var so Colab can redirect to Drive without patching the file.
# e.g.: OUTPUT_DIR=/content/drive/MyDrive/rtdetr-finetune/rtdetr-tennis python train.py
OUTPUT_DIR    = Path(os.environ.get("OUTPUT_DIR", "rtdetr-tennis"))

# ── Hardware capability detection ───────────────────────────────────────────────
# Core hyperparameters (LR, schedule, EMA) are architecture/data-driven and do NOT
# change across machines. Only the throughput knobs below adapt to the device:
#   • RTX 5070 Ti (Blackwell, sm_120) → bf16, TF32, more workers. Requires a
#     CUDA 12.8+ / PyTorch build with sm_120 kernels, else CUDA ops won't launch.
#   • Colab T4 (Turing)               → no bf16/TF32; falls back to fp16 + NaN-safe.
#   • macOS / MPS                     → fp32, single-worker (dev / smoke only).
_ON_CUDA = torch.cuda.is_available()
_ON_MPS  = torch.backends.mps.is_available()

# Prefer bf16 wherever the GPU supports it (Blackwell/Ada/Ampere); else fp16 on CUDA.
BF16 = _ON_CUDA and torch.cuda.is_bf16_supported()
FP16 = _ON_CUDA and not BF16   # T4 etc.: fp16 + NaNSafeTrainer guard
PIN_MEMORY = _ON_CUDA

# Effective batch = BATCH_SIZE × GRAD_ACCUM; we target 16 (the RT-DETR recipe value)
# so the base LR of 1e-4 stays valid without re-tuning.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8" if _ON_CUDA else "4" if _ON_MPS else "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "2" if _ON_CUDA else "1"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "72"))

# Dataloader workers: Windows/Linux + CUDA are safe with spawn under the
# `if __name__ == "__main__"` guard; MPS must stay at 0 (fork/MPS issues).
NUM_WORKERS = int(os.environ.get(
    "NUM_WORKERS", "8" if _ON_CUDA else "0" if _ON_MPS else "2"
))

# torch.compile: best-effort speedup on CUDA. RT-DETR's deformable attention does
# not always compile cleanly, so it's opt-in via env (TORCH_COMPILE=1).
TORCH_COMPILE = _ON_CUDA and os.environ.get("TORCH_COMPILE", "0") == "1"

# ── Optimization recipe (hardware-independent) ──────────────────────────────────
LR            = 1e-4    # base / detection-head learning rate
BACKBONE_LR   = 1e-5    # 10× lower for the pretrained backbone
WEIGHT_DECAY  = 1e-4
WARMUP_RATIO  = 0.03    # linear warmup, then cosine decay over the full schedule

# Model EMA: smooths weights and improves both mAP and score quality. The decay is
# warmed up (see ModelEma) so short fine-tunes still get a useful average.
USE_EMA   = os.environ.get("USE_EMA", "1" if _ON_CUDA else "0") == "1"
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.9999"))

# Early stopping is a *safety net* only — the cosine schedule is meant to run to the
# end so the LR fully anneals. We monitor mAP@[.5:.95] (COCO primary), which keeps
# improving after mAP@50 saturates, so it won't cut the run short prematurely.
MAP_PATIENCE   = int(os.environ.get("MAP_PATIENCE", "20"))
MAP_MIN_EPOCHS = int(os.environ.get("MAP_MIN_EPOCHS", "20"))
MAP_THRESHOLD  = 5e-4  # minimum delta to count as improvement

CHECKPOINT_EVERY_N = 5  # keep one checkpoint every N epochs; all others are deleted

# prepare_data.py normalizes every source dataset to canonical ids (0=ball,
# 1=racket) and writes them into data/, so the training-time remap is identity.
CATEGORY_REMAP = {0: 0, 1: 1}  # canonical: 0=ball, 1=racket
ID2LABEL: dict[int, str]  = {0: "ball", 1: "racket"}
LABEL2ID: dict[str, int]  = {"ball": 0, "racket": 1}


# ── Checkpoint pruning ───────────────────────────────────────────────────────

class SaveEveryNCallback(TrainerCallback):
    """Deletes any checkpoint that does not fall on an epoch multiple of N."""

    def __init__(self, n: int) -> None:
        self.n = n

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **_kwargs: Any,
    ) -> None:
        epoch = round(state.epoch or 0)
        if epoch % self.n != 0:
            ckpt = Path(args.output_dir or str(OUTPUT_DIR)) / f"checkpoint-{state.global_step}"
            # Never prune the checkpoint load_best_model_at_end will reload from.
            if state.best_model_checkpoint and Path(state.best_model_checkpoint) == ckpt:
                return
            if ckpt.exists():
                shutil.rmtree(ckpt)
                logger.info("Pruned checkpoint %s (epoch %d not a multiple of %d)",
                            ckpt.name, epoch, self.n)


# ── MPS compatibility ─────────────────────────────────────────────────────────

def _patch_mps_rt_detr() -> None:
    # transformers hardcodes float64 in build_2d_sinusoidal_position_embedding,
    # which MPS doesn't support. float32 precision is sufficient for pos embeddings.
    import transformers.models.rt_detr.modeling_rt_detr as _m

    def _pos_embed_float32(
        width: int,
        height: int,
        embed_dim: int = 256,
        temperature: float = 10_000.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        grid_w = torch.arange(int(width),  dtype=torch.float32, device=device)
        grid_h = torch.arange(int(height), dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
        ).to(dtype=dtype)

    _m.build_2d_sinusoidal_position_embedding = _pos_embed_float32
    logger.info("Applied MPS float32 patch for RT-DETR sinusoidal position embedding.")


# ── mAP evaluation ────────────────────────────────────────────────────────────

def run_map_eval(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    processor: RTDetrImageProcessor,
    proc_H: int,
    proc_W: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    model.eval()
    n_images = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            model_inputs: dict[str, Any] = {"pixel_values": pixel_values}
            if "pixel_mask" in batch:
                model_inputs["pixel_mask"] = batch["pixel_mask"].to(device)

            outputs = model(**model_inputs)

            target_sizes: list[tuple[int, int]] = [(proc_H, proc_W)] * len(batch["labels"])
            results: list[dict[str, Any]] = cast(
                list[dict[str, Any]],
                processor.post_process_object_detection(
                    outputs, threshold=0.0, target_sizes=target_sizes
                ),
            )

            for result, lbl in zip(results, batch["labels"]):
                n_images += 1
                preds: dict[str, Any] = {
                    "boxes":  result["boxes"].cpu(),
                    "scores": result["scores"].cpu(),
                    "labels": result["labels"].cpu(),
                }
                gt_boxes = lbl["boxes"]   # (N, 4) normalized cxcywh
                if gt_boxes.numel() > 0:
                    cx, cy, w, h = gt_boxes.unbind(-1)
                    gt_xyxy = torch.stack([
                        (cx - w / 2) * proc_W,
                        (cy - h / 2) * proc_H,
                        (cx + w / 2) * proc_W,
                        (cy + h / 2) * proc_H,
                    ], dim=-1).cpu()
                else:
                    gt_xyxy = torch.zeros((0, 4))

                metric.update(
                    [preds],
                    [{"boxes": gt_xyxy, "labels": lbl["class_labels"].cpu()}],
                )

    res: dict[str, Any] = cast(dict[str, Any], metric.compute())
    return {
        "map":      res["map"].item(),
        "map_50":   res["map_50"].item(),
        "map_75":   res["map_75"].item(),
        "n_images": n_images,
    }


# ── Early-stopping callback with mAP evaluation ───────────────────────────────
class MAPEvalCallback(TrainerCallback):
    """Runs mAP evaluation on the validation set after every eval phase.

    Monitors mAP@[.5:.95] (the COCO primary metric) rather than mAP@50: it keeps
    improving after mAP@50 saturates, so it's a better signal for both checkpoint
    selection and the early-stop safety net. The best checkpoint is saved
    explicitly to best_map/ here (rather than via load_best_model_at_end, whose
    metric→checkpoint binding can't express "EMA weights at best mAP").

    When an EMACallback is supplied, eval and the saved best_map/ both use the EMA
    weights; training continues on the live weights afterwards.
    """

    def __init__(
        self,
        val_loader: DataLoader[dict[str, Any]],
        processor: RTDetrImageProcessor,
        patience: int,
        min_epochs: int,
        threshold: float,
        proc_H: int,
        proc_W: int,
        ema_cb: "EMACallback | None" = None,
    ) -> None:
        self.val_loader: DataLoader[dict[str, Any]] = val_loader
        self.processor: RTDetrImageProcessor = processor
        self.patience: int = patience
        self.min_epochs: int = min_epochs
        self.threshold: float = threshold
        self.best_map: float = 0.0
        self.no_improve: int = 0
        self.proc_H: int = proc_H
        self.proc_W: int = proc_W
        self.ema_cb: "EMACallback | None" = ema_cb

        logger.info("MAPEvalCallback ready:")
        logger.info("  processor output size : (%d, %d)", proc_H, proc_W)
        logger.info("  monitor=mAP@[.5:.95]  patience=%d  min_epochs=%d  threshold=%.0e",
                    patience, min_epochs, threshold)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module | None = None,
        **_kwargs: Any,
    ) -> None:
        if model is None:
            logger.warning("MAPEvalCallback: model not available, skipping mAP eval.")
            return

        # Evaluate (and save best_map/) on EMA weights when available.
        ema = self.ema_cb.ema if self.ema_cb is not None else None
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        try:
            res      = run_map_eval(model, self.val_loader, self.processor, self.proc_H, self.proc_W)
            map_val  = res["map"]
            map50    = res["map_50"]
            map75    = res["map_75"]
            n_images = res["n_images"]
            epoch    = int(state.epoch or 0)

            logger.info(
                "[Epoch %d] mAP eval%s  images=%d  "
                "mAP@[.5:.95]=%.4f  mAP@50=%.4f  mAP@75=%.4f",
                epoch, " (EMA)" if ema is not None else "", n_images, map_val, map50, map75,
            )

            if state.log_history:
                state.log_history[-1].update({
                    "eval_map":    map_val,
                    "eval_map_50": map50,
                    "eval_map_75": map75,
                })

            # Save best_map/ from the very first improving eval so a deliverable
            # always exists; only the early-stop counter waits for min_epochs.
            if map_val > self.best_map + self.threshold:
                logger.info("  mAP@[.5:.95] improved  %.4f → %.4f", self.best_map, map_val)
                self.best_map = map_val
                self.no_improve = 0
                best_map_dir = Path(args.output_dir or str(OUTPUT_DIR)) / "best_map"
                cast(PreTrainedModel, model).save_pretrained(str(best_map_dir))
                self.processor.save_pretrained(str(best_map_dir))
                logger.info("  Saved best_map checkpoint%s → %s",
                            " (EMA weights)" if ema is not None else "", best_map_dir)
            elif epoch >= self.min_epochs:
                self.no_improve += 1
                logger.info(
                    "  No improvement  %d / %d  (best mAP@[.5:.95]=%.4f)",
                    self.no_improve, self.patience, self.best_map,
                )
                if self.no_improve >= self.patience:
                    logger.info("  Early stopping triggered at epoch %d.", epoch)
                    control.should_training_stop = True
            else:
                logger.info(
                    "  Warmup: epoch %d < %d, early stopping not active yet.",
                    epoch, self.min_epochs,
                )
        finally:
            if ema is not None:
                ema.restore(model)


# ── Model EMA ─────────────────────────────────────────────────────────────────
class ModelEma:
    """Exponential moving average of the full model state_dict.

    The decay is warmed up — eff = min(decay, (1 + n) / (10 + n)) — so the average
    tracks quickly at the start and short fine-tunes still benefit, instead of
    staying pinned to the initial (pretrained) weights.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def store(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None


class EMACallback(TrainerCallback):
    """Maintains a ModelEma, updated once per optimizer step.

    Evaluation/checkpoint swapping is performed by MAPEvalCallback (which holds a
    reference to this callback), so val mAP and the saved best_map/ reflect the EMA
    weights while training itself continues on the live weights.
    """

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.ema: ModelEma | None = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
        model: nn.Module | None = None, **_kwargs: Any,
    ) -> None:
        if self.ema is None and model is not None:
            self.ema = ModelEma(model, self.decay)
            logger.info("EMA initialized from current weights (decay=%.4f).", self.decay)

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
        model: nn.Module | None = None, **_kwargs: Any,
    ) -> None:
        if self.ema is not None and model is not None:
            self.ema.update(model)


# ── Run provenance ────────────────────────────────────────────────────────────
def _git_commit() -> dict[str, Any]:
    """Best-effort current commit + dirty flag, so a checkpoint can be tied to code."""
    info: dict[str, Any] = {"commit": None, "dirty": None}
    root = Path(__file__).resolve().parent
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip())
    except Exception:
        pass
    return info


def _class_object_counts(ds: TennisRacketDataset) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for anns in ds.ann_by_image.values():
        for a in anns:
            cid = a["category_id"]
            if cid in ds.category_remap:
                counts[ID2LABEL[ds.category_remap[cid]]] += 1
    return dict(counts)


def write_run_metadata(
    final_dir: Path,
    train_ds: TennisRacketDataset,
    val_ds: TennisRacketDataset,
    best_val_map: float,
    test_res: dict[str, Any] | None,
) -> None:
    """Write training_run.json into the model dir so a committed checkpoint is
    self-describing — which code, data, hyperparameters and metrics produced it."""
    meta: dict[str, Any] = {
        "created":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git":             _git_commit(),
        "base_checkpoint": CHECKPOINT,
        "id2label":        ID2LABEL,
        "hyperparameters": {
            "max_epochs":      MAX_EPOCHS,
            "batch_size":      BATCH_SIZE,
            "grad_accum":      GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "lr":              LR,
            "backbone_lr":     BACKBONE_LR,
            "weight_decay":    WEIGHT_DECAY,
            "scheduler":       "cosine",
            "warmup_ratio":    WARMUP_RATIO,
            "ema_decay":       EMA_DECAY if USE_EMA else None,
            "precision":       "bf16" if BF16 else "fp16" if FP16 else "fp32",
        },
        "data": {
            "train_images":  len(train_ds),
            "val_images":    len(val_ds),
            "train_objects": _class_object_counts(train_ds),
            "val_objects":   _class_object_counts(val_ds),
        },
        "metrics": {
            "best_val_map_5095": round(best_val_map, 4),
            "test": (
                {k: round(v, 4) for k, v in test_res.items() if k != "n_images"}
                if test_res else None
            ),
        },
    }
    out = final_dir / "training_run.json"
    out.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote run metadata → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if torch.backends.mps.is_available():
        _patch_mps_rt_detr()

    if _ON_CUDA:
        # TF32 matmuls: free speedup on Ampere+ (incl. Blackwell); ignored elsewhere.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # RTDetrImageProcessor.from_pretrained gives Pylance the exact type,
    # avoiding the untyped return of AutoImageProcessor.from_pretrained.
    logger.info("Loading processor from: %s", CHECKPOINT)
    processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT)
    logger.info("  size=%s  do_resize=%s",
                processor.size, getattr(processor, "do_resize", "?"))

    size   = processor.size
    proc_H: int = size.get("height", size.get("shortest_edge", 640))
    proc_W: int = size.get("width",  size.get("shortest_edge", 640))

    # Datasets
    train_ds = TennisRacketDataset(
        DATA_DIR / "train",
        DATA_DIR / "train" / "_annotations.coco.json",
        CATEGORY_REMAP,
    )
    val_ds = TennisRacketDataset(
        DATA_DIR / "valid",
        DATA_DIR / "valid" / "_annotations.coco.json",
        CATEGORY_REMAP,
    )

    collate    = make_collate_fn(processor, split_name="train")
    val_loader = make_val_loader(val_ds, processor, batch_size=BATCH_SIZE)

    # AutoModelForObjectDetection.from_pretrained has no return annotation in transformers stubs;
    # cast to PreTrainedModel so downstream uses (model.parameters(), model.config, etc.) are typed.
    logger.info("Loading model from: %s", CHECKPOINT)
    model = cast(PreTrainedModel, AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("  num_labels  : %d  →  %s", model.config.num_labels, model.config.id2label)
    logger.info("  num_queries : %s", getattr(model.config, "num_queries", "?"))
    logger.info("  parameters  : %d trainable / %d total", trainable, total)

    # Sanity-check one forward pass before committing to a full training run.
    # This catches shape/label mismatches early (wrong num_labels, bad remap, etc.)
    logger.info("Sanity-check: one forward pass on CPU…")
    sample = collate([train_ds[0]])
    with torch.no_grad():
        out = model(**sample)
    logger.info("  loss        : %.4f", out.loss.item())
    logger.info("  logits      : %s  (batch, num_queries, num_classes+1)",
                tuple(out.logits.shape))
    logger.info("  pred_boxes  : %s  (batch, num_queries, 4 normalized cxcywh)",
                tuple(out.pred_boxes.shape))

    # Detect available device; Trainer will use it automatically.
    if _ON_MPS:
        logger.info("Device: MPS  (PYTORCH_ENABLE_MPS_FALLBACK=1 is set)")
    elif _ON_CUDA:
        logger.info("Device: CUDA  (%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("Device: CPU")
    logger.info(
        "Config — batch=%d × accum=%d (eff=%d)  epochs=%d  workers=%d  "
        "bf16=%s  fp16=%s  tf32=%s  compile=%s  ema=%s  output=%s",
        BATCH_SIZE, GRAD_ACCUM, BATCH_SIZE * GRAD_ACCUM, MAX_EPOCHS, NUM_WORKERS,
        BF16, FP16, _ON_CUDA, TORCH_COMPILE, USE_EMA, OUTPUT_DIR,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        save_strategy="epoch",
        eval_strategy="epoch",
        # We don't use load_best_model_at_end: the deliverable is the EMA-weighted
        # best_map/ checkpoint saved by MAPEvalCallback. (Trainer checkpoints hold
        # live weights and bind selection to a single metric tied to the file, which
        # can't express "EMA weights at best mAP".)
        load_best_model_at_end=False,
        remove_unused_columns=False,   # required: detection models use non-standard columns
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=PIN_MEMORY,
        fp16=FP16,
        bf16=BF16,
        torch_compile=TORCH_COMPILE,
        logging_steps=10,
        report_to="none",
    )

    class NaNSafeTrainer(Trainer):
        """Skips steps where fp16 overflow causes NaN in the decoder output.

        The GradScaler only inspects gradients; it cannot prevent a NaN that
        originates in the forward pass and surfaces as a ValueError in the loss.
        This wrapper catches that, logs it, and returns zero loss so the loop
        continues. The GradScaler will then reduce its scale and self-correct.
        """

        _nan_steps_skipped: int = 0

        def create_optimizer(self, model: nn.Module | None = None) -> torch.optim.Optimizer:
            """AdamW with a 10× lower LR for the pretrained backbone and no weight
            decay on biases / 1-D (norm) parameters. Cosine schedule + warmup are
            built by Trainer.create_scheduler on top of this optimizer.

            Accepts the optional `model` arg that newer transformers pass in; falls
            back to self.model for older versions that call it with no argument."""
            mdl = model if model is not None else self.model
            if self.optimizer is None:
                groups: list[dict[str, Any]] = []
                for is_backbone in (True, False):
                    for decay in (True, False):
                        params = [
                            p for n, p in mdl.named_parameters()
                            if p.requires_grad
                            and (("backbone" in n) == is_backbone)
                            and ((p.ndim >= 2 and not n.endswith(".bias")) == decay)
                        ]
                        if not params:
                            continue
                        groups.append({
                            "params": params,
                            "lr": BACKBONE_LR if is_backbone else self.args.learning_rate,
                            "weight_decay": self.args.weight_decay if decay else 0.0,
                        })
                n_bb = sum(
                    sum(p.numel() for p in g["params"])
                    for g in groups if g["lr"] == BACKBONE_LR
                )
                logger.info(
                    "Optimizer: %d groups; backbone=%d params @ lr=%.0e, head @ lr=%.0e",
                    len(groups), n_bb, BACKBONE_LR, self.args.learning_rate,
                )
                self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
            return self.optimizer

        def training_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            num_items_in_batch: torch.Tensor | int | None = None,
        ) -> torch.Tensor:
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            except (ValueError, RuntimeError) as exc:
                msg = str(exc)
                if "nan" in msg.lower() or "boxes1 must be" in msg or "boxes2 must be" in msg:
                    self._nan_steps_skipped += 1
                    logger.warning(
                        "NaN in decoder output at step %d (skipped so far: %d) — "
                        "fp16 overflow; GradScaler will reduce scale. %.120s",
                        self.state.global_step, self._nan_steps_skipped, msg,
                    )
                    return torch.tensor(
                        0.0, requires_grad=True, device=next(model.parameters()).device
                    )
                raise

    ema_callback = EMACallback(EMA_DECAY) if USE_EMA else None
    logger.info("Model EMA: %s", f"enabled (decay={EMA_DECAY})" if USE_EMA else "disabled")

    map_callback = MAPEvalCallback(
        val_loader=val_loader,
        processor=processor,
        patience=MAP_PATIENCE,
        min_epochs=MAP_MIN_EPOCHS,
        threshold=MAP_THRESHOLD,
        proc_H=proc_H,
        proc_W=proc_W,
        ema_cb=ema_callback,
    )
    callbacks: list[TrainerCallback] = [map_callback, SaveEveryNCallback(n=CHECKPOINT_EVERY_N)]
    if ema_callback is not None:
        callbacks.append(ema_callback)

    trainer = NaNSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        data_collator=collate,
        callbacks=callbacks,
    )

    # Auto-resume from the latest checkpoint if one exists in OUTPUT_DIR.
    resume_ckpt: str | None = None
    ckpts = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        resume_ckpt = str(ckpts[-1])
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
    else:
        logger.info("No checkpoint found — starting from scratch.")

    logger.info(
        "Training — max_epochs=%d  early_stop patience=%d (active after epoch %d)",
        MAX_EPOCHS, MAP_PATIENCE, MAP_MIN_EPOCHS,
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Finalize: deliverable = best EMA mAP checkpoint ───────────────────────
    # best_map/ holds the EMA-weighted weights at the best val mAP@[.5:.95]. Fall
    # back to the last-epoch model only if no best_map was ever saved (e.g. a very
    # short run whose eval never improved past the threshold).
    best_map_dir = OUTPUT_DIR / "best_map"
    final_dir    = OUTPUT_DIR / "final"
    if best_map_dir.exists():
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(best_map_dir, final_dir)
        logger.info("final/ ← best_map/  (best EMA mAP@[.5:.95] checkpoint)")
        # Reload the EMA-best deliverable via from_pretrained so transformers applies
        # its checkpoint↔runtime key mapping. RT-DETR's saved key names differ from the
        # module attribute names (e.g. out_proj/fc1/encoder.encoder on disk vs
        # o_proj/mlp.fc1/encoder.aifi in the model), so a raw load_state_dict mismatches.
        best = cast(PreTrainedModel, AutoModelForObjectDetection.from_pretrained(str(final_dir)))
        best.to(model.device)  # type: ignore  # torch .to() overload stub confuses pyright
        eval_model = best
    else:
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))
        logger.info("final/ ← last-epoch model (no best_map checkpoint was saved)")
        eval_model = model

    # ── Final test-set evaluation (unbiased; val drove early stopping) ────────
    test_res: dict[str, Any] | None = None
    test_ann = DATA_DIR / "test" / "_annotations.coco.json"
    if test_ann.exists():
        logger.info("Running final test-set evaluation…")
        test_ds = TennisRacketDataset(DATA_DIR / "test", test_ann, CATEGORY_REMAP)
        test_loader = make_val_loader(test_ds, processor, batch_size=BATCH_SIZE)
        test_res = run_map_eval(eval_model, test_loader, processor, proc_H, proc_W)
        logger.info(
            "Test  images=%d  mAP@[.5:.95]=%.4f  mAP@50=%.4f  mAP@75=%.4f",
            test_res["n_images"], test_res["map"], test_res["map_50"], test_res["map_75"],
        )
    else:
        logger.info("No test split at %s — skipping test evaluation.", test_ann)

    write_run_metadata(final_dir, train_ds, val_ds, map_callback.best_map, test_res)
    logger.info("Saved deliverable to %s", final_dir)


if __name__ == "__main__":
    main()
