# RT-DETR tennis ball + racket detector

Fine-tunes [`PekingU/rtdetr_r18vd`](https://huggingface.co/PekingU/rtdetr_r18vd)
(HuggingFace Transformers) to detect two classes — **ball (0)** and **racket (1)** —
for a downstream tennis project. The trained deliverable lives in
[`models/final/`](models/final) and is versioned in git alongside a
`training_run.json` provenance file so every model traces back to its run.

This README is the full replication recipe for a **fresh Windows machine with an
RTX 5070 Ti** (Blackwell, 16 GB). A Google Colab Free (T4) fallback is at the end.

---

## 0. What you need

- **The code:** this repo.
- **The data:** two Roboflow COCO exports (not in git — they're large):
  - `ap-tennis.v2i.coco.zip` — ball + racket, single `train/` folder.
  - `Tennis_ball.coco.zip` — ball-only, already split, ball label fragmented
    across several category ids.

  Copy both zips into the repo folder after cloning. You can run with just one;
  `prepare_data.py` skips a missing source with a warning.
- **An NVIDIA driver** new enough for CUDA 12.8 (required for Blackwell / sm_120).
  Recent GeForce drivers (570+) are fine.

---

## 1. Get the code

```powershell
git clone https://github.com/tyevhen/rtdetr-finetune.git
cd rtdetr-finetune
```

## 2. Install uv

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reopen the terminal afterward so `uv` is on `PATH`.

## 3. Create the environment (Blackwell torch FIRST)

The locked dependencies pull a CPU build of torch. The 5070 Ti needs the CUDA 12.8
wheel, so install torch/torchvision from the cu128 index **before** the rest, and
work inside the activated venv (do **not** use `uv run`, which re-syncs from the
lockfile and would replace the CUDA torch with the CPU build).

```powershell
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1

# CUDA 12.8 build for Blackwell (sm_120) — the critical step:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# everything else:
uv pip install transformers accelerate torchmetrics faster-coco-eval pillow scipy safetensors
```

## 4. Verify the GPU before a long run

```powershell
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('cap', torch.cuda.get_device_capability(0)); print('bf16', torch.cuda.is_bf16_supported())"
```

Expect: `cuda True` · `NVIDIA GeForce RTX 5070 Ti` · `cap (12, 0)` · `bf16 True`.
A `no kernel image available` error means torch is too old for Blackwell —
reinstall from the cu128 index.

## 5. Prepare the data

With both `*.coco.zip` files sitting in the repo folder, just run `prepare_data.py`.
It auto-extracts each zip (only if not already extracted), auto-locates the COCO
folders, normalizes each dataset's (incompatible, colliding) category ids to
canonical `0=ball / 1=racket`, and writes a single merged split into `data/`.

```powershell
python prepare_data.py --clean      # extracts + merges into data\{train,valid,test}\
python dataset_stats.py             # sanity check
```

Expected merged totals: **9,935 images — ball 10,281 / racket 5,756**.
`dataset_stats.py` also prints the COCO small/medium/large box-size breakdown
(tennis balls are mostly tiny, which is why ball confidence is the hard part).

> To change the split ratios for single-folder sources:
> `python prepare_data.py --clean --val 0.10 --test 0.05 --seed 42`.
> Pre-split sources (Tennis_ball) keep their own train/valid/test partition.

## 6. Smoke test (1 epoch)

Confirms the whole pipeline on GPU end-to-end before committing hours.

```powershell
$env:MAX_EPOCHS=1; python train.py; Remove-Item Env:\MAX_EPOCHS
```

Watch for: `bf16=True`, the optimizer line showing `backbone @ lr=1e-05, head @
lr=1e-04`, `EMA initialized`, an mAP eval, and `final/` + `training_run.json` written.

## 7. Full training run

```powershell
python train.py
```

Auto-detected on this GPU: **bf16, batch 8 × grad-accum 2 (effective 16),
cosine + warmup, EMA, up to 72 epochs** with early stop on mAP@[.5:.95].
Outputs go to `rtdetr-tennis\`: periodic checkpoints, `best_map\`, and the
finalized `final\` (EMA-best weights) with `final\training_run.json`.

## 8. Evaluate

```powershell
python eval_testset.py      # mAP@[.5:.95] / @50 / @75 on data\test, saves eval_out\
python eval_visualize.py    # GT vs top predictions per image, saves eval_vis\
```

These load `models/final/` by default — point them at a fresh run by copying it in
(next step) first, or edit `MODEL_DIR`.

## 9. Version the trained model

When a run is good, promote it into `models/final/` and commit so the weights are
traceable to their run:

```powershell
Remove-Item -Recurse -Force models\final
Copy-Item -Recurse rtdetr-tennis\final models\final
git add models/final
git commit -m "Model: <note> (val/test mAP in training_run.json)"
git push
```

---

## Tuning knobs (environment variables)

All optional; sensible defaults are auto-detected from the hardware. Set in
PowerShell with `$env:NAME="value"` before `python train.py`.

| Variable        | Default (CUDA) | Purpose |
|-----------------|----------------|---------|
| `BATCH_SIZE`    | `8`            | Per-step batch. Lower if you hit OOM. |
| `GRAD_ACCUM`    | `2`            | Gradient accumulation (effective batch = `BATCH_SIZE × GRAD_ACCUM`, target 16). |
| `MAX_EPOCHS`    | `72`           | Upper bound; early stop usually fires first. |
| `NUM_WORKERS`   | `8`            | Dataloader workers. Drop to `4` if Windows worker spawn is slow. |
| `TORCH_COMPILE` | `0`            | `1` to try `torch.compile` (best-effort speedup). |
| `USE_EMA`       | `1`            | Exponential moving average of weights. |
| `MAP_PATIENCE`  | `20`           | Early-stop patience (evals without mAP improvement). |
| `MAP_MIN_EPOCHS`| `20`           | Warmup epochs before early stop can trigger. |
| `OUTPUT_DIR`    | `rtdetr-tennis`| Where checkpoints/`final/` are written. |

---

## Project layout

| Path | Role |
|------|------|
| `prepare_data.py` | Merge + normalize the source COCO exports into `data/` (canonical 0=ball/1=racket). |
| `dataset_stats.py` | Pre-training overview: per-split image/object counts + box-size breakdown. |
| `dataset.py` | `TennisRacketDataset`, collate, val loader. |
| `train.py` | Hardware-adaptive fine-tuning: EMA, backbone LR, cosine+warmup, mAP early stop. |
| `eval_testset.py` | mAP on `data/test` + annotated samples. |
| `eval_visualize.py` | GT vs predictions per image, sorted worst-first. |
| `models/final/` | Versioned trained model + `training_run.json` provenance. |
| `data/`, `export_*/`, `*.coco.zip` | Generated/large — gitignored. |

---

## Colab Free (T4) fallback

Same flow, with two differences: T4 has no bf16, so `train.py` auto-selects
**fp16 + a NaN-safe guard**, and standard torch already works (no cu128 index).

```python
!pip install -q transformers accelerate torchmetrics faster-coco-eval
# upload the two *.coco.zip exports into the repo folder, then (auto-extracts):
!python prepare_data.py --clean
!python dataset_stats.py
# persist outputs to Drive so they survive a disconnect:
import os; os.environ["OUTPUT_DIR"] = "/content/drive/MyDrive/rtdetr-finetune/rtdetr-tennis"
!python train.py
```
