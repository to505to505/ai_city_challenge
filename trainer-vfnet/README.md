# trainer-convnext — Cascade R-CNN + ConvNeXt-Tiny (mmdet 3.x) for eccv-cross-city

A **separate** Hafnia Training-aaS trainer package, independent from the RF-DETR trainer in the
repo root. It fine-tunes the BDD100K-pretrained **Cascade R-CNN + ConvNeXt-Tiny (FPN)** detector on
`eccv-cross-city`.

## What it does

- **Model:** `CascadeRCNN` (3 IoU stages) + `FPN` neck + `mmpretrain.ConvNeXt arch='tiny'` backbone,
  rebuilt from the config embedded in the original BDD100K checkpoint and adapted to 10 eccv classes.
- **Init:** `load_from` the bundled, slimmed BDD100K detector — the **whole** detector
  (backbone + neck + RPN + cascade heads), not just an ImageNet backbone.
- **Backbone learns slower than neck+head:** AdamW `lr=1e-4` for neck/RPN/RoI-heads, and a
  `backbone` `lr_mult=0.1` → backbone effective `1e-5`. Tune with `--backbone-lr-mult`.
- **Input = resolution-preserving multiscale** (no tiling): `keep_ratio` resize capping the **long
  edge at 1920** with short-edge jitter `[896, 960, 1024, 1080]`, then pad to a multiple of **32**
  (at batch collation, in the data preprocessor). Handles the dataset's mixed sizes (1920×1080 plus
  3072×2048 / 3648×2052 / portrait) without squashing aspect ratio. Val/test: fixed `(1920, 1080)`.
- **Basic augmentations** (same intent as the RF-DETR trainer), all mmdet-native: horizontal flip
  (`p=0.5`), photometric jitter (`PhotoMetricDistortion` — brightness/contrast/saturation/hue) and a
  mild affine (scale 0.9–1.1, translate ±5%, rotate ±5°, via `RandomAffine`). No albumentations: the
  mmdet 3.3 Albu wrapper is incompatible with albumentations ≥1.4 (forwards `img_path`, which it rejects).
- **fp16** (AMP), leave-camera-out validation split (DG-honest), live metrics + best-checkpoint
  mirroring to the Hafnia dashboard.

## Layout

```
trainer-convnext/
├── Dockerfile                       # torch2.1/cu121 + mim(mmcv2.1/mmdet3.3/mmpretrain1.2)
├── requirements.txt                 # hafnia, numpy<2, albumentations, pycocotools (NO torch/mmX)
├── .hafniaignore
├── src/eccv_convnext_trainer/       # validator stub
├── scripts/
│   ├── train.py                     # entry point: Hafnia ↔ mmdet Runner bridge
│   └── prepare_weights.py           # slim the 875 MB checkpoint → bundled 146 MB (fp16)
├── configs/cascade_convnext_eccv.py # mmdet 3.x config (model + pipeline + schedule)
└── weights/cascade_convnext_bdd_slim.pth   # bundled init weights (generated)
```

## Build & run

```bash
# 0. (Re)generate the bundled weights from the repo-root checkpoint (already done — SHIPPED AS fp16).
#    fp16 (~146 MB) loads fine into the fp32 model; drop --fp16 for a fp32 copy (~292 MB).
python scripts/prepare_weights.py --fp16

# 1. Build the trainer.zip (run from inside trainer-convnext/ so its .hafniaignore applies).
cd trainer-convnext
hafnia trainer create-zip .
hafnia trainer view-zip trainer.zip        # sanity-check contents (weights/*_slim.pth must be in)

# 2. Local smoke test mirroring the platform (sample dataset, 1 epoch):
hafnia runc build-local trainer.zip
docker run --rm --gpus all \
    -e HAFNIA_CLOUD=true \
    -v $PWD/../.data/datasets/eccv-cross-city:/opt/ml/input/data/training:ro \
    <image>:latest python scripts/train.py --epochs 1

# 3. Upload via the dashboard (Training-aaS → Create Experiment → Upload new),
#    trainer name e.g.  eccv-cascade-convnext-tiny-v1
#    command:  python scripts/train.py --epochs 24
#    Scale (4×T4):  torchrun --nproc_per_node=4 scripts/train.py --epochs 24 --launcher pytorch
```

## VRAM / batch size

~1920 long-edge is heavy for a T4 (16 GB). Defaults: `--batch-size 2` (per GPU), fp16. If it OOMs,
drop to `--batch-size 1`. The val loader is already `batch_size=1`.

## ⚠️ Not yet build-tested

This package was authored against the embedded BDD config + the Hafnia SDK API, but the OpenMMLab
dependency matrix and the `load_from` key-mapping have **not** been verified inside a Docker build on
this machine (mmdet is not installed locally). Before a paid cloud run, do the **local smoke test**
in step 2 and confirm: (a) the image builds, (b) `load_from` reports near-zero missing/unexpected
keys, (c) one epoch trains + validates and metrics show up. See the repo-root `trainer_instruction.txt`
for the full Hafnia checklist.
