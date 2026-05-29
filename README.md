# ECCV 2026 AI City Challenge — Track 6: Cross-City Object Detection

## TL;DR for a new agent (or human) joining this repo

Train a fine-grained vehicle + person object detector on traffic imagery
from one set of cities, then make it generalize to **unseen target cities**
under domain shift (different camera mounts, viewpoints, weather, road
layouts, vehicle mixes). Submissions are scored on a hidden evaluation
server. Data is privacy-preserved and only accessible through the Hafnia
Training-as-a-Service platform.

The dataset is `eccv-cross-city` v1.0.0, loaded via the `hafnia` Python SDK.

---

## 1. The challenge

- **Event:** 10th AI City Challenge, workshop track at ECCV 2026
  (Malmö, Sweden, Sept 8–9, 2026).
- **Track:** #6 — Cross-City Object Detection.
- **Organizers:** Milestone Project Hafnia + Universidad Autónoma de Madrid.
- **Task:** Object detection of vehicles and persons in real-world traffic
  imagery, with the explicit goal of **robust geographic generalization** —
  train on source-city footage, evaluate on held-out target-city footage.
- **Why it matters:** Most traffic detectors overfit to the cities they were
  trained on (camera height, road furniture, vehicle distribution). Track 6
  measures how well models transfer.
- **Compliance angle:** All data is anonymized and ethically sourced;
  training runs on Hafnia's TaaS platform so raw video never has to leave.

## 2. Key dates (2026)

| Milestone                               | Date         |
| --------------------------------------- | ------------ |
| Train + validation data released        | April 20     |
| Test set + evaluation server opens      | May 18       |
| Registration opens (Hafnia community)   | May 18       |
| Submission deadline                     | July 10 (AoE) |
| Workshop (ECCV, Malmö)                  | Sept 8–9     |

## 3. Dataset: `eccv-cross-city` v1.0.0

Loaded with:

```python
from hafnia.dataset.hafnia_dataset import HafniaDataset
dataset = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
samples = dataset.samples  # polars DataFrame
```

### Sample variant (downloaded locally without paid access)
The local sample is a 300-image preview used for visualization and
pipeline debugging. The full dataset is only materialized when training
on the Hafnia platform.

| Split        | Images | Bboxes |
| ------------ | ------ | ------ |
| train        | 117    | 1,397  |
| validation   | 29     | 386    |
| test         | 154    | 0 (labels held out) |

### Classes (10) and sample-variant frequencies
Long-tail, heavily dominated by cars.

| Class                       | Count |
| --------------------------- | ----- |
| Vehicle.Car                 | 1,192 |
| Vehicle.Pickup Truck        | 262   |
| Person                      | 149   |
| Vehicle.Van                 | 78    |
| Vehicle.Single Truck        | 33    |
| Vehicle.Combo Truck         | 22    |
| Vehicle.Trailer             | 19    |
| Vehicle.Heavy Duty Vehicle  | 18    |
| Vehicle.Motorcycle          | 9     |
| Vehicle.Bicycle             | 1     |

Treat the bicycle / motorcycle / heavy-duty tail as a known weak point —
any submission that ignores class imbalance will tank rare-class AP.

### Per-sample schema
Each row in `dataset.samples` (a `polars.DataFrame`) contains at least:

- `file_path` — absolute path to the JPEG/PNG once downloaded.
- `split` — `train` | `validation` | `test`.
- `bboxes` — list of dicts, each with:
  - `top_left_x`, `top_left_y`, `width`, `height` — all **normalized to
    `[0, 1]`** by image width/height.
  - `class_name` — one of the 10 classes above.

See `visualization/visualize.py` for a working example that draws boxes,
plots class distribution, and breaks counts down per split.

## 4. Hafnia SDK essentials

```bash
pip install hafnia          # or: uv add hafnia
hafnia configure            # required for full datasets + uploads
```

What the SDK gives you:

- `HafniaDataset` — load by name, iterate samples, slice splits.
- `DatasetRecipe` — declarative dataset ops (shuffle, filter, merge) that
  serialize and re-run on the platform.
- `HafniaLogger` — experiment tracking (models, checkpoints, metrics).
- Torch helpers — `torchvision` dataloaders and augmentation pipelines.
- Benchmarking — runs inference + computes detection mAP.
- Importers for YOLO and COCO if you bring your own data.

Without `hafnia configure`, `HafniaDataset.from_name(...)` falls back to
the local sample variant if one is cached; otherwise it errors with
`No active profile configured`.

### CLI command surface

Each group accepts `--help` to list its subcommands and options.

```text
hafnia configure                       Interactive first-time setup (profile name, API key, URL)
hafnia clear                           Remove all stored configuration

hafnia profile     ls | active | use | rm | create               Manage local profiles
hafnia dataset     ls | download | delete                        Manage datasets on the platform
hafnia recipe      ls | create | rm                              Manage dataset recipes on the platform
hafnia trainer     ls | create | update | create-zip | view-zip  Manage trainer packages
hafnia experiment  ls | create | environments                    Launch and inspect experiments
hafnia runc        build | build-local | launch-local            Build and run trainer packages locally
```

## 5. Training on Hafnia: Trainer Packages

### Training-as-a-Service (Training-aaS) concept

Training-aaS lets you train models on **hidden datasets** — datasets that
are usable for training but never available for direct download. Privacy
and licensing are preserved by running *your code* against the data inside
the platform, not the other way around.

Two artifacts make this work:

- **Sample dataset** — a small, anonymized subset of every hidden dataset,
  downloadable locally (this is what `eccv-cross-city` gives you when you
  pull it on your laptop). Use it for development, debugging, and
  visualization.
- **Trainer package** — a ZIP of your training project (code +
  `Dockerfile` + training command) that the platform builds and runs
  against the full dataset.

The same `HafniaDataset.from_name("...")` call returns the sample dataset
locally and the full dataset under Training-aaS, so the training script
itself does not change between environments.

### Bring your own trainer package

We do not use the public no-code trainers for this challenge — every run
goes through a custom trainer package we control end-to-end.

> **Versioning rule — every change is a new trainer.**
> Whenever you modify the trainer (new model, new loss, new augmentations,
> dependency bump, hyperparameter sweep, etc.) upload a **new** trainer
> package whose name reflects what changed. Do not reuse the same trainer
> name for behavioural changes — that breaks reproducibility of past
> experiments.
>
> Naming convention: `<task>-<arch>-<change>-vN`, e.g.
> `eccv-rfdetr-large-base-v1`, `eccv-rfdetr-large-focal-loss-v2`,
> `eccv-rfdetr-large-mosaic-aug-v3`.

Reference templates (read these — they document structure, expected
entry points, and `HafniaLogger` integration):

- [`trainer-classification`](https://github.com/milestone-hafnia/trainer-classification) — image classification.
- [`trainer-object-detection`](https://github.com/milestone-hafnia/trainer-object-detection) — RF-DETR wrapper for detection (a sensible starting point for this challenge).

### Workflow — build locally, upload via the web UI

The flow we use is: build `trainer.zip` on this machine, then upload it
through the Hafnia dashboard.

**1. Build the zip locally**

From the repo root (the folder with the `Dockerfile`):

```bash
hafnia trainer create-zip .
hafnia trainer view-zip trainer.zip   # optional sanity check
```

This walks the tree, applies the patterns in [`.hafniaignore`](.hafniaignore)
(gitignore syntax), and writes `trainer.zip` next to the source.

**2. Upload via the web dashboard**

1. Sign in to <https://hafnia.milestonesys.com/>.
2. Open **Training-aaS → Experiments** in the left menu (or go directly
   to <https://hafnia.milestonesys.com/dashboard/training-aas/experiments>).
3. Click **Create Experiment**.
4. **Data source:** *Dataset* → `eccv-cross-city` v `1.0.0`.
5. **Trainer package:** choose **Upload new**, browse to
   `trainer.zip`, and give it a descriptive name following the versioning
   rule above (e.g. `eccv-rfdetr-large-base-v1`). Wait for upload to finish.
6. **Training command:** `python scripts/train.py --epochs 50`
7. **Environment / Tier:** `Lite` (1 × T4).
8. **Experiment name:** something concrete, e.g. `rfdetr_large_base`.
9. Click **Create Experiment**.

The platform builds the Docker image from your zip (~5 min), then runs
the training command. Metrics show up in the dashboard via
`HafniaLogger`/MLflow. Saved trainer packages can be re-used for future
experiments from the **Or select existing trainer package** option.

### Test the build locally (Docker, optional)

Before uploading, build and run the trainer in the same way the platform
does — useful for catching `Dockerfile` errors and missing deps before
the platform spends time + credits on them:

```bash
# Build the docker image from trainer.zip
hafnia runc build-local trainer.zip

# Run that image locally against the sample dataset
hafnia runc launch-local --dataset eccv-cross-city "python scripts/train.py --epochs 1"
```

This catches syntax errors, missing dependencies, and `Dockerfile`
problems before they fail in the cloud. (VS Code's *Model Training*
debug configuration is also wired up for this.)

### Experiment tracking with `HafniaLogger`

`HafniaLogger` collects the model artifact, checkpoints, hyperparameters,
and training/eval metrics. Locally it writes to
`.data/experiments/{DATE_TIME}/`; under Training-aaS the same data is
streamed into the platform's experiments view.

```python
from hafnia.experiment import HafniaLogger

logger = HafniaLogger(project_name="eccv-cross-city-detector")
logger.log_configuration({"batch_size": 16, "lr": 1e-4, "model": "RFDETRNano"})

ckpt_dir  = logger.path_model_checkpoints()  # save intermediate checkpoints here
model_dir = logger.path_model()              # save the final model here

logger.log_scalar("train/loss", value=0.42, step=100)
logger.log_metric("validation/mAP", value=0.31, step=100)
```

### Composing data with `DatasetRecipe`

A recipe is a serializable spec of "load this dataset, then do these
operations" (shuffle, select, merge, filter, drop classes, split). It
runs locally and uploads to the platform unchanged, so the same training
data definition is used everywhere.

```python
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe

recipe = (
    DatasetRecipe.from_name("eccv-cross-city", version="1.0.0")
        .shuffle()
        .select_samples(n_samples=200)
)
dataset = recipe.build()
recipe.as_platform_recipe(recipe_name="eccv-200-sample", overwrite=True)
```

`hafnia recipe ls | create | rm` manages recipes on the platform.

## 6. RF-DETR Large trainer (this repo)

This repository is laid out as a **Hafnia trainer package** that fine-tunes
**RF-DETR Large** on `eccv-cross-city`. RF-DETR is vendored as a plain folder
(`rf-detr/`, no `.git`) so we can patch it without dealing with submodules.

### Files that matter

- [`scripts/train.py`](scripts/train.py) — entry point. Pulls the dataset via
  `HafniaDataset.from_name`, exports it once to a Roboflow-style COCO layout
  in `.data/coco/eccv-cross-city/`, then runs `RFDETRLarge.train(...)`.
- [`configs/rfdetr_large_eccv.yaml`](configs/rfdetr_large_eccv.yaml) — reference
  config mirroring the same values; usable via `rfdetr fit --config ...` once
  the COCO data has been materialized.
- [`Dockerfile`](Dockerfile) — `pytorch/pytorch:2.5.1-cuda12.1-cudnn9` base,
  installs `rf-detr[train]` from the local folder + `requirements.txt`.
- [`requirements.txt`](requirements.txt) — Hafnia SDK + RF-DETR's runtime deps.
- [`.hafniaignore`](.hafniaignore) — keeps `visualization/`, docs, notebooks,
  and `.data/` out of `trainer.zip`.
- [`rf-detr/`](rf-detr/) — vendored upstream source (`roboflow/rf-detr`).

### Canonical RF-DETR Large settings

| Setting             | Value | Why                                                        |
| ------------------- | ----- | ---------------------------------------------------------- |
| `num_classes`       | 10    | classes in `eccv-cross-city` (see §3)                      |
| `resolution`        | 704   | RF-DETR Large default; must be divisible by `patch*windows = 32` |
| `patch_size`        | 16    | architectural — do not change                              |
| `num_windows`       | 2     | architectural — do not change                              |
| `pretrain_weights`  | `rf-detr-large-2026.pth` | bundled in `weights/` via `scripts/download_weights.py` (offline platform can't fetch at runtime) |
| `dataset_file`      | `roboflow` | matches the layout `to_coco_format` writes             |
| `batch_size × grad_accum` | 8 × 1 = 8 | empirically measured ~12-13 GB on T4 (probe + real-train) — fits with ~3 GB headroom |
| `multi_scale` / `expanded_scales` | `false` | off for Lite (would push activations past 704 on T4); flip to `true` on Scale |

### Logging

Three independent layers, all written into the same run:

| Layer            | Where                                            | How to enable                                     |
| ---------------- | ------------------------------------------------ | ------------------------------------------------- |
| `HafniaLogger`   | Hafnia experiments dashboard                     | Always on (created in `scripts/train.py`)         |
| TensorBoard      | `events.out.tfevents.*` inside the checkpoint dir | `tensorboard=True` (default)                      |
| W&B (optional)   | `wandb.ai/<your-team>/eccv-cross-city`           | Add `--wandb --run-name <name>` to the train cmd  |

For W&B in the cloud, the container needs `WANDB_API_KEY` — pass it inline in the
launch command (see [commands.txt](commands.txt)). RF-DETR's W&B logger is the
PyTorch Lightning `WandbLogger`; HafniaLogger keeps working in parallel.

### Augmentations

Defined in [`scripts/train.py:AUG_CONFIG`](scripts/train.py). Conservative
on geometry (mounted-camera footage, ground bias) and lean on photometric jitter:

- `HorizontalFlip` — `p=0.5`
- `RandomBrightnessContrast` — `±0.15`, `p=0.4`
- `ColorJitter` — `±0.15` brightness/contrast/saturation, `±0.05` hue, `p=0.4`
- `Affine` — scale `0.9–1.1`, ±5° rotation, ±5% translation, `p=0.3`

No `VerticalFlip`, no heavy rotation — they'd break "vehicles sit on the road" priors.
`multi_scale=True` and `expanded_scales=True` give native multi-resolution
training on top of these.

### How to run

```bash
# 0. Populate ./weights (gitignored, so a fresh clone has none). The offline
#    platform can't download at runtime, so the Dockerfile bundles weights/.
python scripts/download_weights.py        # idempotent, MD5-verified

# Local dry-run (uses the 300-image sample dataset, single T4)
python scripts/train.py --epochs 3

# Build the trainer.zip for upload
hafnia trainer create-zip .
```

Then upload `trainer.zip` via the web dashboard (see §5 *Workflow — build
locally, upload via the web UI*). On the platform, set the training
command to `python scripts/train.py --epochs 50` and pick the `Lite`
environment.

Remember the versioning rule from §5: **every meaningful change → new trainer
with a name that reflects what changed** (e.g. `eccv-rfdetr-large-aug-aerial-v1`).

## 7. Repo layout

```
.
├── README.md                          this file
├── commands.txt                       hafnia CLI cheatsheet
├── Dockerfile                         trainer-package container
├── requirements.txt                   Hafnia SDK + RF-DETR runtime deps
├── .hafniaignore                      excludes from trainer.zip
├── configs/
│   └── rfdetr_large_eccv.yaml         reference RF-DETR Large config
├── scripts/
│   └── train.py                       training entry point (called by Dockerfile)
├── src/
│   └── eccv_trainer/__init__.py       project package placeholder
├── rf-detr/                           vendored roboflow/rf-detr source
└── visualization/
    ├── visualize.py                   loads dataset, draws bboxes, plots stats
    ├── examples.png                   9 random annotated samples
    ├── class_distribution.png         per-class bbox counts (log scale)
    └── splits.png                     samples + bboxes per split
```

`visualize.py` writes its outputs to `visualization/` (absolute paths) — if
you re-run it elsewhere, fix those paths first.

## 8. Working notes / gotchas

- **Bboxes are normalized.** Multiply by image `W`, `H` before drawing or
  feeding into detectors that expect absolute coordinates.
- **Test split has zero bboxes** in the public release — ground truth lives
  on the eval server, so don't try to compute metrics on `split == "test"`
  locally.
- **Domain shift is the whole point.** Don't pick a validation set drawn
  from the same city distribution as train if you want a number that
  predicts test-set performance. Hold out at least one city.
- **Class imbalance is severe.** Plan for class-balanced sampling, focal
  loss, or per-class AP analysis from day one.
- **Don't push raw imagery anywhere.** The licensing is privacy-bounded;
  the Hafnia TaaS flow is the supported way to train at scale.

## 9. References

- AI City Challenge home: https://www.aicitychallenge.org/
- Hafnia platform: https://hafnia.milestonesys.com/
- Hafnia docs: https://hafnia.readme.io/docs/welcome-to-hafnia
- API-key guide: https://hafnia.readme.io/docs/create-an-api-key
- Hafnia on PyPI: https://pypi.org/project/hafnia/
- Hafnia SDK/CLI source: https://github.com/milestone-hafnia/hafnia
- Reference trainer — classification: https://github.com/milestone-hafnia/trainer-classification
- Reference trainer — object detection: https://github.com/milestone-hafnia/trainer-object-detection
- Data library: https://hafnia.milestonesys.com/training-aas/datasets
- RF-DETR: https://github.com/roboflow/rf-detr — docs: https://rfdetr.roboflow.com
- ECCV 2026: https://eccv.ecva.net/
