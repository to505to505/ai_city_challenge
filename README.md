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

### Option A — public trainer package (no code)

For common tasks Hafnia ships ready-made trainers. To launch one:

1. Sign in to the Hafnia platform and open the experiments dashboard.
2. Click **Create Experiment**, pick your dataset (or data recipe).
3. Under **Trainer package**, open the **Public Trainers** tab and select
   one — e.g. *Object Detection Trainer* (wraps RF-DETR, compatible with
   COCO-style datasets).
4. Set the training command (e.g. `python scripts/train.py`) and the
   configuration tier (`Lite`, `Pro`, `Scale`).
5. **Create Experiment** and monitor progress in the dashboard.

The default Object Detection trainer converges in ~4 hours on
`midwest-vehicle-detection` at the `Lite` tier.

### Option B — bring your own trainer package

When you need a different model, custom loop, or your own metrics,
package your project as a trainer ZIP and launch it like a public one.

Reference templates (read these — they document structure, expected
entry points, and `HafniaLogger` integration):

- [`trainer-classification`](https://github.com/milestone-hafnia/trainer-classification) — image classification.
- [`trainer-object-detection`](https://github.com/milestone-hafnia/trainer-object-detection) — RF-DETR wrapper for detection (a sensible starting point for this challenge).

CLI workflow:

```bash
# Build trainer.zip from the current project
hafnia trainer create-zip .

# One-shot: package + upload + launch
hafnia trainer create .                                # upload only
hafnia experiment create \
    --dataset eccv-cross-city \
    --trainer-path . \
    --cmd "python scripts/train.py --epochs 50"        # package + upload + launch

# Manage existing trainers
hafnia trainer ls                                       # your trainers
hafnia trainer ls --visibility public                   # public trainers
hafnia trainer update <trainer-id> .                    # push a new version
hafnia trainer view-zip trainer.zip                     # inspect ZIP contents

# Launch against an already-uploaded trainer
hafnia experiment create --dataset eccv-cross-city --trainer-id <trainer-id>
```

Trainer packages are visible to your whole organization — saved trainers
can be reused by teammates.

### Test the build locally (Docker)

Before uploading, build the trainer in the same way the platform does
and run it against a sample dataset:

```bash
# Build the docker image from trainer.zip
hafnia runc build-local trainer.zip

# Run that image locally against a sample dataset
hafnia runc launch-local --dataset eccv-cross-city "python scripts/train.py"
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

## 6. Repo layout

```
.
├── README.md                          this file
└── visualization/
    ├── visualize.py                   reference script: loads the dataset,
    │                                  draws bboxes, plots class/split stats
    ├── examples.png                   9 random annotated samples
    ├── class_distribution.png         per-class bbox counts (log scale)
    └── splits.png                     samples + bboxes per split
```

`visualize.py` writes its outputs to `/home/dsa/hafnia/visualization/`
(hardcoded for the original author's environment) — if you re-run it
locally, fix those paths first.

## 7. Working notes / gotchas

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

## 8. References

- AI City Challenge home: https://www.aicitychallenge.org/
- Hafnia platform: https://hafnia.milestonesys.com/
- Hafnia docs: https://hafnia.readme.io/docs/welcome-to-hafnia
- API-key guide: https://hafnia.readme.io/docs/create-an-api-key
- Hafnia on PyPI: https://pypi.org/project/hafnia/
- Hafnia SDK/CLI source: https://github.com/milestone-hafnia/hafnia
- Reference trainer — classification: https://github.com/milestone-hafnia/trainer-classification
- Reference trainer — object detection: https://github.com/milestone-hafnia/trainer-object-detection
- Data library: https://hafnia.milestonesys.com/training-aas/datasets
- ECCV 2026: https://eccv.ecva.net/
