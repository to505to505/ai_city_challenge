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

## 5. Repo layout

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

## 6. Working notes / gotchas

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

## 7. References

- AI City Challenge home: https://www.aicitychallenge.org/
- Hafnia platform: https://hafnia.milestonesys.com/
- Hafnia docs: https://hafnia.readme.io/docs/welcome-to-hafnia
- Hafnia on PyPI: https://pypi.org/project/hafnia/
- ECCV 2026: https://eccv.ecva.net/
