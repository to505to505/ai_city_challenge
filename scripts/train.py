"""Fine-tune RF-DETR Large on eccv-cross-city via the Hafnia SDK.

Pipeline:
1. Pull the dataset with HafniaDataset.from_name (sample locally, full under Training-aaS).
2. Export it to a Roboflow-style COCO layout (cached on disk; skipped if already present).
3. Build RFDETRLarge at its native 704x704 resolution with num_classes=10.
4. Run `.train()` with basic augmentations suited to a small in-the-wild traffic dataset.

Run locally:
    python scripts/train.py --epochs 5 --devices 1

Run on Hafnia (one of these, see commands.txt):
    hafnia experiment create -d eccv-cross-city -p . -e Lite  -c "python scripts/train.py --epochs 50"
    hafnia experiment create -d eccv-cross-city -p . -e Scale -c "python scripts/train.py --epochs 50 --devices 4"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# rf-detr lives in the repo as a plain folder (not pip-installed) so we expose its source.
sys.path.insert(0, str(REPO_ROOT / "rf-detr" / "src"))

from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402
from hafnia.experiment import HafniaLogger  # noqa: E402
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job  # noqa: E402

from rfdetr import RFDETRLarge  # noqa: E402

# Hafnia Training-aaS containers are network-isolated (only the platform's MLflow VPC endpoint
# is reachable). RF-DETR's default flow downloads pretrain weights from storage.googleapis.com,
# which is blocked. We bundle the .pth into the trainer.zip and point RFDETRLarge to it.
_BUNDLED_PRETRAIN = REPO_ROOT / "weights" / "rf-detr-large-2026.pth"

# 10 classes in eccv-cross-city v1.0.0, in the order returned by `dataset.info`.
CLASS_NAMES = [
    "Vehicle.Car",
    "Vehicle.Pickup Truck",
    "Vehicle.Single Truck",
    "Vehicle.Combo Truck",
    "Vehicle.Heavy Duty Vehicle",
    "Vehicle.Trailer",
    "Vehicle.Motorcycle",
    "Vehicle.Bicycle",
    "Vehicle.Van",
    "Person",
]
NUM_CLASSES = len(CLASS_NAMES)

# RF-DETR Large defaults (rfdetr.config.RFDETRLargeConfig): patch_size=16, num_windows=2.
# resolution must be divisible by patch_size * num_windows = 32. 704 / 32 = 22 ✓.
RESOLUTION = 704

# Basic augmentations — dataset is small (~117 train images) and traffic imagery is mounted-camera,
# so we keep geometry mild (no vertical flip, small rotations) and lean on photometric jitter.
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
    "ColorJitter": {
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.15,
        "hue": 0.05,
        "p": 0.4,
    },
    "Affine": {
        "scale": (0.9, 1.1),
        "translate_percent": (-0.05, 0.05),
        "rotate": (-5, 5),
        "p": 0.3,
    },
}


def load_hafnia_dataset(dataset_name: str, version: str) -> HafniaDataset:
    """Load HafniaDataset.

    On Hafnia Training-aaS the platform mounts the full hidden dataset into the container
    at $MDI_DATASET_DIR (default /opt/ml/input/data/training) and sets HAFNIA_CLOUD=true.
    Locally we fall back to `from_name`, which downloads the sample dataset (needs `hafnia configure`).
    """
    if is_hafnia_cloud_job():
        mounted = get_dataset_path_in_hafnia_cloud()
        print(f"[data] HAFNIA_CLOUD=true — loading from mounted path {mounted}")
        return HafniaDataset.from_path(mounted)
    print(f"[data] loading sample dataset via HafniaDataset.from_name({dataset_name!r}, {version!r})")
    return HafniaDataset.from_name(dataset_name, version=version)


def export_hafnia_to_coco(dataset_name: str, version: str, out_dir: Path) -> Path:
    """Export HafniaDataset to Roboflow-style COCO on disk, idempotent."""
    sentinel = out_dir / "train" / "_annotations.coco.json"
    if sentinel.exists():
        print(f"[data] reusing cached COCO dataset at {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_hafnia_dataset(dataset_name, version)
    print(f"[data] exporting to roboflow COCO layout at {out_dir}")
    dataset.to_coco_format(out_dir, coco_format_type="roboflow")
    return out_dir


def publish_metrics_to_hafnia(hafnia_logger: HafniaLogger, output_dir: Path) -> None:
    """Replay RF-DETR's metrics.csv into HafniaLogger.

    Lightning's CSVLogger writes one row per logging event with columns like
    `train/loss`, `val/map`, `val/per_class/Vehicle.Car/AP`, plus `step` and `epoch`.
    We forward every numeric cell to HafniaLogger so the platform's MLflow run gets
    populated with the same series — no duplicate run, no live forwarding.
    """
    import csv

    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"[metrics] no metrics.csv at {csv_path} — nothing to publish")
        return
    published = 0
    rows = 0
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            try:
                step = int(float(row.get("step") or row.get("epoch") or 0))
            except (TypeError, ValueError):
                step = 0
            for key, cell in row.items():
                if key in ("step", "epoch") or cell in (None, ""):
                    continue
                try:
                    value = float(cell)
                except (TypeError, ValueError):
                    continue
                # log_metric is for "evaluation" metrics; log_scalar for everything else.
                # Both end up in the platform's UI, the distinction is just label-grouping.
                fn = hafnia_logger.log_metric if "/" in key and key.split("/", 1)[0] in {"val", "validation", "test"} else hafnia_logger.log_scalar
                try:
                    fn(name=key, value=value, step=step)
                    published += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[metrics] skipped {key}={value} step={step}: {exc!r}")
                    break
    print(f"[metrics] published {published} datapoints from {rows} rows of {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    # Tuned for a single T4 (16 GB VRAM): bs=8, no grad accumulation.
    # VRAM @ bs=8 / 704 / fp32: ~12-13 GB peak (probe + real-train overhead).
    # If you see OOM on the platform, drop to --batch-size 4 --grad-accum-steps 2 (same effective batch).
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1, help="effective batch = batch_size * grad_accum_steps * devices")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--devices", type=int, default=1, help="1 for Lite (1 × T4), 4 for Scale (4 × T4)")
    p.add_argument("--resolution", type=int, default=RESOLUTION, help="must be divisible by 32 for RF-DETR Large")
    p.add_argument("--dataset-name", default="eccv-cross-city")
    p.add_argument("--dataset-version", default="1.0.0")
    p.add_argument(
        "--coco-dir",
        default=str(REPO_ROOT / ".data" / "coco" / "eccv-cross-city"),
        help="where to materialize the COCO copy of the dataset",
    )
    # NOTE: Hafnia Training-aaS containers are network-isolated — wandb.ai is unreachable.
    # Keep --wandb only for local runs. HafniaLogger writes via the platform's VPC MLflow.
    p.add_argument("--wandb", action="store_true", help="local only — W&B is blocked in Hafnia cloud")
    p.add_argument("--wandb-project", default="eccv-cross-city")
    p.add_argument("--run-name", default=None, help="W&B run name (also used as the experiment name)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger = HafniaLogger(project_name="eccv-cross-city-rfdetr-large")
    logger.log_configuration(
        {
            "model": "RFDETRLarge",
            "resolution": args.resolution,
            "patch_size": 16,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "lr_encoder": args.lr_encoder,
            "devices": args.devices,
            "augmentations": AUG_CONFIG,
        }
    )

    coco_dir = export_hafnia_to_coco(args.dataset_name, args.dataset_version, Path(args.coco_dir))

    pretrain_kwargs = {}
    if _BUNDLED_PRETRAIN.exists():
        print(f"[model] using bundled pretrain weights {_BUNDLED_PRETRAIN}")
        pretrain_kwargs["pretrain_weights"] = str(_BUNDLED_PRETRAIN)
    elif is_hafnia_cloud_job():
        raise FileNotFoundError(
            f"Pretrain weights not bundled at {_BUNDLED_PRETRAIN}. Hafnia cloud blocks "
            "outbound network, so RF-DETR cannot download them at runtime. Place "
            "rf-detr-large-2026.pth in ./weights/ and rebuild the trainer."
        )

    model = RFDETRLarge(num_classes=NUM_CLASSES, resolution=args.resolution, **pretrain_kwargs)

    # We intentionally do NOT pass `mlflow=True` to RF-DETR. RF-DETR would spin up its own PTL
    # MLflowLogger and create a SECOND mlflow run alongside the one HafniaLogger already started
    # (the platform's UI then shows two sibling runs in the experiment). Instead, we let RF-DETR's
    # CSVLogger write `metrics.csv` to output_dir and replay it into HafniaLogger after training.
    model.train(
        dataset_dir=str(coco_dir),
        dataset_file="roboflow",
        output_dir=str(logger.path_model_checkpoints()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        num_workers=args.num_workers,
        devices=args.devices,
        class_names=CLASS_NAMES,
        aug_config=AUG_CONFIG,
        # multi_scale + expanded_scales make per-batch resolution vary up to 768 — too risky
        # on a single 16 GB T4 at base resolution 704. Disable both for Lite; flip to True on Scale.
        multi_scale=False,
        expanded_scales=False,
        square_resize_div_64=True,
        use_ema=True,
        tensorboard=True,
        wandb=args.wandb,
        project=args.wandb_project,
        run=args.run_name,
    )

    # Replay PTL CSVLogger metrics into HafniaLogger so they land in the platform's MLflow run.
    publish_metrics_to_hafnia(logger, Path(logger.path_model_checkpoints()))

    # Copy the best checkpoint into HafniaLogger's model dir so the platform surfaces it as the
    # "Trained model" artifact (the Hafnia dashboard reads /opt/ml/model, not /opt/ml/checkpoints).
    ckpt_dir = Path(logger.path_model_checkpoints())
    model_dir = Path(logger.path_model())
    model_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        src = ckpt_dir / name
        if src.exists():
            dst = model_dir / name
            shutil.copy2(src, dst)
            saved.append(dst.name)
    if saved:
        print(f"[done] published model checkpoints to {model_dir}: {saved}")
    else:
        print(f"[warn] no best-checkpoint files found in {ckpt_dir}; nothing copied to model dir")
    print(f"[done] all checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
