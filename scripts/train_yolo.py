"""Train YOLO11 on eccv-cross-city under the SAME conditions as the RF-DETR runs
(identical leave-camera-out split, same 10 classes, same held-out cameras) so the
cross-camera val mAP is directly comparable to RF-DETR's.

Pipeline:
1. Load HafniaDataset (full under TaaS, sample locally).
2. Re-split by camera (leave-camera-out, seed 42) — identical to scripts/train.py.
3. Export labeled splits to YOLO format (HafniaDataset.to_yolo_format) + write data.yaml.
4. Train ultralytics YOLO11 from COCO-pretrained weights (bundled, offline).
5. Forward val mAP to HafniaLogger each epoch; mirror best.pt to path_model().

Run on Hafnia:
  hafnia experiment create -d eccv-cross-city -p . -e Lite \
    -c "python scripts/train_yolo.py --epochs 100 --imgsz 1280 --model yolo11l"
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent

from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402
from hafnia.experiment import HafniaLogger  # noqa: E402
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job  # noqa: E402

CLASS_NAMES = [
    "Vehicle.Car", "Vehicle.Pickup Truck", "Vehicle.Single Truck", "Vehicle.Combo Truck",
    "Vehicle.Heavy Duty Vehicle", "Vehicle.Trailer", "Vehicle.Motorcycle", "Vehicle.Bicycle",
    "Vehicle.Van", "Person",
]
NUM_CLASSES = len(CLASS_NAMES)
# Bundled COCO-pretrained YOLO weights — the Hafnia cloud is network-isolated, so ultralytics
# cannot auto-download. Place yolo11<scale>.pt in ./weights/ (scripts/download_weights does not
# cover these; download once with `yolo` or curl and commit-bundle).
WEIGHTS_DIR = REPO_ROOT / "weights"


def load_hafnia_dataset(dataset_name: str, version: str) -> HafniaDataset:
    if is_hafnia_cloud_job():
        mounted = get_dataset_path_in_hafnia_cloud()
        print(f"[data] HAFNIA_CLOUD=true — loading from mounted path {mounted}")
        return HafniaDataset.from_path(mounted)
    print(f"[data] loading sample via from_name({dataset_name!r}, {version!r})")
    return HafniaDataset.from_name(dataset_name, version=version)


def reassign_splits_by_camera(dataset: HafniaDataset, val_fraction: float, seed: int) -> HafniaDataset:
    """IDENTICAL logic to scripts/train.py — hold out whole cameras for validation so the
    held-out set matches the RF-DETR runs exactly (same seed -> same 6 cameras)."""
    df = dataset.samples
    if "camera_info" not in df.columns:
        print("[split] no camera_info — keeping native splits")
        return dataset
    cam = df.select(pl.col("camera_info").struct.field("name")).to_series().to_list()
    orig = df["split"].to_list()
    labeled = {"train", "validation"}
    cams_labeled = sorted({c for c, s in zip(cam, orig) if s in labeled and c is not None})
    if not cams_labeled:
        return dataset
    n_val = max(1, round(len(cams_labeled) * val_fraction))
    shuffled = list(cams_labeled)
    random.Random(seed).shuffle(shuffled)
    val_cams = set(shuffled[:n_val])
    new_split = [("validation" if c in val_cams else "train") if s in labeled else s
                 for c, s in zip(cam, orig)]
    out = dataset.update_samples(df.with_columns(pl.Series(new_split).alias("split")))
    print(f"[split] leave-camera-out (seed={seed}): held out {n_val}/{len(cams_labeled)} cameras")
    print(f"[split] held-out cameras: {sorted(val_cams)}")
    print(f"[split] new split counts: {dict(Counter(new_split))}")
    return out


def export_yolo(dataset_name: str, version: str, out_dir: Path, val_frac: float, seed: int) -> Path:
    """Export train+validation to YOLO format. Idempotent. Returns the data.yaml path."""
    data_yaml = out_dir / "data.yaml"
    if data_yaml.exists():
        print(f"[data] reusing YOLO export at {out_dir}")
        return data_yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_hafnia_dataset(dataset_name, version)
    if val_frac and val_frac > 0:
        ds = reassign_splits_by_camera(ds, val_frac, seed)
    # Drop the unlabeled native `test` split so we don't copy ~15k label-less images.
    df = ds.samples
    ds = ds.update_samples(df.filter(pl.col("split").is_in(["train", "validation"])))
    print(f"[data] exporting YOLO format to {out_dir} ...")
    ds.to_yolo_format(out_dir)
    # ultralytics finds labels next to images (path has /data/, not /images/).
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"path: {out_dir}\n"
        f"train: train/data\n"
        f"val: validation/data\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"[data] wrote {data_yaml}")
    return data_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26l",
                   help="latest is yolo26* (Jan 2026, NMS-free + ProgLoss/STAL for small objects); "
                        "bundled as weights/<model>.pt for the offline platform")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO's high-res strength (cheap vs RF-DETR)")
    p.add_argument("--batch", type=int, default=-1, help="-1 = ultralytics auto-batch (fits ~60% VRAM)")
    p.add_argument("--patience", type=int, default=25, help="early-stop patience (epochs)")
    p.add_argument("--devices", default="0", help="'0' for 1 GPU (Lite); 'cpu' locally")
    p.add_argument("--dataset-name", default="eccv-cross-city")
    p.add_argument("--dataset-version", default="1.0.0")
    p.add_argument("--yolo-dir", default=str(REPO_ROOT / ".data" / "yolo" / "eccv-cross-city"))
    p.add_argument("--val-camera-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    logger = HafniaLogger(project_name="eccv-cross-city-yolo11")
    logger.log_configuration({
        "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch,
        "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
        "val_camera_frac": args.val_camera_frac, "split_seed": args.split_seed,
    })

    suffix = f"_locamout{args.val_camera_frac:g}_s{args.split_seed}"
    yolo_dir = Path(args.yolo_dir).parent / (Path(args.yolo_dir).name + suffix)
    data_yaml = export_yolo(args.dataset_name, args.dataset_version, yolo_dir,
                            args.val_camera_frac, args.split_seed)

    weights = WEIGHTS_DIR / f"{args.model}.pt"
    if not weights.exists():
        if is_hafnia_cloud_job():
            raise FileNotFoundError(
                f"{weights} missing. Hafnia cloud is offline — bundle {args.model}.pt in ./weights/.")
        print(f"[model] {weights} not bundled; ultralytics will fetch {args.model}.pt (local only)")
        weights = f"{args.model}.pt"
    print(f"[model] init: {weights}")

    ckpt_dir = Path(logger.path_model_checkpoints())
    model_dir = Path(logger.path_model())
    model_dir.mkdir(parents=True, exist_ok=True)

    # Forward YOLO's val metrics to HafniaLogger (main thread → lands in the official MLflow run),
    # and mirror best.pt to path_model() each epoch so a cancel still leaves the best checkpoint.
    _MAP = {"metrics/mAP50-95(B)": "val/mAP_50_95", "metrics/mAP50(B)": "val/mAP_50",
            "metrics/precision(B)": "val/precision", "metrics/recall(B)": "val/recall"}

    def on_fit_epoch_end(trainer):
        ep = int(getattr(trainer, "epoch", 0))
        for k, name in _MAP.items():
            v = (trainer.metrics or {}).get(k)
            if v is not None:
                try:
                    logger.log_metric(name=name, value=float(v), step=ep)
                except Exception as exc:  # noqa: BLE001
                    print(f"[metrics] skip {name}: {exc!r}")
        best = Path(trainer.save_dir) / "weights" / "best.pt"
        if best.exists():
            try:
                shutil.copy2(best, model_dir / "best.pt")
            except Exception as exc:  # noqa: BLE001
                print(f"[ckpt] copy best.pt failed: {exc!r}")

    run_name = args.model
    model = YOLO(str(weights))
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.devices, project=str(ckpt_dir), name=run_name, exist_ok=True,
        patience=args.patience, plots=False, verbose=True,
    )
    # Final copy of best.pt
    best = ckpt_dir / run_name / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, model_dir / "best.pt")
    print(f"[done] best.pt in {model_dir}")


if __name__ == "__main__":
    main()
