"""Fine-tune Cascade R-CNN + ConvNeXt-Tiny (mmdet 3.x) on eccv-cross-city via the Hafnia SDK.

Pipeline:
1. Pull the dataset (mounted under Training-aaS, sampled locally) with the Hafnia SDK.
2. Re-split leave-camera-out (DG-honest proxy for the hidden target city), then export to a
   Roboflow-style COCO layout on disk (cached; skipped if already present).
3. Build the mmdet Runner from configs/cascade_convnext_eccv.py, injecting the real COCO paths,
   the Hafnia checkpoint dir, the bundled BDD-pretrained weights, and the requested hyper-params.
4. Stream metrics + best checkpoints to the Hafnia dashboard LIVE using the SAME mechanism as the
   RF-DETR trainer: a tiny mmengine hook writes metrics.csv, and a background TrainStreamingWatcher
   tails it, re-binds the MLflow run inside its own thread, and forwards rows via log_scalar/log_metric.

Run locally (sample dataset, 1 GPU):
    python scripts/train.py --epochs 1

Run on Hafnia (see commands.txt / trainer_instruction.txt):
    Lite  (1 x T4):  python scripts/train.py --epochs 24
    Scale (4 x T4):  torchrun --nproc_per_node=4 scripts/train.py --epochs 24 --launcher pytorch
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.experiment import HafniaLogger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job

from mmengine.config import Config
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.runner import Runner

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "vfnet_eccv.py"
# Slimmed COCO-pretrained VFNet R-50 (state_dict only). Bundled — runtime is network-isolated.
BUNDLED_WEIGHTS = REPO_ROOT / "weights" / "vfnet_r50_coco_slim.pth"

# eccv-cross-city v1.0.0 classes, in dataset.info order (must match the config's `classes`).
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


# ============================ dataset → COCO ============================
def load_hafnia_dataset(dataset_name: str, version: str) -> HafniaDataset:
    """Load HafniaDataset from the mounted cloud path, or the local sample dataset."""
    if is_hafnia_cloud_job():
        mounted = get_dataset_path_in_hafnia_cloud()
        print(f"[data] HAFNIA_CLOUD=true — loading mounted dataset at {mounted}")
        return HafniaDataset.from_path(mounted)
    print(f"[data] local — HafniaDataset.from_name({dataset_name!r}, {version!r})")
    return HafniaDataset.from_name(dataset_name, version=version)


def reassign_splits_by_camera(dataset: HafniaDataset, val_fraction: float, seed: int) -> HafniaDataset:
    """Hold whole cameras out for validation (leave-camera-out).

    The native split shares cameras between train and validation, which measures same-camera
    generalization. The real test set is a hidden second city, so a camera the model never saw is
    a better proxy. Only labeled (train/validation) frames are re-split; native `test` is untouched.
    """
    df = dataset.samples
    if "camera_info" not in df.columns:
        print("[split] no camera_info column — keeping native splits")
        return dataset

    cam = df.select(pl.col("camera_info").struct.field("name")).to_series().to_list()
    orig = df["split"].to_list()
    labeled = {"train", "validation"}
    cams_labeled = sorted({c for c, s in zip(cam, orig) if s in labeled and c is not None})
    if not cams_labeled:
        print("[split] no labeled cameras found — keeping native splits")
        return dataset

    n_val = max(1, round(len(cams_labeled) * val_fraction))
    shuffled = list(cams_labeled)
    random.Random(seed).shuffle(shuffled)
    val_cams = set(shuffled[:n_val])

    new_split = [
        ("validation" if c in val_cams else "train") if s in labeled else s
        for c, s in zip(cam, orig)
    ]
    out = dataset.update_samples(df.with_columns(pl.Series(new_split).alias("split")))
    print(f"[split] leave-camera-out (seed={seed}): {n_val}/{len(cams_labeled)} cameras held out")
    print(f"[split] new split counts: {dict(Counter(new_split))}")
    return out


def export_to_coco(args) -> Dict[str, Dict[str, str]]:
    """Export the (re-split) Hafnia dataset to Roboflow-style COCO; return {split: {ann, img}} abs paths.

    Idempotent: if the export dir already has a train annotation file we reuse it, but we always
    re-derive the split→path mapping by scanning the directory so it works on a warm cache too.
    """
    suffix = (
        f"_locamout{args.val_camera_frac:g}_s{args.split_seed}"
        if args.val_camera_frac > 0
        else "_native"
    )
    out_dir = Path(args.coco_dir + suffix)

    roboflow = {"train": "train", "validation": "valid", "test": "test"}
    ann_name = "_annotations.coco.json"
    # Treat the cache as warm only if BOTH labeled splits we need exist — a half-written export from
    # an earlier crash must trigger a clean re-export, not a confusing 'missing validation' later.
    required = [out_dir / "train" / ann_name, out_dir / "valid" / ann_name]
    cache_warm = all(p.exists() for p in required)

    if not cache_warm:
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_hafnia_dataset(args.dataset_name, args.dataset_version)
        if args.val_camera_frac and args.val_camera_frac > 0:
            dataset = reassign_splits_by_camera(dataset, args.val_camera_frac, args.split_seed)
        else:
            print("[split] --val-camera-frac<=0 — using native splits")
        print(f"[data] exporting Roboflow COCO to {out_dir}")
        split_paths = dataset.to_coco_format(out_dir, coco_format_type="roboflow")
        splits = {
            sp.split: {"ann": str(sp.path_instances_json), "img": str(sp.path_images)}
            for sp in split_paths
        }
    else:
        print(f"[data] reusing cached COCO export at {out_dir}")
        splits = {}
        for hafnia_name, robo in roboflow.items():
            ann = out_dir / robo / ann_name
            if ann.exists():
                splits[hafnia_name] = {"ann": str(ann), "img": str(out_dir / robo)}

    print(f"[data] discovered splits: { {k: v['img'] for k, v in splits.items()} }")
    return splits


# ============================ metrics: SAME mechanism as the RF-DETR trainer ============================
# A mmengine hook writes <ckpt_dir>/metrics.csv (rank 0 only); the TrainStreamingWatcher below — ported
# verbatim from scripts/train.py (RF-DETR) — tails that CSV, re-binds the MLflow run inside its own
# thread (so HafniaLogger.log_scalar/log_metric land in the official run, not an orphan), and mirrors
# best checkpoints to path_model(). Column convention: train/* -> log_scalar, val/* -> log_metric.


class MetricsCsvHook(Hook):
    """Dump mmdet train/val scalars to <ckpt_dir>/metrics.csv for the watcher to tail (rank 0 only).

    Train rows carry train/<key> columns (loss, lr, ...); val rows carry val/<metric> columns
    (bbox_mAP, ...). The file is rewritten atomically on every update so a mid-write read can't see a
    torn CSV. This hook only PRODUCES the file — all HafniaLogger/MLflow forwarding is the watcher's job.
    """

    priority = "BELOW_NORMAL"

    def __init__(self, csv_path: Path, interval: int = 50) -> None:
        self.csv_path = Path(csv_path)
        self.interval = max(1, interval)
        self._rows: List[dict] = []
        self._fields: List[str] = ["step"]

    def _emit(self, row: dict) -> None:
        for k in row:
            if k not in self._fields:
                self._fields.append(k)
        self._rows.append(row)
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.csv_path.with_suffix(".tmp")
            with tmp.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fields)
                writer.writeheader()
                writer.writerows(self._rows)
            os.replace(tmp, self.csv_path)  # atomic — watcher always sees a complete file
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics-csv] write failed: {exc!r}")

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        if not is_main_process() or not self.every_n_train_iters(runner, self.interval):
            return
        row = {"step": runner.iter}
        for key, buf in runner.message_hub.log_scalars.items():
            if "coco" in key or key.startswith("val"):  # eval scalars are emitted by after_val_epoch
                continue
            try:
                value = float(buf.current())
            except Exception:  # noqa: BLE001
                continue
            if value != value:  # NaN guard
                continue
            row[f"train/{key}"] = value
        if len(row) > 1:
            self._emit(row)

    def after_val_epoch(self, runner, metrics: Optional[Dict] = None) -> None:
        if not is_main_process():
            return
        row = {"step": runner.iter}
        for key, value in (metrics or {}).items():
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            row[f"val/{key.split('/', 1)[-1]}"] = fval  # 'coco/bbox_mAP' -> 'val/bbox_mAP'
        if len(row) > 1:
            self._emit(row)


def _capture_mlflow_run_id() -> Optional[str]:
    """Return the run_id of the MLflow run HafniaLogger started, or None.

    MUST be called from the MAIN thread (where HafniaLogger called mlflow.start_run()). MLflow keeps
    the active-run stack per-thread, so the watcher thread re-binds this run_id (mlflow.start_run(
    run_id=...)) so the documented logger.log_scalar/log_metric API lands in the correct run.
    """
    try:
        import mlflow

        active = mlflow.active_run()
        return active.info.run_id if active is not None else None
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics] mlflow run capture failed ({exc!r}); local fallback")
        return None


class TrainStreamingWatcher:
    """Background thread that tails metrics.csv into HafniaLogger LIVE (ported from the RF-DETR trainer).

    `runner.train()` blocks for hours. Without live publishing the dashboard stays empty until the run
    ends, and a mid-run crash loses everything. So this watcher runs in parallel:
      * tails <ckpt_dir>/metrics.csv and forwards new rows via log_scalar/log_metric
      * copies the freshest best_coco*.pth from ckpt_dir to path_model() on mtime advance
    Both are idempotent (row cursor; mtime check), so the final flush in stop() is safe.
    """

    def __init__(
        self,
        hafnia_logger: HafniaLogger,
        ckpt_dir: Path,
        model_dir: Path,
        interval_seconds: float = 30.0,
    ) -> None:
        self.logger = hafnia_logger
        self.ckpt_dir = Path(ckpt_dir)
        self.model_dir = Path(model_dir)
        self.interval = interval_seconds
        self.csv_path = self.ckpt_dir / "metrics.csv"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rows_published = 0
        self._best_mtime = 0.0
        # Capture the platform's MLflow run_id in the MAIN thread; the watcher re-binds it in its own.
        self._mlflow_run_id = _capture_mlflow_run_id()

    def _publish_row(self, row: dict, step: int) -> int:
        """Push every numeric cell via log_metric (val/* eval series) / log_scalar (everything else)."""
        n = 0
        for key, cell in row.items():
            if key in ("step", "epoch") or cell in (None, ""):
                continue
            try:
                value = float(cell)
            except (TypeError, ValueError):
                continue
            is_eval = "/" in key and key.split("/", 1)[0] in {"val", "validation", "test"}
            fn = self.logger.log_metric if is_eval else self.logger.log_scalar
            try:
                fn(name=key, value=value, step=step)
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[watcher] skipped {key}={value} step={step}: {exc!r}")
        return n

    def _tick(self) -> None:
        # 1. metrics.csv -> official run
        if self.csv_path.exists():
            try:
                with self.csv_path.open() as fh:
                    rows = list(csv.DictReader(fh))
            except Exception as exc:  # noqa: BLE001
                print(f"[watcher] csv read error: {exc!r}")
                rows = []
            new = rows[self._rows_published :]
            pushed = 0
            for row in new:
                try:
                    step = int(float(row.get("step") or row.get("epoch") or 0))
                except (TypeError, ValueError):
                    step = 0
                pushed += self._publish_row(row, step)
            if new:
                self._rows_published = len(rows)
                print(f"[watcher] +{pushed} metrics from {len(new)} new rows (total {self._rows_published})")

        # 2. best_coco*.pth -> path_model()/best.pth
        try:
            best = sorted(self.ckpt_dir.glob("best_coco*.pth"), key=lambda p: p.stat().st_mtime)
            if best:
                newest = best[-1]
                mtime = newest.stat().st_mtime
                if mtime > self._best_mtime:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(newest, self.model_dir / "best.pth")
                    self._best_mtime = mtime
                    print(f"[watcher] mirrored {newest.name} -> {self.model_dir / 'best.pth'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[watcher] mirror error: {exc!r}")

    def _bind_mlflow_run_in_thread(self) -> None:
        if self._mlflow_run_id is None:
            return
        try:
            import mlflow

            mlflow.start_run(run_id=self._mlflow_run_id)
            print(f"[watcher] bound MLflow run {self._mlflow_run_id} in watcher thread")
        except Exception as exc:  # noqa: BLE001
            print(f"[watcher] could not bind MLflow run ({exc!r}); metrics may orphan")

    def _run(self) -> None:
        self._bind_mlflow_run_in_thread()
        try:
            self._tick()
            while not self._stop.wait(self.interval):
                try:
                    self._tick()
                except Exception as exc:  # noqa: BLE001
                    print(f"[watcher] tick error: {exc!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"[watcher] thread crashed: {exc!r}")

    def start(self) -> "TrainStreamingWatcher":
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="hafnia-stream", daemon=True)
        self._thread.start()
        print(f"[watcher] started (interval={self.interval}s) — tailing {self.csv_path}")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
        try:
            self._tick()  # final flush so the last rows/ckpt land
        except Exception as exc:  # noqa: BLE001
            print(f"[watcher] final flush error: {exc!r}")
        print(f"[watcher] stopped — rows published: {self._rows_published}")


# ============================ config assembly ============================
def build_cfg(args, splits: Dict[str, Dict[str, str]], work_dir: Path) -> Config:
    cfg = Config.fromfile(str(CONFIG_PATH))

    if "train" not in splits or "validation" not in splits:
        raise RuntimeError(
            f"need both 'train' and 'validation' splits in the COCO export, got {list(splits)}. "
            "Set --val-camera-frac>0 (default 0.2) so labeled cameras are held out for validation."
        )
    train_sp, val_sp = splits["train"], splits["validation"]

    # Point every dataset at the ABSOLUTE exported paths (data_root=None disables path joining).
    for dl, sp in ((cfg.train_dataloader, train_sp), (cfg.val_dataloader, val_sp)):
        dl.dataset.data_root = None
        dl.dataset.ann_file = sp["ann"]
        dl.dataset.data_prefix = dict(img=sp["img"] + "/")
        dl.dataset.metainfo = dict(classes=tuple(CLASS_NAMES))
    cfg.val_evaluator.ann_file = val_sp["ann"]

    # Hyper-params.
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_cfg.max_epochs = args.epochs
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.optim_wrapper.paramwise_cfg.custom_keys["backbone"]["lr_mult"] = args.backbone_lr_mult

    # LR schedule scaled to the requested epochs. Keep MultiStep milestones strictly inside
    # (0, epochs) and de-duplicated (they degenerate for tiny smoke runs), and shrink the linear
    # warmup so a 1-epoch run actually finishes warming up instead of sitting at start_factor*lr.
    cfg.param_scheduler[1].end = args.epochs
    ms = sorted({m for m in (args.epochs - 8, args.epochs - 2) if 1 <= m < args.epochs})
    cfg.param_scheduler[1].milestones = ms or [max(1, args.epochs - 1)]
    n_train = len(json.load(open(train_sp["ann"]))["images"])
    iters_per_epoch = max(1, math.ceil(n_train / args.batch_size))
    cfg.param_scheduler[0].end = min(500, max(50, iters_per_epoch * args.epochs // 2))

    # Runtime wiring.
    cfg.work_dir = str(work_dir)
    cfg.load_from = str(BUNDLED_WEIGHTS) if BUNDLED_WEIGHTS.exists() else None
    if cfg.load_from is None:
        if is_hafnia_cloud_job():
            raise FileNotFoundError(
                f"bundled weights missing at {BUNDLED_WEIGHTS}. The runtime is network-isolated, so "
                "the checkpoint must be baked into the image — run scripts/prepare_weights.py and rebuild."
            )
        print(f"[warn] !!! bundled weights missing at {BUNDLED_WEIGHTS} — training from RANDOM init. "
              "Run scripts/prepare_weights.py; a local smoke test without weights is NOT representative. !!!")
    cfg.launcher = args.launcher
    if args.num_workers is not None:
        cfg.train_dataloader.num_workers = args.num_workers
        cfg.val_dataloader.num_workers = args.num_workers
    # test == val by design; deepcopy so they are independent objects (no shared-reference footgun).
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=2, help="per-GPU; tuned for a single T4 16 GB")
    p.add_argument("--lr", type=float, default=1e-4, help="neck+head learning rate ('normal')")
    p.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="backbone lr = lr * this (default 0.1 -> backbone learns 10x slower)",
    )
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--launcher", default="none", choices=["none", "pytorch", "slurm", "mpi"])
    p.add_argument("--dataset-name", default="eccv-cross-city")
    p.add_argument("--dataset-version", default="1.0.0")
    p.add_argument("--coco-dir", default=str(REPO_ROOT / ".data" / "coco" / "eccv-cross-city"))
    p.add_argument(
        "--val-camera-frac",
        type=float,
        default=0.2,
        help="fraction of cameras held out ENTIRELY for validation (leave-camera-out). 0 = native splits.",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--scalar-interval", type=int, default=50, help="iters between metrics.csv rows")
    p.add_argument(
        "--stream-interval",
        type=float,
        default=30.0,
        help="seconds between watcher ticks that tail metrics.csv -> HafniaLogger and mirror best ckpts",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger = HafniaLogger(project_name="eccv-cross-city-vfnet")
    logger.log_configuration(
        {
            "model": "VFNet R-50 FPN (mmdet3)",
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr_mult": args.backbone_lr_mult,
            "input": "keep_ratio multiscale (long edge 1920) + pad/32",
            "augmentations": "h-flip, brightness/contrast, colorjitter, mild affine",
            "val_camera_frac": args.val_camera_frac,
            "split_seed": args.split_seed,
            "load_from": str(BUNDLED_WEIGHTS),
        }
    )

    splits = export_to_coco(args)
    work_dir = Path(logger.path_model_checkpoints())
    model_dir = Path(logger.path_model())

    cfg = build_cfg(args, splits, work_dir)
    print(f"[train] launcher={args.launcher} epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} backbone_lr_mult={args.backbone_lr_mult}")

    runner = Runner.from_cfg(cfg)
    # mmdet writes metrics.csv via the hook; the RF-DETR-style watcher tails it and forwards to
    # HafniaLogger + the official MLflow run, and mirrors best checkpoints to path_model().
    runner.register_hook(
        MetricsCsvHook(work_dir / "metrics.csv", interval=args.scalar_interval),
        priority="BELOW_NORMAL",
    )
    watcher = None
    if is_main_process():  # one watcher, rank 0 only (the CSV hook also writes on rank 0 only)
        watcher = TrainStreamingWatcher(
            logger, work_dir, model_dir, interval_seconds=args.stream_interval
        ).start()
    try:
        runner.train()
    finally:
        # Stop the watcher whether train() succeeded, raised, or got SIGTERM'd. stop() runs a final
        # tick so partial writes between the last interval and end-of-training still make it across.
        if watcher is not None:
            watcher.stop()

    print(f"[done] checkpoints in {work_dir}; trained model in {model_dir}")


if __name__ == "__main__":
    main()
