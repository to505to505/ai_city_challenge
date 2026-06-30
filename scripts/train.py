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
import csv
import random
import shutil
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import polars as pl
import torch

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

# Phase-2 domain-generalization preset: stronger PHOTOMETRIC + sensor-noise variation to
# simulate a DIFFERENT camera / lighting (the hidden target city), pushing the model toward
# shape/structure cues over source-city appearance. Geometry stays mild (cameras are fixed,
# mounted) — no vertical flip / big rotations. Only transforms that appear in RF-DETR's own
# presets are used, so the param names are proven and it works under BOTH the Albumentations
# (default) and Kornia/GPU backends (no risk of an unsupported-key ValueError).
AUG_DG = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.4, "contrast_limit": 0.4, "p": 0.7},
    "ColorJitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.4, "hue": 0.1, "p": 0.6},
    "GaussNoise": {"std_range": (0.01, 0.08), "p": 0.4},
    "GaussianBlur": {"blur_limit": 3, "p": 0.25},
    "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.05, 0.05), "rotate": (-5, 5), "p": 0.3},
}

# Phase-3 preset: simulate the two camera types that hurt cross-camera most (diagnosed this run).
#   * FISHEYE (primary, GEOMETRIC): OpticalDistortion(mode="fisheye") warps the image like the
#     GRANDVIEW fisheye cam — the camera type higher-res alone (v7) only partly fixed (0.378->0.433).
#     RF-DETR routes OpticalDistortion through a bbox-aware Compose, so boxes are warped+clipped with
#     the image (verified). This is the genuinely NEW lever vs the photometric DG preset.
#   * NIGHT (secondary, PHOTOMETRIC): darken + IR-grayscale + sensor noise + cool tint to mimic
#     night / IR cameras. HONEST CAVEAT: photometric domain-randomization (the DG preset) did NOT
#     move cross-camera mAP for us (v6 ~= v5), so night is the weaker side-bet; fisheye should drive
#     any gain. All names/params are real Albumentations 2.x transforms (misconfigured ones are
#     silently skipped by from_config, so this preset is verified to build 8/8 before launch).
AUG_FISHEYE_NIGHT = {
    "HorizontalFlip": {"p": 0.5},
    # --- lens distortion: MILD + FRAME-FILLING to match the real cams. We verified (scripts/
    # real_fisheye_check.py) that this dataset's "fisheye" cams (GRANDVIEW) are actually high-mounted
    # WIDE-ANGLE that FILL the frame with only slight barrel — NOT strong fisheye, and with NO black
    # corners. So: small distort_limit + border_mode=4 (cv2.BORDER_REFLECT_101) so corners are
    # reflected, never black (black corners are an artifact the model could latch onto as a fake
    # domain cue). boxes are auto-warped by the bbox-aware Compose. ---
    "OpticalDistortion": {"distort_limit": (0.05, 0.20), "mode": "fisheye", "border_mode": 4, "p": 0.35},
    # --- night: darken (always negative -> never brightens), gamma, IR-gray, ISO noise, cool tint ---
    "RandomBrightnessContrast": {"brightness_limit": (-0.5, -0.05), "contrast_limit": (-0.15, 0.15), "p": 0.35},
    "RandomGamma": {"gamma_limit": (110, 220), "p": 0.30},
    "ToGray": {"p": 0.12},
    "ISONoise": {"color_shift": (0.01, 0.05), "intensity": (0.1, 0.5), "p": 0.20},
    "RGBShift": {"r_shift_limit": 10, "g_shift_limit": 10, "b_shift_limit": 25, "p": 0.20},
    # mild geometry (cameras are fixed/mounted — no vertical flip, small rotation/scale only)
    "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.05, 0.05), "rotate": (-5, 5), "p": 0.3},
}

# === Job #1 cross-city stack: Tier1 geometry + Tier2 sensor/weather + coupled day-night ==========
# Only transforms VERIFIED to build on albumentations 2.0.8 with 2.x param names (a 1.x name like
# var_limit / quality_lower / scale_min / max_holes builds-but-no-ops SILENTLY — never use them).
# Dict order == application order (from_config), staged: extra geometry -> photometric ->
# CoupledDayNight -> sensor noise/blur/compression -> weather -> occlusion. The resize OneOf +
# --multi-scale already supply base scale/resolution jitter upstream; copy-paste is a __getitem__
# hook (enable with --copy-paste), and FourierAmplitudeMix is appended only under --fourier-mix.
AUG_XCITY = {
    # --- Tier 1 geometry (mild; mounted cams -> no vertical flip; boxes warped by the wrapper) ---
    "HorizontalFlip": {"p": 0.5},
    "Affine": {"scale": (0.9, 1.1), "translate_percent": (0.0, 0.05), "rotate": (-8, 8), "shear": (-5, 5), "p": 0.5},
    "Perspective": {"scale": (0.02, 0.05), "p": 0.3},
    "GridDistortion": {"num_steps": 5, "distort_limit": 0.1, "p": 0.2},
    # --- Tier 1 photometric (box-safe pixel ops) ---
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05, "p": 0.5},
    "HueSaturationValue": {"hue_shift_limit": 10, "sat_shift_limit": 20, "val_shift_limit": 10, "p": 0.3},
    "RandomGamma": {"gamma_limit": (80, 120), "p": 0.3},
    "RandomToneCurve": {"scale": 0.1, "p": 0.2},
    "CLAHE": {"clip_limit": 4.0, "tile_grid_size": (8, 8), "p": 0.2},
    # --- coupled continuous day->evening->night (+ symmetric noon), box-safe custom transform ---
    "CoupledDayNight": {"p": 0.6, "night_bias": 0.65, "gamma_max": 2.2, "min_mean_luma": 0.06,
                        "hard_floor": 0.015, "ir_prob": 0.10, "motion_blur_max": 7},
    # --- Tier 2 sensor (box-safe; 2.x param names) ---
    "GaussNoise": {"std_range": (0.05, 0.15), "p": 0.3},
    "ISONoise": {"color_shift": (0.01, 0.05), "intensity": (0.1, 0.5), "p": 0.2},
    "MotionBlur": {"blur_limit": 7, "p": 0.2},
    "GaussianBlur": {"blur_limit": (3, 7), "p": 0.2},
    "ImageCompression": {"quality_range": (50, 90), "p": 0.3},
    "Downscale": {"scale_range": (0.5, 0.9), "p": 0.2},
    # --- weather (low p) ---
    "RandomFog": {"p": 0.1},
    "RandomShadow": {"p": 0.2},
    # --- occlusion (box-safe; degenerate boxes filtered downstream) ---
    "CoarseDropout": {"num_holes_range": (1, 4), "hole_height_range": (8, 32), "hole_width_range": (8, 32), "p": 0.3},
}

# A/B (merged over AUG_XCITY by --aug-aggressive): stronger perspective/optical/grid distortion.
AUG_XCITY_AGGRESSIVE_GEOM = {
    "Perspective": {"scale": (0.05, 0.12), "p": 0.4},
    "OpticalDistortion": {"distort_limit": 0.2, "p": 0.3},
    "GridDistortion": {"num_steps": 5, "distort_limit": 0.25, "p": 0.3},
}

AUG_PRESETS = {"baseline": AUG_CONFIG, "dg": AUG_DG, "fisheye_night": AUG_FISHEYE_NIGHT, "xcity": AUG_XCITY}


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


def reassign_splits_by_camera(dataset: HafniaDataset, val_fraction: float, seed: int) -> HafniaDataset:
    """Re-split the LABELED samples so whole cameras are held out for validation.

    Why: the native split shares cameras between train and validation (verified:
    every native-val camera also appears in train). That measures *same-camera*
    generalization. The real test set is a HIDDEN second city, so a camera the
    model never saw is a far better proxy. We hold out `val_fraction` of cameras
    *entirely* — none of their frames appear in train — to get a DG-honest signal.

    Only frames from the native train/validation splits (which carry ground truth)
    are re-split; native `test` frames (GT withheld) are left untouched and unused.
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

    counts = Counter(new_split)
    print(f"[split] leave-camera-out (seed={seed}): held out "
          f"{n_val}/{len(cams_labeled)} cameras for validation")
    print(f"[split] held-out cameras: {sorted(val_cams)}")
    print(f"[split] new split counts: {dict(counts)}")
    return out


def reassign_splits_stratified(dataset: HafniaDataset, val_fraction: float, seed: int) -> HafniaDataset:
    """Stratified-per-camera split for the FINAL (submission) model: every camera/view stays in TRAIN,
    and only a small ``val_fraction`` of each camera's frames is sampled into validation.

    Unlike :func:`reassign_splits_by_camera` (leave-camera-out, DG-honest, smaller train), this
    MAXIMIZES train size and guarantees the model sees ALL views during training. Trade-off: the
    resulting val mAP is *same-camera* (optimistic) and is only a checkpoint-selection signal, NOT a
    cross-city generalization proxy. Native ``test`` frames (GT withheld) are left untouched.

    Every camera keeps at least one frame in train, so no view is ever fully removed from training.
    """
    df = dataset.samples
    if "camera_info" not in df.columns:
        print("[split] no camera_info column — keeping native splits")
        return dataset

    cam = df.select(pl.col("camera_info").struct.field("name")).to_series().to_list()
    orig = df["split"].to_list()
    labeled = {"train", "validation"}

    by_cam: dict = defaultdict(list)
    for i, (c, s) in enumerate(zip(cam, orig)):
        if s in labeled and c is not None:
            by_cam[c].append(i)

    rng = random.Random(seed)
    val_idx: set = set()
    for c, idxs in by_cam.items():
        if len(idxs) <= 1:
            continue  # single-frame camera -> keep entirely in train (view must stay in train)
        k = min(round(len(idxs) * val_fraction), len(idxs) - 1)  # always leave >=1 frame in train
        order = list(idxs)
        rng.shuffle(order)
        val_idx.update(order[:k])

    new_split = [
        ("validation" if i in val_idx else "train") if s in labeled else s
        for i, s in enumerate(orig)
    ]
    out = dataset.update_samples(df.with_columns(pl.Series(new_split).alias("split")))

    counts = Counter(new_split)
    print(f"[split] stratified-per-camera (seed={seed}, val_fraction={val_fraction}): "
          f"all {len(by_cam)} cameras kept in train; val sampled within cameras")
    print(f"[split] new split counts: {dict(counts)}")
    return out


def export_hafnia_to_coco(dataset_name: str, version: str, out_dir: Path,
                          val_camera_frac: float, split_seed: int, split_mode: str = "camera_out") -> Path:
    """Export HafniaDataset to Roboflow-style COCO on disk, idempotent.

    ``split_mode`` selects how the labeled frames are re-split:
    - ``camera_out`` (default): leave-camera-out, DG-honest (held-out cameras absent from train).
    - ``stratified``: per-camera stratified, all cameras in train, bigger train (final-model mode).
    - ``native``: keep the dataset's native split (also shares cameras between train and val).
    ``val_camera_frac`` is the held-out fraction (of cameras for camera_out, of frames for stratified).
    """
    sentinel = out_dir / "train" / "_annotations.coco.json"
    if sentinel.exists():
        print(f"[data] reusing cached COCO dataset at {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_hafnia_dataset(dataset_name, version)
    if split_mode == "native" or not (val_camera_frac and val_camera_frac > 0):
        print("[split] using native dataset splits (cameras shared between train/val -> all views in train)")
    elif split_mode == "stratified":
        dataset = reassign_splits_stratified(dataset, val_camera_frac, split_seed)
    else:
        dataset = reassign_splits_by_camera(dataset, val_camera_frac, split_seed)
    print(f"[data] exporting to roboflow COCO layout at {out_dir}")
    dataset.to_coco_format(out_dir, coco_format_type="roboflow")
    return out_dir


BEST_CKPT_FILES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
)


def _to_bbox_primitives(predictions, image_shape, bbox_task):
    """Convert supervision Detections -> list[Bbox] with NORMALIZED coords (mirrors the official
    trainer-object-detection reference). image_shape = (H, W). Drops the n+1-th "background" class."""
    from hafnia.dataset.primitives import Bbox

    out = []
    for bbox, class_idx, confidence in zip(predictions.xyxy, predictions.class_id, predictions.confidence):
        if int(class_idx) == len(bbox_task.classes):  # n+1-th index = "no object" / background
            continue
        out.append(
            Bbox(
                height=float((bbox[3] - bbox[1]) / image_shape[0]),
                width=float((bbox[2] - bbox[0]) / image_shape[1]),
                top_left_x=float(bbox[0] / image_shape[1]),
                top_left_y=float(bbox[1] / image_shape[0]),
                class_idx=int(class_idx),
                class_name=bbox_task.classes[int(class_idx)].name,
                confidence=float(confidence),
                ground_truth=False,
            )
        )
    return out


def run_test_predictions(model, dataset, logger, threshold: float) -> None:
    """Run inference on the TEST split and save predictions in the Hafnia submission format.

    This replicates the official ``trainer-object-detection`` reference: build a HafniaDataset with
    predicted ``Bbox`` primitives (``ground_truth=False``) and persist it with
    ``write_annotations(logger._path_artifacts())`` — i.e. ``annotations.parquet`` + ``dataset_info.json``
    keyed by ``remote_path``, written to ``/opt/ml/output/data``. The platform scorer reads THAT — a
    COCO-results JSON in ``/opt/ml/model`` is the wrong format AND the wrong location.
    """
    from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
    from hafnia.dataset.benchmark.inference_model import InferenceModel
    from hafnia.dataset.dataset_names import SampleField, SplitName
    from hafnia.dataset.hafnia_dataset_types import ModelInfo
    from hafnia.dataset.primitives import Bbox

    task_info = dataset.info.get_tasks_by_primitive(Bbox)[0]
    dataset_test = dataset.create_split_dataset(split_name=SplitName.TEST)
    print(f"[predict-test] test split: {len(dataset_test)} samples; task='{task_info.name}', threshold={threshold}")

    class _Wrap(InferenceModel):
        def __init__(self, m, task, thr):
            self.model, self.task, self.thr = m, task, thr

        def get_model_info(self):
            return ModelInfo(name=self.model.__class__.__name__, tasks=[self.task])

        def predict(self, images, sample_dict=None):
            dets = self.model.predict(images, threshold=self.thr)
            return _to_bbox_primitives(dets, images.shape[:2], self.task)

    preds = run_inference_on_dataset(dataset=dataset_test, model=_Wrap(model, task_info, threshold))

    out = logger._path_artifacts()
    Path(out).mkdir(parents=True, exist_ok=True)
    drop = [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META]
    preds.samples = preds.samples.drop(drop, strict=False)
    preds.write_annotations(out)
    print(f"[predict-test] wrote Hafnia annotations (remote_path + predicted bboxes) -> {out}")


def _capture_mlflow_run_id() -> Optional[str]:
    """Return the run_id of the MLflow run HafniaLogger started, or None.

    MUST be called from the MAIN thread (where HafniaLogger called mlflow.start_run()).
    MLflow keeps the active-run stack per-thread, so a background thread calling
    HafniaLogger.log_scalar (→ fluent mlflow.log_metric) would NOT see this run and would
    spawn a fresh auto-named "orphan" run. The watcher re-binds this run_id inside its own
    thread (mlflow.start_run(run_id=...)) so the documented logger.log_scalar/log_metric API
    lands in the correct run.
    """
    try:
        import mlflow

        active = mlflow.active_run()
        return active.info.run_id if active is not None else None
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics] mlflow run capture failed ({exc!r}); local fallback")
        return None


class TrainStreamingWatcher:
    """Background thread that tails RF-DETR outputs into HafniaLogger LIVE.

    `model.train()` blocks for hours. Without live publishing the platform dashboard
    stays empty until training finishes, and if training crashes mid-run nothing
    survives. So we run this watcher in parallel:

      * tails `<output_dir>/metrics.csv` and forwards new rows via log_scalar/log_metric
      * copies `checkpoint_best_*.pth` from `output_dir` to `path_model()` whenever
        their mtime advances — so the platform's "Trained model" artifact updates
        as soon as a new best epoch lands, even if training is later killed

    Both operations are idempotent (cursor for CSV rows; mtime check for ckpts) so
    a final flush after `train()` returns is safe and recommended.
    """

    def __init__(
        self,
        hafnia_logger: HafniaLogger,
        ckpt_dir: Path,
        model_dir: Path,
        interval_seconds: float = 30.0,
    ) -> None:
        self.logger = hafnia_logger
        self.ckpt_dir = ckpt_dir
        self.model_dir = model_dir
        self.interval = interval_seconds
        self.csv_path = ckpt_dir / "metrics.csv"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows_published = 0
        self._ckpt_mtime: Dict[str, float] = {}
        # Capture the platform's MLflow run_id in the MAIN thread. The watcher re-binds it inside
        # its own thread so the documented logger.log_scalar/log_metric API targets this run
        # (and not an auto-created orphan run). None locally / when mlflow is absent.
        self._mlflow_run_id = _capture_mlflow_run_id()

    def _publish_row(self, row: dict, step: int) -> int:
        """Push every numeric cell in one CSV row via the documented HafniaLogger API.

        Uses ONLY logger.log_metric (evaluation series) / logger.log_scalar (everything else),
        exactly as the Hafnia docs and the reference trainer-classification prescribe. The
        watcher thread has already re-bound the platform's MLflow run (see _run), so these calls
        land in the official run rather than an orphan one. Returns the count pushed.
        """
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
                print(f"[metrics] skipped {key}={value} step={step}: {exc!r}")
        return n

    def _tick(self) -> None:
        # 1. metrics.csv → official run (stdout + MLflow)
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
                print(f"[watcher] +{pushed} metrics from {len(new)} new rows (total {self._rows_published} rows)")

        # 2. checkpoint_best_*.pth → path_model()
        for name in BEST_CKPT_FILES:
            src = self.ckpt_dir / name
            if not src.exists():
                continue
            mtime = src.stat().st_mtime
            if self._ckpt_mtime.get(name, 0.0) >= mtime:
                continue
            try:
                shutil.copy2(src, self.model_dir / name)
                self._ckpt_mtime[name] = mtime
                print(f"[watcher] published {name} → {self.model_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"[watcher] copy {name} error: {exc!r}")

    def _bind_mlflow_run_in_thread(self) -> None:
        """Re-bind the platform's MLflow run to THIS thread's active-run stack.

        After this, the documented logger.log_scalar/log_metric (→ fluent mlflow.log_metric)
        resolves to the platform's run instead of spawning an orphan. We never end_run() here —
        HafniaLogger.end_run() in the main thread owns the run lifecycle.
        """
        if self._mlflow_run_id is None:
            return
        try:
            import mlflow

            mlflow.start_run(run_id=self._mlflow_run_id)
            print(f"[watcher] bound MLflow run {self._mlflow_run_id} in watcher thread")
        except Exception as exc:  # noqa: BLE001
            print(f"[watcher] could not bind MLflow run in thread ({exc!r}); metrics may orphan")

    def _run(self) -> None:
        # Bind the official MLflow run to this thread FIRST so every log_scalar/log_metric
        # lands in it. Then tick once immediately (early metrics appear without an interval
        # delay) and enter the periodic loop. A final tick runs in stop() to flush the tail.
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
        # One final synchronous flush so partial writes from the last interval land.
        try:
            self._tick()
        except Exception as exc:  # noqa: BLE001
            print(f"[watcher] final flush error: {exc!r}")
        print(f"[watcher] stopped — total rows published: {self._rows_published}, ckpts mirrored: {sorted(self._ckpt_mtime)}")


def prepare_warmstart_for_encoder(init_path: Path, encoder: str, work_dir: Path) -> Path:
    """Make a warm-start checkpoint safe for a DINOv3 backbone, returning the path to use.

    RF-DETR's published checkpoint carries a DINOv2 backbone. Fine-tuning a DINOv3-backbone model
    FROM it must NOT try to load those DINOv2 backbone tensors: they cannot populate a RoPE DINOv3
    backbone, and load_state_dict raises on the first same-name/different-shape collision (e.g.
    ``embeddings.mask_token`` is [1,384] in DINOv2 vs [1,1,384] in DINOv3). The DINOv3 backbone is
    already initialised from its own bundled self-supervised weights, so here we drop every
    ``backbone.0.encoder.*`` tensor from the checkpoint and keep only the projector / transformer /
    detection-head weights to warm-start. The class head is resized by RF-DETR's own loader.

    For a non-DINOv3 encoder, or a checkpoint that is already DINOv3 (no learned backbone
    position-embeddings — the DINOv2 signature), the original path is returned unchanged.
    """
    if not encoder.startswith("dinov3"):
        return init_path
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(sd, dict):
        print(f"[model] warm-start {init_path.name}: unexpected checkpoint structure — passing through unchanged")
        return init_path
    backbone_prefix = "backbone.0.encoder."
    # DINOv2 backbones carry learned position_embeddings; DINOv3 (RoPE) does not — use that as the
    # signal that this checkpoint's backbone is a foreign family we must strip.
    is_foreign_backbone = any(k.startswith(backbone_prefix) and "position_embeddings" in k for k in sd)
    if not is_foreign_backbone:
        return init_path
    stripped = {k: v for k, v in sd.items() if not k.startswith(backbone_prefix)}
    n_dropped = len(sd) - len(stripped)
    ckpt["model"] = stripped
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"warmstart_nobackbone_{init_path.stem}.pth"
    torch.save(ckpt, out_path)
    print(f"[model] {encoder}: stripped {n_dropped} DINOv2 backbone tensors from warm-start checkpoint "
          f"→ {out_path.name}. Head/neck/transformer warm-started; DINOv3 backbone from bundled weights.")
    return out_path


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
    p.add_argument(
        "--strategy",
        default=None,
        help="PTL strategy. Default: 'ddp' when --devices>1, else 'auto'. RF-DETR Large "
             "(detection) leaves some params unused per step, so multi-GPU DDP needs "
             "find_unused_parameters=True — our patched trainer enables that for strategy='ddp'.",
    )
    p.add_argument("--resolution", type=int, default=RESOLUTION, help="must be divisible by 32 for RF-DETR Large")
    p.add_argument(
        "--encoder",
        choices=["dinov2_windowed_small", "dinov3_small", "dinov3_base", "dinov3_large"],
        default="dinov2_windowed_small",
        help="backbone. Default 'dinov2_windowed_small' (RF-DETR Large stock, windowed DINOv2 ViT-S). "
             "'dinov3_*' swaps in the DINOv3 ViT (RoPE, non-windowed): its bundled self-supervised "
             "weights load for the backbone, and a DINOv2 warm-start checkpoint is reused for the "
             "head/neck/transformer ONLY (DINOv2 backbone keys are stripped — they cannot populate a "
             "RoPE backbone and would crash load_state_dict on shape mismatch).",
    )
    p.add_argument("--dataset-name", default="eccv-cross-city")
    p.add_argument("--dataset-version", default="1.0.0")
    p.add_argument(
        "--coco-dir",
        default=str(REPO_ROOT / ".data" / "coco" / "eccv-cross-city"),
        help="where to materialize the COCO copy of the dataset",
    )
    p.add_argument(
        "--val-camera-frac",
        type=float,
        default=0.2,
        help="fraction of cameras held out ENTIRELY for validation (leave-camera-out, "
             "a DG-honest proxy for the hidden target city). 0 = use the native splits.",
    )
    p.add_argument("--split-seed", type=int, default=42, help="seed for which cameras are held out")
    p.add_argument(
        "--split-mode",
        choices=["camera_out", "stratified", "native"],
        default="camera_out",
        help="camera_out = leave-camera-out (DG-honest, smaller train, default); "
             "stratified = per-camera split keeping ALL views in train with a small val (biggest "
             "train — for the FINAL submission model); native = dataset's native split.",
    )
    p.add_argument(
        "--aug-preset",
        choices=["baseline", "dg", "fisheye_night", "xcity"],
        default="baseline",
        help="baseline = mild reference augs; dg = stronger photometric+noise (Phase 2); "
             "fisheye_night = fisheye OpticalDistortion (geometric, targets the fisheye cam) + "
             "night photometric (darken/IR-gray/noise). See AUG_FISHEYE_NIGHT.",
    )
    p.add_argument(
        "--init-weights",
        default=None,
        help="Warm-start checkpoint to fine-tune FROM (path, relative to repo root or absolute), "
             "e.g. weights/v5_best_ema.pth. Overrides the bundled COCO-pretrained weights. Must be "
             "bundled under weights/ so it ships inside trainer.zip (cloud has no network).",
    )
    p.add_argument(
        "--lr-scheduler",
        choices=["step", "cosine"],
        default="step",
        help="step = RF-DETR default (lr_drop); cosine = decay toward ~0 over --epochs "
             "(better for short fine-tunes off a converged checkpoint).",
    )
    p.add_argument(
        "--multi-scale",
        action="store_true",
        help="vary input resolution per batch (scale robustness — the lever for small-object / "
             "cross-camera generalization). Costs more VRAM (peaks above --resolution).",
    )
    p.add_argument(
        "--expanded-scales",
        action="store_true",
        help="wider multi-scale range (even more VRAM; use only with small --batch-size).",
    )
    p.add_argument(
        "--letterbox",
        action="store_true",
        help="resize preserving aspect ratio (longest side -> --resolution) and pad to a square with "
             "centered gray bars, instead of squashing into a square. Avoids distorting objects "
             "(esp. tall/thin like Person and portrait frames); costs ~padding compute.",
    )
    p.add_argument(
        "--cd-fkd",
        action="store_true",
        help="CD-FKD self-distillation for domain generalization: per step, run the backbone on a CLEAN "
             "(teacher, no-grad) and a downscaled+corrupted (student) view, and add a feature-mimic loss so "
             "the student learns scale/corruption-invariant features (targets small-object + cross-camera). "
             "Uses ONLY our own data — no external teacher/data. Costs ~1.4-1.6x compute/step.",
    )
    p.add_argument("--cd-fkd-alpha", type=float, default=1.0, help="weight of the CD-FKD feature-mimic loss")
    p.add_argument("--cd-fkd-min-scale", type=float, default=0.4,
                   help="student view is downscaled to U(min_scale,1.0)x then back up (small-object simulation)")
    p.add_argument("--cd-fkd-noise-std", type=float, default=0.05,
                   help="gaussian-noise std (normalized-image space) added to the corrupted student view")
    p.add_argument(
        "--rfs",
        action="store_true",
        help="LVIS Repeat-Factor Sampling: oversample frames containing rare classes via a "
             "WeightedRandomSampler so the DETR matcher gets far more positive matches per epoch for "
             "the long tail (Person/Bicycle/Motorcycle/HeavyDuty). Targets per-class AP, not overall mAP.",
    )
    p.add_argument("--rfs-thresh", type=float, default=0.001,
                   help="RFS frequency threshold t; only classes appearing in < t of images are up-weighted. "
                        "LVIS default 0.001; with only 10 classes you likely need higher (e.g. 0.01-0.1).")
    p.add_argument("--rfs-max", type=float, default=20.0,
                   help="RFS clamp on the per-image repeat factor (caps how hard a rare frame is oversampled).")
    p.add_argument("--rfs-num-samples", type=int, default=0,
                   help="RFS samples per epoch (0 = auto = sum of repeat factors, ~2-3x dataset). "
                        "Set a value to cap epoch length / training time.")
    p.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the DINOv2 backbone (requires_grad=False on all encoder params): train only "
             "projector + decoder + heads. Frees VRAM/compute and avoids further source-overfit of the "
             "backbone — sensible when warm-starting from a checkpoint already fine-tuned on this data "
             "(e.g. v28). NOTE: --lr-encoder becomes a no-op (no encoder params get updated).",
    )
    # --- cross-city aug stack (Tier 3 copy-paste + Tier 4 Fourier + aggressive-geometry A/B) ---
    p.add_argument("--copy-paste", action="store_true",
                   help="Tier 3: paste rare-class crops (built leakage-free from the train split) into "
                        "train images. Adds boxes; runs before the geometry augs. Use with --aug-preset xcity.")
    p.add_argument("--copy-paste-max-n", type=int, default=3, help="max instances pasted per image (k~U(1,n))")
    p.add_argument("--copy-paste-p", type=float, default=0.5, help="probability an image receives any pastes")
    p.add_argument("--fourier-mix", action="store_true",
                   help="Tier 4 (A/B): FACT low-freq amplitude mix (phase-preserved -> boxes valid). "
                        "References drawn from TRAIN images only. OFF by default.")
    p.add_argument("--fourier-beta", type=float, default=0.05, help="FACT low-freq window ratio (0,0.2]")
    p.add_argument("--fourier-lambda-max", type=float, default=0.3, help="FACT max mix weight; lam~U(0,this)")
    p.add_argument("--fourier-p", type=float, default=0.3, help="FACT apply probability per image")
    p.add_argument("--fourier-ref-split", choices=["train", "test"], default="train",
                   help="FACT reference source: 'train' = rules-clean cross-camera style; 'test' = FDA "
                        "toward the hidden TARGET city (strongest, BUT transductive use of test data — "
                        "verify competition rules; potentially disqualifying).")
    p.add_argument("--aug-aggressive", action="store_true",
                   help="A/B: merge stronger perspective/optical/grid distortion over the xcity preset.")
    # NOTE: Hafnia Training-aaS containers are network-isolated — wandb.ai is unreachable.
    # Keep --wandb only for local runs. HafniaLogger writes via the platform's VPC MLflow.
    p.add_argument("--wandb", action="store_true", help="local only — W&B is blocked in Hafnia cloud")
    p.add_argument("--wandb-project", default="eccv-cross-city")
    p.add_argument("--run-name", default=None, help="W&B run name (also used as the experiment name)")
    p.add_argument("--predict-test", action="store_true",
                   help="After (0-epoch) eval, run inference on the unlabeled TEST split and write a "
                        "COCO-format predictions_test.json to the experiment outputs (for submission).")
    p.add_argument("--predict-threshold", type=float, default=0.05,
                   help="Min confidence kept when writing test predictions (low = higher recall for mAP).")
    p.add_argument("--predict-batch", type=int, default=8,
                   help="Images per inference batch for --predict-test (keep small on a 16 GB T4).")
    p.add_argument(
        "--pseudo-test-labels",
        default=None,
        help="Self-training: path (relative to repo root or absolute) to a slimmed pseudo-label file "
             "(weights/pseudo_test_labels.json, built by slim_pseudo_predictions.py from the WBF-fused "
             "ensemble test predictions). The pseudo-labeled TEST images are folded into the COCO "
             "train/ split as extra TRAIN samples (labeled train stays as the clean anchor). Must be "
             "bundled under weights/ so it ships in trainer.zip (cloud has no network).",
    )
    p.add_argument("--pseudo-threshold", type=float, default=0.5,
                   help="Min fused ensemble confidence kept as a pseudo-box (must be >= the slim "
                        "file's floor). 0.5 ≈ 6.9 boxes/img; lower keeps more (noisier) long-tail boxes.")
    p.add_argument(
        "--stream-interval",
        type=float,
        default=30.0,
        help="seconds between watcher ticks that tail metrics.csv → HafniaLogger and mirror best ckpts to path_model()",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aug = dict(AUG_PRESETS[args.aug_preset])
    # A/B knobs that mutate the resolved aug dict (order matters: Fourier is appended LAST).
    if args.aug_aggressive:
        aug = {**aug, **AUG_XCITY_AGGRESSIVE_GEOM}
    if args.fourier_mix:
        aug = {**aug, "FourierAmplitudeMix": {"beta": args.fourier_beta,
                                              "lambda_max": args.fourier_lambda_max, "p": args.fourier_p}}

    logger = HafniaLogger(project_name="eccv-cross-city-rfdetr-large")
    logger.log_configuration(
        {
            "model": "RFDETRLarge",
            "encoder": args.encoder,
            "resolution": args.resolution,
            "patch_size": 16,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "lr_encoder": args.lr_encoder,
            "lr_scheduler": args.lr_scheduler,
            "devices": args.devices,
            "val_camera_frac": args.val_camera_frac,
            "split_seed": args.split_seed,
            "aug_preset": args.aug_preset,
            "letterbox": args.letterbox,
            "init_weights": args.init_weights,
            "multi_scale": args.multi_scale,
            "expanded_scales": args.expanded_scales,
            "cd_fkd": args.cd_fkd,
            "cd_fkd_alpha": args.cd_fkd_alpha,
            "cd_fkd_min_scale": args.cd_fkd_min_scale,
            "cd_fkd_noise_std": args.cd_fkd_noise_std,
            "rfs": args.rfs,
            "rfs_thresh": args.rfs_thresh,
            "rfs_max": args.rfs_max,
            "rfs_num_samples": args.rfs_num_samples,
            "freeze_encoder": args.freeze_encoder,
            "copy_paste": args.copy_paste,
            "copy_paste_max_n": args.copy_paste_max_n,
            "copy_paste_p": args.copy_paste_p,
            "fourier_mix": args.fourier_mix,
            "fourier_beta": args.fourier_beta,
            "fourier_lambda_max": args.fourier_lambda_max,
            "fourier_p": args.fourier_p,
            "fourier_ref_split": args.fourier_ref_split,
            "aug_aggressive": args.aug_aggressive,
            "pseudo_test_labels": args.pseudo_test_labels,
            "pseudo_threshold": args.pseudo_threshold,
            "augmentations": aug,
        }
    )

    # Keep the COCO cache keyed to the split scheme so leave-camera-out and native
    # exports never collide on disk. A pseudo-label mix gets its OWN suffix so it never
    # reuses (or pollutes) a plain export's cached tree.
    _base = Path(args.coco_dir)
    if args.split_mode == "stratified":
        suffix = f"_strat{args.val_camera_frac:g}_s{args.split_seed}"
    elif args.split_mode == "native" or args.val_camera_frac <= 0:
        suffix = "_native"
    else:
        suffix = f"_locamout{args.val_camera_frac:g}_s{args.split_seed}"
    if args.pseudo_test_labels:
        suffix += f"_pseudo{args.pseudo_threshold:g}"
    coco_dir = _base.parent / (_base.name + suffix)
    # Detect a fresh export BEFORE the (idempotent) export call: the pseudo merge mutates
    # train/_annotations.coco.json in place, so it must run exactly once — only when we just
    # built the tree, never on a cached reuse (which would double-add the pseudo images).
    fresh_export = not (coco_dir / "train" / "_annotations.coco.json").exists()
    coco_dir = export_hafnia_to_coco(
        args.dataset_name, args.dataset_version, coco_dir,
        val_camera_frac=args.val_camera_frac, split_seed=args.split_seed, split_mode=args.split_mode,
    )

    # Self-training: fold the ensemble's pseudo-labeled TEST images into train/ as extra samples.
    if args.pseudo_test_labels:
        from pseudo_labels import merge_pseudo_test_into_coco  # local module (scripts/ on sys.path)

        pseudo_path = Path(args.pseudo_test_labels)
        if not pseudo_path.is_absolute():
            pseudo_path = REPO_ROOT / pseudo_path
        if not pseudo_path.exists():
            raise FileNotFoundError(
                f"--pseudo-test-labels not found at {pseudo_path}. Bundle it under weights/ "
                "(cloud is network-isolated) and rebuild the trainer."
            )
        if not fresh_export:
            print(f"[pseudo] reusing cached pseudo-merged COCO at {coco_dir} (pseudo already folded in)")
        else:
            stats = merge_pseudo_test_into_coco(coco_dir, pseudo_path, args.pseudo_threshold)
            print(f"[pseudo] merged pseudo-test @thr={args.pseudo_threshold} from {pseudo_path.name}: {stats}")
            if stats["images_added"] == 0:
                raise RuntimeError(
                    "[pseudo] 0 pseudo-test images merged — file_name keys did not match any exported "
                    f"test image (unmatched={stats['unmatched']}, test_images={stats['test_images']}). "
                    "The pseudo file_names must match the dataset's test image filenames."
                )
            if stats["unmatched"] > stats["images_added"]:
                print(f"[pseudo] WARNING: more unmatched ({stats['unmatched']}) than matched "
                      f"({stats['images_added']}) — check the pseudo file_name keying.")

    # Pick the init checkpoint: --init-weights (warm-start / fine-tune FROM a prior run) wins;
    # otherwise the bundled COCO-pretrained RF-DETR Large. A warm-start ckpt already has the
    # 10-class head, so it loads cleanly (no head re-init), unlike the 90-class COCO weights.
    if args.init_weights:
        init_path = Path(args.init_weights)
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
    else:
        init_path = _BUNDLED_PRETRAIN

    pretrain_kwargs = {}
    if init_path.exists():
        print(f"[model] init weights: {init_path}"
              + (" (warm-start / fine-tune)" if args.init_weights else " (bundled COCO-pretrain)"))
        # When swapping in a DINOv3 backbone, drop the checkpoint's DINOv2 backbone tensors so the
        # warm-start loads only head/neck/transformer (the DINOv3 backbone is already loaded from its
        # bundled self-supervised weights). No-op for the default DINOv2 encoder.
        init_path = prepare_warmstart_for_encoder(init_path, args.encoder, REPO_ROOT / ".data")
        pretrain_kwargs["pretrain_weights"] = str(init_path)
    elif is_hafnia_cloud_job():
        raise FileNotFoundError(
            f"Init weights not found at {init_path}. Hafnia cloud blocks outbound network, so "
            "RF-DETR cannot download them at runtime. Bundle the checkpoint under ./weights/ and "
            "rebuild the trainer (see scripts/download_weights.py for the base weights)."
        )

    model = RFDETRLarge(
        num_classes=NUM_CLASSES, resolution=args.resolution, encoder=args.encoder,
        freeze_encoder=args.freeze_encoder, **pretrain_kwargs
    )

    # Resolve once so the watcher and model.train() share the exact same directories.
    ckpt_dir = Path(logger.path_model_checkpoints())
    model_dir = Path(logger.path_model())

    # We intentionally do NOT pass `mlflow=True` to RF-DETR — it would start a SECOND mlflow run
    # next to the one HafniaLogger already owns. Instead the watcher tails RF-DETR's metrics.csv
    # and forwards rows through the documented HafniaLogger.log_scalar/log_metric API LIVE during
    # training (re-binding the run inside its thread so they land in the official run), with a
    # final flush in `finally` for the tail.
    watcher = TrainStreamingWatcher(
        hafnia_logger=logger,
        ckpt_dir=ckpt_dir,
        model_dir=model_dir,
        interval_seconds=args.stream_interval,
    ).start()

    # Multi-GPU (Scale) uses DDP. RF-DETR Large detection leaves some parameters unused
    # on a given step, which plain DDP rejects ("parameters that were not used in producing
    # the loss"). We pass strategy="ddp" so the patched trainer builds
    # DDPStrategy(find_unused_parameters=True). Single-GPU stays on "auto" (no DDP).
    strategy = args.strategy or ("ddp" if args.devices > 1 else "auto")
    print(f"[train] devices={args.devices} strategy={strategy!r}")

    try:
        model.train(
            dataset_dir=str(coco_dir),
            dataset_file="roboflow",
            output_dir=str(ckpt_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            lr_encoder=args.lr_encoder,
            num_workers=args.num_workers,
            devices=args.devices,
            strategy=strategy,
            lr_scheduler=args.lr_scheduler,
            class_names=CLASS_NAMES,
            aug_config=aug,
            # multi_scale varies per-batch resolution (scale robustness — key for small objects /
            # cross-camera). It raises peak VRAM well above --resolution, so pair --multi-scale with
            # a small --batch-size (e.g. 2) + larger --grad-accum-steps on a 16 GB T4.
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            cd_fkd=args.cd_fkd,
            cd_fkd_alpha=args.cd_fkd_alpha,
            cd_fkd_min_scale=args.cd_fkd_min_scale,
            cd_fkd_noise_std=args.cd_fkd_noise_std,
            rfs=args.rfs,
            rfs_thresh=args.rfs_thresh,
            rfs_max=args.rfs_max,
            rfs_num_samples=args.rfs_num_samples,
            copy_paste=args.copy_paste,
            copy_paste_max_n=args.copy_paste_max_n,
            copy_paste_p=args.copy_paste_p,
            fourier_ref_split=args.fourier_ref_split,
            square_resize_div_64=True,
            letterbox=args.letterbox,
            use_ema=True,
            tensorboard=True,
            # RF-DETR defaults train_log_on_step=False → train metrics aggregate to ONE point
            # per epoch, so charts look like a single dot on short runs. Turn it on so train
            # metrics are logged per-step (CSVLogger flushes every log_every_n_steps=50) → a real
            # curve. Validation stays one-point-per-epoch (you only validate once per epoch — that
            # is inherent, not a bug).
            train_log_on_step=True,
            wandb=args.wandb,
            project=args.wandb_project,
            run=args.run_name,
        )
    finally:
        # Stop the watcher whether train() succeeded, raised, or got SIGTERM'd.
        # `stop()` also runs one last tick so partial writes between the final interval
        # and end-of-training still make it into HafniaLogger / path_model().
        watcher.stop()

    # Eval-only run (--epochs 0 = pure inference): validate() writes no checkpoint, so persist the
    # evaluated weights as the official "Trained model" artifact. This makes a 0-epoch run download-
    # able like a trained one and lets it complete SUCCEEDED — used to lock in a known-good checkpoint
    # (e.g. v5 @ R1280, which scored the epoch-0 0.4754) without the degrading fine-tune.
    if args.epochs == 0 and init_path.exists():
        import shutil

        model_dir.mkdir(parents=True, exist_ok=True)
        dst = model_dir / "checkpoint_best_ema.pth"
        shutil.copy(str(init_path), str(dst))
        print(f"[eval-only] persisted evaluated weights -> {dst}")

    # Optional: inference on the TEST split -> Hafnia submission annotations (the platform scorer
    # reads write_annotations() output in /opt/ml/output/data, NOT a COCO JSON in /opt/ml/model).
    if args.predict_test:
        # After training, `model.model` holds the LAST epoch's regular weights — but the best
        # checkpoint (EMA/regular winner picked by BestModelCallback on val mAP) is usually better.
        # Reload it for inference so the submission comes from the best model, not the final one.
        # For --epochs 0 the loaded init weights ARE the evaluated model — use them as-is.
        predict_model = model
        if args.epochs > 0:
            best = next((ckpt_dir / f for f in BEST_CKPT_FILES if (ckpt_dir / f).exists()), None)
            if best is not None:
                print(f"[predict-test] loading best checkpoint for inference: {best.name}")
                predict_model = RFDETRLarge(
                    num_classes=NUM_CLASSES, resolution=args.resolution, encoder=args.encoder,
                    pretrain_weights=str(best),
                )
            else:
                print("[predict-test] WARNING: no best checkpoint found; predicting with final weights")
        # The competition TEST split is the dataset's NATIVE test (never touched by any re-split),
        # so load the raw dataset and predict on its test split directly — independent of split_mode.
        ds_pred = load_hafnia_dataset(args.dataset_name, args.dataset_version)
        run_test_predictions(predict_model, ds_pred, logger, args.predict_threshold)

    print(f"[done] checkpoints in {ckpt_dir}; trained model in {model_dir}")


if __name__ == "__main__":
    main()
