# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningDataModule for RF-DETR dataset construction and loaders."""

import math
from collections import Counter
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.utils.data
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets import build_dataset
from rfdetr.datasets.aug_config import AUG_CONFIG
from rfdetr.utilities.box_ops import box_xyxy_to_cxcywh
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import make_collate_fn

logger = get_logger()

_MIN_TRAIN_BATCHES = 5


def _has_cuda_device() -> bool:
    """Return ``True`` when the runtime has a CUDA accelerator available.

    Uses the fork-safe global ``DEVICE`` constant instead of direct ``torch.cuda.is_available()`` calls to avoid
    creating a CUDA context in fork-based notebook/DDP workflows.
    """
    from rfdetr.config import DEVICE

    return str(DEVICE).startswith("cuda")


class GradAccumAlignedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that pads length to a multiple of ``effective_batch_size * world_size``.

    Workaround for https://github.com/Lightning-AI/pytorch-lightning/issues/19987: PTL fires the optimizer on partial
    accumulation windows at the tail of the dataset, causing the last optimizer step to be under-scaled.  Padding the
    dataset to a multiple of ``effective_batch_size * world_size`` ensures that ``drop_last=True`` on the DataLoader
    becomes a true no-op — every accumulation window is always complete.

    Padding indices are drawn randomly from the original dataset.  Because RF-DETR uses online augmentation, each padded
    sample receives a fresh random augmentation at ``__getitem__`` time, so it behaves like a new training example
    rather than a true duplicate.

    This wrapper can be removed once the upstream PTL issue is resolved.

    Args:
        dataset: The underlying dataset to wrap.
        effective_batch_size: ``batch_size * grad_accum_steps``.
        world_size: Number of DDP processes (default 1 for single-GPU/CPU).
            The alignment unit is ``effective_batch_size * world_size`` so that after PTL's ``DistributedSampler``
            splits samples across ranks each rank still receives an exact multiple of ``effective_batch_size``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        effective_batch_size: int,
        world_size: int = 1,
    ) -> None:
        if effective_batch_size < 1:
            raise ValueError(f"effective_batch_size must be >= 1, got {effective_batch_size}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")

        self._dataset = dataset
        self._dataset_length = len(dataset)  # type: ignore[arg-type]
        pad_unit = effective_batch_size * world_size
        remainder = self._dataset_length % pad_unit
        pad_count = (pad_unit - remainder) % pad_unit
        pad_index_generator = torch.Generator()
        pad_index_generator.manual_seed(0)
        self._pad_indices: list[int] = (
            torch.randint(
                0,
                self._dataset_length,
                (pad_count,),
                generator=pad_index_generator,
            ).tolist()
            if pad_count > 0
            else []
        )
        self._length = self._dataset_length + pad_count

    def __len__(self) -> int:
        """Return the padded dataset length (always a multiple of the alignment unit)."""
        return self._length

    def __getitem__(self, idx: int) -> Any:
        """Return the item at the (possibly remapped) index."""
        # pad_indices are fixed at __init__ time; same indices reused every epoch
        # (different augmentations per epoch due to online augmentation)
        dataset_idx = idx if idx < self._dataset_length else self._pad_indices[idx - self._dataset_length]
        return self._dataset[dataset_idx]


def _resolve_augmentation_backend(backend: str) -> str:
    """Resolve ``"auto"`` to ``"cpu"`` or ``"gpu"`` based on runtime availability.

    For ``"cpu"`` and ``"gpu"`` the value is returned unchanged.  For ``"auto"`` the function checks CUDA and kornia
    availability and returns ``"gpu"`` only when both are present; otherwise ``"cpu"``.

    Called before dataset construction so that ``gpu_postprocess`` in the dataset builders always matches what the
    DataModule will actually do in ``on_after_batch_transfer``.

    Args:
        backend: Value of ``TrainConfig.augmentation_backend``.

    Returns:
        Resolved backend string, either ``"cpu"`` or ``"gpu"``.

    Examples:
        >>> _resolve_augmentation_backend("cpu")
        'cpu'
        >>> _resolve_augmentation_backend("gpu")
        'gpu'
    """
    if backend != "auto":
        return backend
    if not _has_cuda_device():
        return "cpu"
    try:
        import kornia.augmentation  # noqa: F401 # type: ignore[import-not-found]

        return "gpu"
    except ImportError:
        return "cpu"


def repeat_factor_category_factors(
    per_image_categories: List[Set[int]],
    thresh: float = 0.001,
    max_factor: float = 20.0,
) -> "dict[int, tuple[int, float, float]]":
    """Per-category RFS diagnostics: ``{category: (image_count, image_freq, repeat_factor)}``.

    For each category ``c`` returns how many images contain it, its image-frequency ``f_c``, and the
    LVIS category repeat factor ``r_c = clamp(sqrt(thresh / f_c), 1, max_factor)``.  Used both to build
    the per-image weights (:func:`repeat_factor_weights`) and to log the long-tail picture so ``thresh``
    can be tuned — a category with ``repeat_factor == 1.0`` is not being oversampled at all.

    Args:
        per_image_categories: For each image, the set of contiguous class labels present.
        thresh: Frequency threshold ``t``; only categories with ``f_c < thresh`` get ``r_c > 1``.
        max_factor: Upper clamp on the repeat factor.

    Returns:
        Mapping from category label to ``(image_count, image_freq, repeat_factor)``.  Empty when there
        are no images.
    """
    n_images = len(per_image_categories)
    if n_images == 0:
        return {}

    cat_image_count: "Counter[int]" = Counter()
    for cats in per_image_categories:
        for c in cats:
            cat_image_count[c] += 1

    stats: "dict[int, tuple[int, float, float]]" = {}
    for c, count in cat_image_count.items():
        freq = count / n_images
        factor = min(max_factor, max(1.0, math.sqrt(thresh / freq)))
        stats[c] = (count, freq, factor)
    return stats


def repeat_factor_weights(
    per_image_categories: List[Set[int]],
    thresh: float = 0.001,
    max_factor: float = 20.0,
) -> List[float]:
    """Compute LVIS Repeat-Factor Sampling image-level repeat factors.

    Implements the image resampling scheme from LVIS (Gupta et al., CVPR 2019, arXiv:1908.03195),
    matching the ``ClassBalancedDataset`` / ``RepeatFactorTrainingSampler`` formulation used by
    MMDetection and Detectron2.  For each category ``c`` the image-frequency ``f_c`` is the fraction
    of images that contain it; the category repeat factor is ``r_c = clamp(sqrt(thresh / f_c), 1, max_factor)``
    and the per-image factor is the max over the categories present (the rarest category dominates).
    Images with no annotations get factor ``1.0`` (never up-weighted).

    These factors are used as per-image weights for a ``WeightedRandomSampler`` so that frames
    containing rare classes are sampled more often — giving the DETR Hungarian matcher far more
    positive matches per epoch for the long-tail classes whose embeddings otherwise barely move.

    Args:
        per_image_categories: For each image (in dataset ``__getitem__`` order) the set of
            contiguous class labels present in that image.
        thresh: Frequency threshold ``t``.  Only categories with ``f_c < thresh`` are up-weighted.
            The LVIS default is ``0.001``; raise it for datasets with few classes/images.
        max_factor: Upper clamp on the per-category (and hence per-image) repeat factor.

    Returns:
        A list of per-image weights, same length and order as ``per_image_categories``.
    """
    cat_factor = {
        c: factor
        for c, (_, _, factor) in repeat_factor_category_factors(per_image_categories, thresh, max_factor).items()
    }
    return [max((cat_factor[c] for c in cats), default=1.0) for cats in per_image_categories]


def aligned_num_samples(weights: List[float], effective_batch_size: int, override: int = 0) -> int:
    """Pick ``num_samples`` for the weighted sampler, floored to a multiple of ``effective_batch_size``.

    Keeping the per-epoch sample count an exact multiple of ``batch_size * grad_accum_steps`` means
    ``drop_last=True`` never fires the optimizer on a partial gradient-accumulation window — the same
    invariant :class:`GradAccumAlignedDataset` enforces for the shuffled path.

    Args:
        weights: Per-image repeat-factor weights.  ``sum(weights)`` is the natural (full-RFS) epoch
            size when *override* is not given.
        effective_batch_size: ``batch_size * grad_accum_steps`` (per process).
        override: When ``> 0``, target this many samples per epoch instead of ``sum(weights)``
            (use it to cap the ~2-3x epoch growth that full RFS introduces).

    Returns:
        ``num_samples`` — a positive multiple of *effective_batch_size* (at least one batch).
    """
    target = override if override > 0 else round(sum(weights))
    aligned = (target // effective_batch_size) * effective_batch_size
    return max(effective_batch_size, aligned)


class RFDETRDataModule(LightningDataModule):
    """LightningDataModule wrapping RF-DETR dataset construction and data loading.

    Args:
        model_config: Architecture configuration (used for resolution, patch_size, etc.).
        train_config: Training hyperparameter configuration (used for dataset params).
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        # Backbone divisibility requirement: inputs with windowed attention must
        # have H and W divisible by patch_size * num_windows. The collate_fn
        # below rounds batch-max H/W up to this value so the mask accurately
        # marks every pad pixel.
        block_size = model_config.patch_size * model_config.num_windows
        if block_size <= 0:
            raise ValueError(
                "Computed collate block_size must be > 0, got "
                f"{block_size} from patch_size={model_config.patch_size} "
                f"and num_windows={model_config.num_windows}."
            )
        self._collate_fn = make_collate_fn(
            block_size=block_size,
        )

        self._dataset_train: Optional[torch.utils.data.Dataset] = None
        self._dataset_val: Optional[torch.utils.data.Dataset] = None
        self._dataset_test: Optional[torch.utils.data.Dataset] = None

        # GPU augmentation pipeline (Kornia); built lazily in setup("fit").
        self._kornia_pipeline: Any | None = None
        self._kornia_normalize: Any | None = None
        # Sentinel: True once _setup_kornia_pipeline has run (even on fallback paths
        # where _kornia_pipeline stays None), preventing redundant re-runs on repeated
        # setup("fit") calls (e.g. during validation loops in some PTL strategies).
        self._kornia_setup_done: bool = False

        self._num_workers: int = self.train_config.num_workers

        # Use the fork-safe DEVICE constant instead of torch.cuda.is_available(),
        # which creates a CUDA driver context that breaks fork-based DDP.
        from rfdetr.config import DEVICE

        accelerator = str(self.train_config.accelerator).lower()
        uses_cuda_accelerator = accelerator in {"auto", "gpu", "cuda"}
        self._pin_memory: bool = (
            (DEVICE == "cuda" and uses_cuda_accelerator)
            if self.train_config.pin_memory is None
            else bool(self.train_config.pin_memory)
        )
        self._persistent_workers: bool = (
            self._num_workers > 0
            if self.train_config.persistent_workers is None
            else bool(self.train_config.persistent_workers)
        )
        if self._num_workers > 0:
            self._prefetch_factor = (
                self.train_config.prefetch_factor if self.train_config.prefetch_factor is not None else 2
            )
        else:
            self._prefetch_factor = None

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """Build datasets for the requested stage.

        PTL calls this on every process before the corresponding dataloader method.  Datasets are built lazily — a
        dataset is only constructed once even if ``setup`` is called multiple times.

        Args:
            stage: PTL stage identifier — one of ``"fit"``, ``"validate"``,
                ``"test"``, or ``"predict"``.
        """
        resolution = self.model_config.resolution
        ns = _namespace_from_configs(self.model_config, self.train_config)
        if stage == "fit":
            # Resolve 'auto' to an actual backend before building datasets so that
            # gpu_postprocess in dataset builders always matches what the DataModule
            # will actually do in on_after_batch_transfer.  Without this, 'auto' on
            # a machine without CUDA/kornia would strip CPU Normalize from datasets
            # while _kornia_pipeline stays None, leaving training inputs unnormalized.
            resolved = _resolve_augmentation_backend(self.train_config.augmentation_backend)
            if resolved != self.train_config.augmentation_backend:
                ns.augmentation_backend = resolved
            if self._dataset_train is None:
                self._dataset_train = build_dataset("train", ns, resolution)
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
            # Build Kornia GPU augmentation pipeline (once).
            # Use _kornia_setup_done (not _kornia_pipeline is None) so that fallback
            # paths — where the pipeline stays None — do not re-run on every setup("fit").
            if not self._kornia_setup_done:
                self._setup_kornia_pipeline()
                self._kornia_setup_done = True
        elif stage == "validate":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
        elif stage == "test":
            if self._dataset_test is None:
                split = "test" if self.train_config.dataset_file == "roboflow" else "val"
                self._dataset_test = build_dataset(split, ns, resolution)
        elif stage == "predict":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader.

        Uses a replacement sampler when the dataset is too small to fill ``_MIN_TRAIN_BATCHES`` effective batches
        (matching legacy behaviour in ``main.py``).  Otherwise wraps the dataset with :class:`GradAccumAlignedDataset`
        to ensure its length is an exact multiple of ``effective_batch_size * world_size`` (workaround for
        https://github.com/Lightning-AI/pytorch-lightning/issues/19987) and then uses ``shuffle=True, drop_last=True``
        so that PTL can auto-inject ``DistributedSampler`` in DDP mode.

        Returns:
            DataLoader for the training dataset.
        """
        dataset = self._dataset_train
        batch_size = self.train_config.batch_size
        effective_batch_size = batch_size * self.train_config.grad_accum_steps
        num_workers = self._num_workers

        # Repeat-Factor Sampling: oversample frames containing rare classes (LVIS). Takes precedence
        # over both the small-dataset and aligned-shuffle paths because the WeightedRandomSampler's
        # num_samples (kept a multiple of effective_batch_size) handles length/alignment itself, so no
        # GradAccumAlignedDataset wrapper is needed. Single-GPU only injects this sampler directly; for
        # DDP the trainer must set use_distributed_sampler=False (RFS is a single-GPU Lite feature here).
        if getattr(self.train_config, "rfs", False):
            weights = self._compute_repeat_factor_weights()
            num_samples = aligned_num_samples(
                weights, effective_batch_size, override=self.train_config.rfs_num_samples
            )
            logger.info(
                "RFS enabled: %d weighted samples/epoch over %d images "
                "(effective_batch_size=%d, thresh=%g, max=%g)",
                num_samples,
                len(dataset),
                effective_batch_size,
                self.train_config.rfs_thresh,
                self.train_config.rfs_max,
            )
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=True,
                collate_fn=self._collate_fn,
                num_workers=num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
            )

        if len(dataset) < effective_batch_size * _MIN_TRAIN_BATCHES:
            logger.info(
                "Training with uniform sampler because dataset is too small: %d < %d",
                len(dataset),
                effective_batch_size * _MIN_TRAIN_BATCHES,
            )
            sampler = torch.utils.data.RandomSampler(
                dataset,
                replacement=True,
                num_samples=effective_batch_size * _MIN_TRAIN_BATCHES,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self._collate_fn,
                num_workers=num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
            )

        # Pad the dataset to a multiple of effective_batch_size * world_size so
        # that drop_last=True below becomes a true no-op and PTL never fires the
        # optimizer on a partial accumulation window.
        # See https://github.com/Lightning-AI/pytorch-lightning/issues/19987
        world_size: int = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        dataset = GradAccumAlignedDataset(dataset, effective_batch_size, world_size)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # no-op after alignment, but keeps intent explicit
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def _compute_repeat_factor_weights(self) -> List[float]:
        """Build per-image LVIS repeat-factor weights for the training dataset.

        Reads category presence per image from the COCO API object exposed by :class:`CocoDetection`
        (``self._dataset_train.coco``), in ``self._dataset_train.ids`` order so the weights align with
        the dataset's ``__getitem__`` indices.  Sparse COCO ``category_id`` values are mapped to
        contiguous labels via ``cat2label`` when present.  Crowd annotations are ignored.

        Returns:
            One weight per training image, aligned to the dataset's sample order.

        Raises:
            ValueError: If the training dataset does not expose a ``.coco`` COCO API object.
        """
        dataset = self._dataset_train
        coco = getattr(dataset, "coco", None)
        ids = getattr(dataset, "ids", None)
        if coco is None or ids is None:
            raise ValueError(
                "RFS (rfs=True) requires a COCO-style training dataset exposing '.coco' and '.ids'; "
                f"got {type(dataset).__name__}."
            )
        cat2label = getattr(dataset, "cat2label", None)

        per_image_categories: List[Set[int]] = []
        for image_id in ids:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
            categories: Set[int] = set()
            for ann in anns:
                if ann.get("iscrowd", 0) == 1:
                    continue
                category_id = ann["category_id"]
                if cat2label is not None:
                    category_id = cat2label.get(category_id, category_id)
                categories.add(category_id)
            per_image_categories.append(categories)

        thresh = self.train_config.rfs_thresh
        max_factor = self.train_config.rfs_max
        weights = repeat_factor_weights(per_image_categories, thresh=thresh, max_factor=max_factor)

        # Diagnostic: print the long-tail picture and per-class repeat factors so rfs_thresh can be
        # tuned. A class with repeat_factor == 1.0 is NOT being oversampled (thresh is below its
        # image-frequency); raise --rfs-thresh until the rare classes you care about get factor > 1.
        stats = repeat_factor_category_factors(per_image_categories, thresh=thresh, max_factor=max_factor)
        # Map contiguous labels back to human-readable class names via the COCO API when possible.
        label_to_name: dict = {}
        cats_meta = getattr(coco, "cats", None)
        if cat2label is not None and isinstance(cats_meta, dict):
            id_to_name = {cid: cats_meta[cid].get("name", str(cid)) for cid in cats_meta}
            label_to_name = {label: id_to_name.get(cid, str(cid)) for cid, label in cat2label.items()}
        oversample = sum(weights) / len(weights) if weights else 1.0
        logger.info(
            "RFS per-class repeat factors (thresh=%g, max=%g) over %d images — epoch oversample N/M=%.3f:",
            thresh, max_factor, len(per_image_categories), oversample,
        )
        for label, (count, freq, factor) in sorted(stats.items(), key=lambda kv: kv[1][2], reverse=True):
            name = label_to_name.get(label, str(label))
            flag = "" if factor > 1.0 else "   (not oversampled — raise --rfs-thresh)"
            logger.info("  %-28s images=%6d  freq=%.4f  repeat=%.2f%s", name, count, freq, factor, flag)

        return weights

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Returns:
            DataLoader for the validation dataset with sequential sampling.
        """
        return DataLoader(
            self._dataset_val,
            batch_size=self.train_config.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_val),
            drop_last=False,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader.

        Returns:
            DataLoader for the test dataset with sequential sampling.
        """
        return DataLoader(
            self._dataset_test,
            batch_size=self.train_config.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_test),
            drop_last=False,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the predict DataLoader (reuses the validation dataset, no augmentation).

        Returns:
            DataLoader for the validation dataset with sequential sampling.
        """
        return DataLoader(
            self._dataset_val,
            batch_size=self.train_config.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_val),
            drop_last=False,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def _setup_kornia_pipeline(self) -> None:
        """Resolve augmentation backend and build the Kornia pipeline if applicable.

        Called once during ``setup("fit")``.  When ``augmentation_backend`` is ``"cpu"`` this is a no-op.  For
        ``"auto"`` the method falls back silently when CUDA or Kornia are unavailable.  For ``"gpu"`` missing
        requirements raise hard errors.
        """
        backend = self.train_config.augmentation_backend
        if backend == "cpu":
            return

        if backend == "auto":
            if not _has_cuda_device():
                logger.warning("augmentation_backend='auto': no CUDA, falling back to CPU augmentation")
                return
            try:
                import kornia.augmentation  # type: ignore[import-not-found]
            except ImportError:
                logger.warning("augmentation_backend='auto': kornia not installed, using CPU augmentation")
                return
        elif backend == "gpu":
            if not _has_cuda_device():
                raise RuntimeError("augmentation_backend='gpu' requires a CUDA device")
            try:
                import kornia.augmentation  # noqa: F401 # type: ignore[import-not-found]
            except ImportError as err:
                raise ImportError(
                    "GPU augmentation requires kornia. Install with: pip install 'rfdetr[kornia]'"
                ) from err

        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline, build_normalize

        self._kornia_pipeline = build_kornia_pipeline(
            self.train_config.aug_config if self.train_config.aug_config is not None else AUG_CONFIG,
            self.model_config.resolution,
            with_masks=self.model_config.segmentation_head,
        )
        self._kornia_normalize = build_normalize()
        logger.info("Kornia GPU augmentation pipeline built (backend=%s)", backend)

    def on_after_batch_transfer(self, batch: Tuple, dataloader_idx: int) -> Tuple:
        """Apply Kornia GPU augmentation after the batch is transferred to device.

        When ``_kornia_pipeline`` is set and the trainer is in training mode, augmentation and normalization are applied
        on the GPU.  Validation and test batches pass through unchanged.

        Segmentation models use a mask-aware pipeline (``with_masks=True``) so images, boxes, and per-instance masks are
        augmented in sync.

        Args:
            batch: Tuple of ``(NestedTensor, list[dict])`` already on device.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The (possibly augmented) batch.
        """
        if self.trainer is None or not self.trainer.training or self._kornia_pipeline is None:
            return batch

        from rfdetr.datasets.kornia_transforms import collate_boxes, collate_masks, unpack_boxes
        from rfdetr.utilities.tensors import NestedTensor

        samples, targets = batch
        img = samples.tensors  # [B, C, H, W]
        # Move Kornia modules to the batch device (no-op if already there).
        # nn.Module.to() is in-place; no reassignment needed.
        self._kornia_pipeline.to(img.device)
        self._kornia_normalize.to(img.device)
        boxes_padded, valid = collate_boxes(targets, img.device)

        if self.model_config.segmentation_head:
            image_height, image_width = img.shape[-2:]
            masks_padded = collate_masks(
                targets, img.device, n_max=valid.shape[1], image_height=image_height, image_width=image_width
            )
            img_aug, boxes_aug, masks_aug = self._kornia_pipeline(img, boxes_padded, masks_padded)
            img_aug = self._kornia_normalize(img_aug)
            targets = unpack_boxes(boxes_aug, valid, targets, *img_aug.shape[-2:], masks_aug=masks_aug)
        else:
            img_aug, boxes_aug = self._kornia_pipeline(img, boxes_padded)
            img_aug = self._kornia_normalize(img_aug)
            targets = unpack_boxes(boxes_aug, valid, targets, *img_aug.shape[-2:])

        height, width = img_aug.shape[-2:]
        for target in targets:
            boxes = target["boxes"]
            if boxes.numel() == 0:
                continue
            scale = boxes.new_tensor([width, height, width, height])
            target["boxes"] = box_xyxy_to_cxcywh(boxes) / scale
        batch = (NestedTensor(img_aug, samples.mask), targets)
        return batch

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> Optional[List[str]]:
        """Class names from the training or validation dataset annotation file.

        Reads category names from the first available COCO-style dataset. Returns ``None`` if no dataset has been set up
        yet or the dataset does not expose COCO-style category information.

        Returns:
            Sorted list of class name strings, or ``None``.
        """
        for dataset in (self._dataset_train, self._dataset_val):
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                return [coco.cats[k]["name"] for k in sorted(coco.cats.keys())]
        return None

    def transfer_batch_to_device(self, batch: Tuple, device: torch.device, dataloader_idx: int) -> Tuple:
        """Move a ``(NestedTensor, targets)`` batch to *device*.

        PTL's default iterates tuple elements and calls ``.to(device)``; that works for plain tensors but
        ``NestedTensor`` must be moved explicitly.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            device: Target device.
            dataloader_idx: Index of the dataloader providing this batch.

        Returns:
            Batch with all tensors on ``device``.
        """
        samples, targets = batch
        non_blocking = device.type == "cuda"
        samples = samples.to(device, non_blocking=non_blocking)
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]
        return samples, targets
