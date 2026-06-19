# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for LVIS Repeat-Factor Sampling helpers in module_data."""

from __future__ import annotations

import math
from types import SimpleNamespace

from torch.utils.data import RandomSampler, WeightedRandomSampler

from rfdetr.config import TrainConfig
from rfdetr.training.module_data import (
    RFDETRDataModule,
    aligned_num_samples,
    repeat_factor_category_factors,
    repeat_factor_weights,
)


class _FakeCoco:
    """Minimal pycocotools.COCO stand-in: getAnnIds / loadAnns over an in-memory ann list."""

    def __init__(self, img_to_cats: dict[int, list[int]]) -> None:
        self._anns: dict[int, dict] = {}
        self._img_to_annids: dict[int, list[int]] = {}
        ann_id = 0
        for img_id, cats in img_to_cats.items():
            ids: list[int] = []
            for c in cats:
                self._anns[ann_id] = {"id": ann_id, "image_id": img_id, "category_id": c, "iscrowd": 0}
                ids.append(ann_id)
                ann_id += 1
            self._img_to_annids[img_id] = ids

    def getAnnIds(self, imgIds: int | list[int]) -> list[int]:  # noqa: N802 (match pycocotools API)
        ids = imgIds if isinstance(imgIds, (list, tuple)) else [imgIds]
        out: list[int] = []
        for i in ids:
            out.extend(self._img_to_annids.get(i, []))
        return out

    def loadAnns(self, ann_ids: list[int]) -> list[dict]:  # noqa: N802 (match pycocotools API)
        return [self._anns[i] for i in ann_ids]


class _FakeDataset:
    """CocoDetection stand-in exposing the attributes RFS reads: ids, coco, cat2label."""

    def __init__(self, img_to_cats: dict[int, list[int]]) -> None:
        self.ids = list(img_to_cats.keys())
        self.coco = _FakeCoco(img_to_cats)
        self.cat2label = None

    def __len__(self) -> int:
        return len(self.ids)


def _make_datamodule(dataset: _FakeDataset, **train_overrides) -> RFDETRDataModule:
    model_config = SimpleNamespace(patch_size=16, num_windows=2)
    train_config = SimpleNamespace(
        num_workers=0,
        accelerator="cpu",
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        augmentation_backend="cpu",
        batch_size=1,
        grad_accum_steps=8,
        rfs=False,
        rfs_thresh=0.5,
        rfs_max=20.0,
        rfs_num_samples=0,
    )
    for key, value in train_overrides.items():
        setattr(train_config, key, value)
    dm = RFDETRDataModule(model_config, train_config)
    dm._dataset_train = dataset
    return dm


class TestRepeatFactorWeights:
    """LVIS repeat-factor formula: r_c = clamp(sqrt(t/f_c), 1, max); r_i = max_c r_c."""

    def test_uniform_single_class_all_ones(self) -> None:
        # Every image has the same single class -> f_c = 1.0 -> factor clamps to 1.0.
        per_image = [{0}, {0}, {0}, {0}]
        weights = repeat_factor_weights(per_image, thresh=0.001, max_factor=20.0)
        assert weights == [1.0, 1.0, 1.0, 1.0]

    def test_boosts_rare_over_common(self) -> None:
        # class 0 in 9/10 images (common), class 1 in 1/10 (rare). With t=0.5 the rare
        # class exceeds the threshold and is up-weighted; the common one stays at 1.0.
        per_image = [{0}] * 9 + [{1}]
        weights = repeat_factor_weights(per_image, thresh=0.5, max_factor=20.0)
        f_rare = 1 / 10
        expected_rare = math.sqrt(0.5 / f_rare)  # sqrt(5) ~= 2.236
        assert weights[-1] == expected_rare
        assert all(w == 1.0 for w in weights[:9])
        assert weights[-1] > weights[0]

    def test_clamped_to_max_factor(self) -> None:
        # An extremely rare class would yield a huge sqrt; it must be clamped.
        per_image = [{0}] * 999 + [{1}]
        weights = repeat_factor_weights(per_image, thresh=0.5, max_factor=3.0)
        assert weights[-1] == 3.0  # sqrt(0.5/0.001) ~= 22 -> clamped to 3.0

    def test_image_takes_rarest_category(self) -> None:
        # An image holding both a common and a rare class gets the rare class's factor.
        per_image = [{0}] * 9 + [{0, 1}]
        weights = repeat_factor_weights(per_image, thresh=0.5, max_factor=20.0)
        f_rare = 1 / 10
        assert weights[-1] == math.sqrt(0.5 / f_rare)

    def test_empty_image_is_one(self) -> None:
        # Images with no annotations are never repeated.
        per_image = [set(), {0}]
        weights = repeat_factor_weights(per_image, thresh=0.5, max_factor=20.0)
        assert weights[0] == 1.0

    def test_preserves_order_and_length(self) -> None:
        per_image = [{1}, {0}, {0}, {1}, {0}]
        weights = repeat_factor_weights(per_image, thresh=0.5, max_factor=20.0)
        assert len(weights) == len(per_image)
        # positions with the rare class (1) must outweigh the common one (0)
        assert weights[0] == weights[3] > weights[1] == weights[2] == weights[4]


class TestRepeatFactorCategoryFactors:
    """Per-category diagnostics: {category: (image_count, image_freq, repeat_factor)}."""

    def test_returns_count_freq_factor(self) -> None:
        per_image = [{0}] * 9 + [{1}]  # class 0 in 9/10 images, class 1 in 1/10
        stats = repeat_factor_category_factors(per_image, thresh=0.5, max_factor=20.0)
        assert stats[0] == (9, 0.9, 1.0)  # common: factor clamps up to 1.0
        count, freq, factor = stats[1]
        assert count == 1
        assert freq == 0.1
        assert factor == math.sqrt(0.5 / 0.1)  # rare: sqrt(5)

    def test_factor_clamped_to_max(self) -> None:
        per_image = [{0}] * 999 + [{1}]
        stats = repeat_factor_category_factors(per_image, thresh=0.5, max_factor=3.0)
        assert stats[1][2] == 3.0

    def test_empty_input(self) -> None:
        assert repeat_factor_category_factors([], thresh=0.5) == {}

    def test_consistent_with_repeat_factor_weights(self) -> None:
        # The per-image weight must equal the max category factor over the image's classes.
        per_image = [{0, 1}, {0}, {1}]
        stats = repeat_factor_category_factors(per_image, thresh=0.5, max_factor=20.0)
        weights = repeat_factor_weights(per_image, thresh=0.5, max_factor=20.0)
        assert weights[0] == max(stats[0][2], stats[1][2])
        assert weights[1] == stats[0][2]
        assert weights[2] == stats[1][2]


class TestAlignedNumSamples:
    """num_samples for WeightedRandomSampler, floored to a multiple of effective_batch_size."""

    def test_auto_floors_sum_to_eff_bs(self) -> None:
        # sum(weights) = 21 -> floor to a multiple of 8 -> 16.
        weights = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]  # sum 21
        assert aligned_num_samples(weights, effective_batch_size=8) == 16

    def test_override_floors_to_eff_bs(self) -> None:
        weights = [1.0] * 1000
        assert aligned_num_samples(weights, effective_batch_size=8, override=100) == 96

    def test_never_below_one_batch(self) -> None:
        weights = [1.0, 1.0]  # sum 2 -> floor to mult of 8 would be 0; must be >= 8
        assert aligned_num_samples(weights, effective_batch_size=8) == 8


class TestTrainDataloaderRFSWiring:
    """train_dataloader swaps in a WeightedRandomSampler only when rfs is enabled."""

    def test_uses_weighted_sampler_when_rfs_enabled(self) -> None:
        img_to_cats = {i: [0] for i in range(9)}
        img_to_cats[9] = [1]  # one rare-class image
        dm = _make_datamodule(_FakeDataset(img_to_cats), rfs=True, rfs_thresh=0.5)
        loader = dm.train_dataloader()
        assert isinstance(loader.sampler, WeightedRandomSampler)
        # num_samples must stay a multiple of effective_batch_size (1*8) for grad-accum alignment
        assert loader.sampler.num_samples % 8 == 0
        # the rare-class image must carry a larger weight than the common-class images
        weights = loader.sampler.weights
        assert weights[9] > weights[0]

    def test_plain_sampler_when_rfs_disabled(self) -> None:
        img_to_cats = {i: [0] for i in range(10)}  # below the min-batches threshold -> small branch
        dm = _make_datamodule(_FakeDataset(img_to_cats), rfs=False)
        loader = dm.train_dataloader()
        assert not isinstance(loader.sampler, WeightedRandomSampler)
        assert isinstance(loader.sampler, RandomSampler)


class TestTrainConfigRFSFields:
    """The RFS knobs must exist on TrainConfig so model.train(**kwargs) forwards them."""

    def test_defaults_off(self) -> None:
        cfg = TrainConfig(dataset_dir=".")
        assert cfg.rfs is False
        assert cfg.rfs_thresh == 0.001
        assert cfg.rfs_max == 20.0
        assert cfg.rfs_num_samples == 0

    def test_accepts_overrides(self) -> None:
        cfg = TrainConfig(dataset_dir=".", rfs=True, rfs_thresh=0.01, rfs_max=10.0, rfs_num_samples=5000)
        assert cfg.rfs is True
        assert cfg.rfs_thresh == 0.01
        assert cfg.rfs_max == 10.0
        assert cfg.rfs_num_samples == 5000
