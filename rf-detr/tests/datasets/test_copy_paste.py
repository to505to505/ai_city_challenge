# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rare-class copy-paste augmentation (xcity_augs)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset
from rfdetr.datasets.xcity_augs import CopyPaste, RareInstanceBank, _feather_alpha


@pytest.fixture(scope="module")
def coco_train() -> CocoDetection:
    d = tempfile.mkdtemp()
    generate_coco_dataset(
        d, num_images=40, img_size=128, max_objects=6,
        split_ratios=DatasetSplitRatios(train=1.0, val=0.0, test=0.0),
    )
    train = os.path.join(d, "train")
    return CocoDetection(
        train, os.path.join(train, "_annotations.coco.json"), transforms=None, remap_category_ids=True
    )


class TestFeatherAlpha:
    def test_interior_opaque_edges_fade(self) -> None:
        a = _feather_alpha(30, 40, border=4)
        assert a.shape == (30, 40) and a.dtype == np.uint8
        assert a[15, 20] == 255  # interior fully opaque
        assert a[0, 20] < 255 and a[29, 20] < 255  # top/bottom edges fade
        assert a[15, 0] < 255 and a[15, 39] < 255  # left/right edges fade


class TestRareInstanceBank:
    def test_contains_only_requested_rare_labels(self, coco_train: CocoDetection) -> None:
        rare = {1, 2}
        bank = RareInstanceBank(coco_train.coco, coco_train.root, rare, cat2label=coco_train.cat2label)
        assert len(bank) > 0
        assert all(label in rare for _, _, label in bank.entries)

    def test_empty_when_no_rare_present(self, coco_train: CocoDetection) -> None:
        bank = RareInstanceBank(coco_train.coco, coco_train.root, {999}, cat2label=coco_train.cat2label)
        assert len(bank) == 0

    def test_min_box_filters_small(self, coco_train: CocoDetection) -> None:
        big = RareInstanceBank(coco_train.coco, coco_train.root, {0, 1, 2}, cat2label=coco_train.cat2label, min_box=1)
        small_only = RareInstanceBank(
            coco_train.coco, coco_train.root, {0, 1, 2}, cat2label=coco_train.cat2label, min_box=10_000
        )
        assert len(small_only) == 0 and len(big) > 0

    def test_sample_returns_rgba_crop_and_rare_label(self, coco_train: CocoDetection) -> None:
        bank = RareInstanceBank(coco_train.coco, coco_train.root, {0, 1, 2}, cat2label=coco_train.cat2label)
        crop, label = bank.sample(np.random.default_rng(0))
        assert crop.ndim == 3 and crop.shape[2] == 4 and crop.dtype == np.uint8  # RGBA
        assert label in {0, 1, 2}


class TestCopyPaste:
    def test_empty_bank_is_noop(self, coco_train: CocoDetection) -> None:
        bank = RareInstanceBank(coco_train.coco, coco_train.root, {999}, cat2label=coco_train.cat2label)
        cp = CopyPaste(bank, max_n=3, p=1.0)
        img, target = coco_train[0]
        n0 = len(target["boxes"])
        img2, target2 = cp(img, target)
        assert len(target2["boxes"]) == n0

    def test_paste_increases_box_count_and_appends_rare_labels(self, coco_train: CocoDetection) -> None:
        rare = {0, 1, 2}
        bank = RareInstanceBank(coco_train.coco, coco_train.root, rare, cat2label=coco_train.cat2label)
        cp = CopyPaste(bank, max_n=4, p=1.0, seed=0)
        img, target = coco_train[1]
        n0 = len(target["boxes"])
        _, out = cp(img, target)
        assert len(out["boxes"]) > n0
        added = out["labels"][n0:]
        assert all(int(label) in rare for label in added)

    def test_zero_max_n_is_identity(self, coco_train: CocoDetection) -> None:
        bank = RareInstanceBank(coco_train.coco, coco_train.root, {0, 1, 2}, cat2label=coco_train.cat2label)
        cp = CopyPaste(bank, max_n=0, p=1.0)
        img, target = coco_train[2]
        n0 = len(target["boxes"])
        _, out = cp(img, target)
        assert len(out["boxes"]) == n0

    def test_probability_zero_is_identity(self, coco_train: CocoDetection) -> None:
        bank = RareInstanceBank(coco_train.coco, coco_train.root, {0, 1, 2}, cat2label=coco_train.cat2label)
        cp = CopyPaste(bank, max_n=4, p=0.0)
        img, target = coco_train[3]
        n0 = len(target["boxes"])
        _, out = cp(img, target)
        assert len(out["boxes"]) == n0

    def test_pasted_boxes_are_inbounds_float32_xyxy(self, coco_train: CocoDetection) -> None:
        bank = RareInstanceBank(coco_train.coco, coco_train.root, {0, 1, 2}, cat2label=coco_train.cat2label)
        cp = CopyPaste(bank, max_n=4, p=1.0, seed=1)
        img, target = coco_train[4]
        w, h = img.size
        n0 = len(target["boxes"])
        out_img, out = cp(img, target)
        assert out["boxes"].dtype == torch.float32
        new = out["boxes"][n0:]
        for x1, y1, x2, y2 in new.tolist():
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        # per-instance fields stay aligned with boxes
        assert len(out["labels"]) == len(out["boxes"]) == len(out["area"]) == len(out["iscrowd"])
