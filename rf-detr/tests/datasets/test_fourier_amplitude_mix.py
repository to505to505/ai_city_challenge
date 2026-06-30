# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for FACT Fourier amplitude-mix augmentation (xcity_augs)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import albumentations as alb

from rfdetr.datasets.transforms import AlbumentationsWrapper
from rfdetr.datasets.xcity_augs import (
    FourierAmplitudeMix,
    _fourier_amplitude_mix,
    _get_reference_pool,
    set_reference_image_dir,
)


def _img(seed: int, h: int = 48, w: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(h, w, 3), dtype=np.uint8)


class TestFourierCore:
    def test_shape_dtype_preserved_non_square(self) -> None:
        out = _fourier_amplitude_mix(_img(0, 48, 64), _img(1, 48, 64), beta=0.05, lam=0.5)
        assert out.shape == (48, 64, 3) and out.dtype == np.uint8

    def test_lambda_zero_is_identity(self) -> None:
        img = _img(0)
        out = _fourier_amplitude_mix(img, _img(1), beta=0.05, lam=0.0)
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1

    def test_larger_lambda_changes_image_more(self) -> None:
        # lam controls mix strength: bigger lam -> the (phase-preserved) output departs further from
        # the source. (Re-FFT'ing the uint8 output to read window amplitude is unreliable after the
        # real()+clip projection, so we measure the actual pixel effect instead.)
        img, ref = _img(0), _img(1)
        d_lo = np.abs(_fourier_amplitude_mix(img, ref, beta=0.05, lam=0.2).astype(int) - img.astype(int)).mean()
        d_hi = np.abs(_fourier_amplitude_mix(img, ref, beta=0.05, lam=0.8).astype(int) - img.astype(int)).mean()
        assert d_hi > d_lo > 0


class TestFourierTransform:
    def test_registered_and_builds_via_from_config(self) -> None:
        assert getattr(alb, "FourierAmplitudeMix", None) is FourierAmplitudeMix
        assert len(AlbumentationsWrapper.from_config({"FourierAmplitudeMix": {"p": 1.0}})) == 1

    def test_is_pixel_level_box_safe(self) -> None:
        w = AlbumentationsWrapper.from_config({"FourierAmplitudeMix": {"p": 1.0}})[0]
        assert w._is_geometric is False

    def test_bad_params_raise(self) -> None:
        with pytest.raises(ValueError):
            FourierAmplitudeMix(beta=0.9)  # window ratio must be small
        with pytest.raises(ValueError):
            FourierAmplitudeMix(lambda_max=2.0)  # mix weight must be in [0,1]

    def test_empty_pool_is_passthrough(self) -> None:
        set_reference_image_dir(None)
        import rfdetr.datasets.xcity_augs as m
        m._REF_POOL = None  # reset cache
        t = FourierAmplitudeMix(p=1.0)
        img = _img(3)
        out = t(image=img)["image"]
        assert np.array_equal(out, img)  # no reference -> identity

    def test_pool_seeds_from_dir(self) -> None:
        d = tempfile.mkdtemp()
        from PIL import Image
        for i in range(3):
            Image.fromarray(_img(i, 70, 90)).save(os.path.join(d, f"r{i}.png"))
        set_reference_image_dir(d)
        import rfdetr.datasets.xcity_augs as m
        m._REF_POOL = None
        pool = _get_reference_pool()
        assert len(pool) == 3
        assert _get_reference_pool() is pool  # cached on second call


class TestReferenceDirWorkerSafety:
    """The dir must (re)assert inside each worker without rebuilding the pool every item."""

    def test_ensure_preserves_pool_when_dir_unchanged(self) -> None:
        import rfdetr.datasets.xcity_augs as m
        from rfdetr.datasets.xcity_augs import _ensure_reference_image_dir

        d = tempfile.mkdtemp()
        from PIL import Image
        Image.fromarray(_img(0, 40, 40)).save(os.path.join(d, "a.png"))
        set_reference_image_dir(d)
        m._REF_POOL = None
        pool = _get_reference_pool()
        _ensure_reference_image_dir(d)  # same dir -> must NOT reset the cached pool
        assert _get_reference_pool() is pool
        _ensure_reference_image_dir(None)  # different -> resets
        assert m._REF_DIR is None

    def test_builder_wires_train_ref_dir_and_getitem_reasserts(self) -> None:
        # Simulates a fresh DataLoader worker: build the train dataset (main process sets the dir),
        # then wipe the process global and confirm __getitem__ re-asserts it -> pool populates.
        import rfdetr.datasets.xcity_augs as m
        from types import SimpleNamespace

        from rfdetr.datasets.coco import build_roboflow_from_coco
        from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset

        d = tempfile.mkdtemp()
        generate_coco_dataset(
            d, num_images=15, img_size=128, max_objects=4,
            split_ratios=DatasetSplitRatios(train=1.0, val=0.0, test=0.0),
        )
        args = SimpleNamespace(
            dataset_dir=d, square_resize_div_64=True, letterbox=False, segmentation_head=False,
            multi_scale=False, expanded_scales=False, do_random_resize_via_padding=False,
            patch_size=16, num_windows=2, augmentation_backend="cpu", copy_paste=False,
            aug_config={"FourierAmplitudeMix": {"p": 1.0}},
        )
        ds = build_roboflow_from_coco("train", args, 256)
        assert ds._fourier_ref_dir is not None
        set_reference_image_dir(None)  # wipe global as a freshly-(re)imported worker would have it
        m._REF_POOL = None
        _ = ds[0]  # __getitem__ must re-assert the dir inside the (would-be) worker
        assert len(_get_reference_pool()) > 0

    def test_ref_split_test_points_to_test_images(self) -> None:
        # --fourier-ref-split=test must source references from the TEST image folder (FDA).
        import os
        from types import SimpleNamespace

        from rfdetr.datasets.coco import build_roboflow_from_coco
        from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset

        d = tempfile.mkdtemp()
        generate_coco_dataset(
            d, num_images=24, img_size=128, max_objects=4,
            split_ratios=DatasetSplitRatios(train=0.6, val=0.1, test=0.3),
        )
        args = SimpleNamespace(
            dataset_dir=d, square_resize_div_64=True, letterbox=False, segmentation_head=False,
            multi_scale=False, expanded_scales=False, do_random_resize_via_padding=False,
            patch_size=16, num_windows=2, augmentation_backend="cpu", copy_paste=False,
            aug_config={"FourierAmplitudeMix": {"p": 1.0}}, fourier_ref_split="test",
        )
        ds = build_roboflow_from_coco("train", args, 256)
        assert os.path.basename(ds._fourier_ref_dir.rstrip("/")) == "test"  # not the train fallback
