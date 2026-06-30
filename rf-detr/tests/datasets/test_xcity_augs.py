# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the coupled day->night cross-city augmentation (xcity_augs)."""

from __future__ import annotations

import numpy as np
import pytest

import albumentations as alb

from rfdetr.datasets.transforms import AlbumentationsWrapper
from rfdetr.datasets.xcity_augs import CoupledDayNight, coupled_day_night


def _mid_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # mid-brightness colourful image so darkening/brightening is measurable
    return rng.integers(60, 200, size=(64, 80, 3), dtype=np.uint8)


def _luma(img_u8: np.ndarray) -> np.ndarray:
    x = img_u8.astype(np.float32)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _saturation(img_u8: np.ndarray) -> float:
    x = img_u8.astype(np.float32)
    mx = x.max(axis=2)
    mn = x.min(axis=2)
    return float((mx - mn).mean())


class TestCoupledDayNightCore:
    """The pure coupled function: one signed parameter s in [-1, 1] drives everything."""

    def test_s_zero_is_identity(self) -> None:
        img = _mid_image()
        out = coupled_day_night(img, 0.0, noise_std=0.0)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1  # only rounding

    def test_night_darkens_monotonically(self) -> None:
        img = _mid_image()
        m0 = _luma(coupled_day_night(img, 0.0, noise_std=0.0)).mean()
        m5 = _luma(coupled_day_night(img, 0.5, noise_std=0.0)).mean()
        m9 = _luma(coupled_day_night(img, 0.9, noise_std=0.0)).mean()
        assert m9 < m5 < m0

    def test_day_brightens(self) -> None:
        img = _mid_image()
        m0 = _luma(coupled_day_night(img, 0.0, noise_std=0.0)).mean()
        md = _luma(coupled_day_night(img, -0.9, noise_std=0.0)).mean()
        assert md > m0

    def test_night_crushes_shadows_more_than_highlights(self) -> None:
        # vertical gradient 0..255 so we have clear dark vs bright regions
        col = np.linspace(0, 255, 80, dtype=np.uint8)
        img = np.repeat(np.tile(col, (64, 1))[:, :, None], 3, axis=2)
        out = coupled_day_night(img, 0.9, noise_std=0.0, min_mean_luma=0.0, hard_floor=0.0)
        din = _luma(img)
        dout = _luma(out)
        dark = din < 64
        bright = din > 192
        drop_dark = (din[dark].mean() - dout[dark].mean()) / (din[dark].mean() + 1e-6)
        drop_bright = (din[bright].mean() - dout[bright].mean()) / (din[bright].mean() + 1e-6)
        assert drop_dark > drop_bright

    def test_saturation_drops_at_night(self) -> None:
        img = _mid_image()
        s0 = _saturation(coupled_day_night(img, 0.0, noise_std=0.0))
        s9 = _saturation(coupled_day_night(img, 0.9, noise_std=0.0))
        assert s9 < s0

    def test_darkening_cap_keeps_a_luma_floor(self) -> None:
        img = _mid_image()
        out = coupled_day_night(img, 0.95, noise_std=0.0, min_mean_luma=0.06, hard_floor=0.015)
        assert _luma(out).mean() >= 0.06 * 255 - 2.0
        assert out.min() >= 0.015 * 255 - 1.0

    def test_cap_does_not_brighten_a_naturally_dark_scene(self) -> None:
        dark = (_mid_image() // 12).astype(np.uint8)  # very dark image
        out = coupled_day_night(dark, 0.0, noise_std=0.0, min_mean_luma=0.06, hard_floor=0.0)
        # s=0 must be a no-op even though the scene is below the luma floor
        assert np.abs(out.astype(int) - dark.astype(int)).max() <= 1

    def test_warm_shifts_red_over_blue(self) -> None:
        img = _mid_image()
        out = coupled_day_night(img, 0.8, noise_std=0.0, warm=1.0, min_mean_luma=0.0, hard_floor=0.0)
        r_gain = out[..., 0].astype(float).mean() - img[..., 0].astype(float).mean()
        b_gain = out[..., 2].astype(float).mean() - img[..., 2].astype(float).mean()
        assert r_gain > b_gain  # warm cast lifts red relative to blue


class TestCoupledDayNightTransform:
    """The Albumentations ImageOnlyTransform wrapper + registration."""

    def test_registered_in_albumentations_namespace(self) -> None:
        assert getattr(alb, "CoupledDayNight", None) is CoupledDayNight

    def test_from_config_builds_one_not_skipped(self) -> None:
        wrappers = AlbumentationsWrapper.from_config({"CoupledDayNight": {"p": 1.0}})
        assert len(wrappers) == 1

    def test_is_pixel_level_not_geometric(self) -> None:
        wrappers = AlbumentationsWrapper.from_config({"CoupledDayNight": {"p": 1.0}})
        assert wrappers[0]._is_geometric is False

    def test_box_safe_through_wrapper(self) -> None:
        wrapper = AlbumentationsWrapper.from_config({"CoupledDayNight": {"p": 1.0}})[0]
        import torch
        from PIL import Image

        image = Image.fromarray(_mid_image())
        target = {
            "boxes": torch.tensor([[5.0, 6.0, 40.0, 50.0]]),
            "labels": torch.tensor([3]),
            "area": torch.tensor([35.0 * 44.0]),
            "iscrowd": torch.tensor([0]),
            "size": torch.tensor([64, 80]),
        }
        _, out = wrapper(image, target)
        assert torch.equal(out["boxes"], target["boxes"])
        assert torch.equal(out["labels"], target["labels"])

    def test_apply_preserves_shape_dtype(self) -> None:
        t = CoupledDayNight(p=1.0)
        out = t(image=_mid_image())["image"]
        assert out.shape == (64, 80, 3) and out.dtype == np.uint8

    def test_init_args_names_match_constructor(self) -> None:
        # get_transform_init_args_names must list every constructor arg except p, so the params
        # survive any (de)serialization that reads them. (The trainer builds via from_config, not
        # albumentations' to_dict/from_dict registry, so we check the invariant directly.)
        import inspect

        t = CoupledDayNight(p=0.7)
        ctor_args = set(inspect.signature(CoupledDayNight.__init__).parameters) - {"self", "p"}
        assert set(t.get_transform_init_args_names()) == ctor_args
