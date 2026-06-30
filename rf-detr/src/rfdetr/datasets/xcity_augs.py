# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Cross-city domain-generalization augmentations (custom, beyond stock Albumentations).

This module adds transforms that the requested cross-city stack needs but Albumentations does not
ship as a single coupled operation:

* :class:`CoupledDayNight` — one continuous parameter ``s`` drives a *physically coherent*
  day -> evening -> night shift (and the symmetric harsh-noon / overcast direction): gamma
  shadow-crush, brightness/wash, saturation loss, warm/cool colour temperature and sensor noise are
  all tied to the same ``s`` so the result transfers (a dark frame is also noisier, desaturated and
  colour-cast). A content-aware darkening cap keeps small/distant annotated objects from vanishing
  into pure black.

All transforms here are :class:`~albumentations.ImageOnlyTransform` (box-safe — they never move
bounding boxes) and are **registered into the ``albumentations`` namespace at import** so the
trainer's config-driven ``AlbumentationsWrapper.from_config`` (which resolves transforms via
``getattr(albumentations, name)``) can build them by name, and so DataLoader workers that re-import
the augmentation stack re-register them.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import albumentations as alb
    from albumentations.core.transforms_interface import ImageOnlyTransform
except ImportError:  # pragma: no cover - albumentations is a hard dependency at train time
    alb = None  # type: ignore[assignment]
    ImageOnlyTransform = object  # type: ignore[assignment,misc]


def _luma(x: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a float RGB image in ``[0, 1]`` (``H x W`` output)."""
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def coupled_day_night(
    img: np.ndarray,
    s: float,
    *,
    gamma_max: float = 2.2,
    sat_min: float = 0.6,
    warm: float = 1.0,
    noise_std: float = 0.0,
    min_mean_luma: float = 0.06,
    hard_floor: float = 0.015,
    ir: bool = False,
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Apply the coupled day<->night shift to a ``uint8`` RGB image.

    A single signed parameter ``s in [-1, 1]`` drives everything so the result is physically
    coherent: ``s > 0`` is night (``t = s``) — gamma darkening with shadow-crush, desaturation,
    colour-temperature cast and noise all scale with ``t``; ``s < 0`` is the symmetric harsh-noon /
    overcast direction (``d = -s``) — brighten and wash out contrast. ``s == 0`` is the identity.

    Args:
        img: ``H x W x 3`` ``uint8`` RGB image.
        s: Signed strength in ``[-1, 1]`` (``>0`` night, ``<0`` day, ``0`` identity).
        gamma_max: Gamma at full night (``t=1``); darkening exponent ``lerp(1, gamma_max, t)``.
        sat_min: Saturation multiplier at full night (``lerp(1, sat_min, t)``).
        warm: Colour-temperature direction at night (``>0`` warm/sodium, ``<0`` cool/LED).
        noise_std: Std (in ``[0, 1]`` space) of additive Gaussian sensor noise, scaled by ``t``.
        min_mean_luma: Darkening cap — the night result's mean luma is lifted back toward the
            original (never above it) until it reaches this floor, so small objects stay visible.
        hard_floor: Per-pixel minimum (in ``[0, 1]``) clamped at night so nothing is pure black.
        ir: When ``True`` (night only), emulate an IR/low-light mono camera (grayscale + cool cast).
        rng: NumPy generator for the noise term (a fresh default generator when ``None``).

    Returns:
        The augmented ``uint8`` RGB image, same shape as *img*.
    """
    rng = rng if rng is not None else np.random.default_rng()
    x = img.astype(np.float32) / 255.0
    orig = x.copy()
    t = max(s, 0.0)
    d = max(-s, 0.0)

    # 1. Tone. Night: gamma > 1 darkens and crushes shadows harder than highlights.
    #    Day: gamma < 1 brightens, then pull contrast toward mid-grey (overcast wash).
    if t > 0.0:
        x = np.power(x, 1.0 + (gamma_max - 1.0) * t)
    if d > 0.0:
        x = np.power(x, 1.0 - 0.4 * d)
        x = 0.5 + (x - 0.5) * (1.0 - 0.3 * d)

    if t > 0.0:
        if ir:
            # IR / low-light mono: collapse to luma, then a slight cool cast.
            gray = _luma(x)[..., None]
            x = np.repeat(gray, 3, axis=2)
            cool = 0.10 * t
            x[..., 0] *= 1.0 - cool
            x[..., 2] *= 1.0 + cool
        else:
            # 2. Saturation loss at night (blend toward luma).
            sat = 1.0 + (sat_min - 1.0) * t
            lum = _luma(x)[..., None]
            x = lum + (x - lum) * sat
            # 3. Colour temperature: warm (sodium) or cool (LED), scaled by t.
            amt = 0.15 * t * warm
            x[..., 0] *= 1.0 + amt
            x[..., 2] *= 1.0 - amt
        # 4. Sensor noise rises with night.
        if noise_std > 0.0:
            x = x + rng.normal(0.0, noise_std * t, size=x.shape).astype(np.float32)

    x = np.clip(x, 0.0, 1.0)

    # 5. Darkening cap (night only): lift the result back TOWARD the original (never above it) until
    #    its mean luma reaches the floor, so annotated small/distant objects never disappear.
    if t > 0.0:
        mean_out = float(_luma(x).mean())
        mean_orig = float(_luma(orig).mean())
        if mean_out < min_mean_luma and mean_orig > mean_out:
            alpha = (min_mean_luma - mean_out) / (mean_orig - mean_out)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            x = x * (1.0 - alpha) + orig * alpha
        if hard_floor > 0.0:
            x = np.maximum(x, hard_floor)

    return np.clip(x * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


class CoupledDayNight(ImageOnlyTransform):
    """Coupled continuous day -> evening -> night (and symmetric noon) augmentation.

    One per-call parameter ``s`` (sampled from a ``night_bias``-weighted distribution) drives gamma
    darkening, desaturation, colour temperature and noise together (see :func:`coupled_day_night`),
    plus an occasional IR night mode and a deep-night motion blur. Box-safe (image-only).

    Args:
        p: Probability of applying the transform.
        night_bias: Probability that a sampled frame is pushed toward night (vs the day/noon side).
        gamma_max: Maximum night gamma (darkening exponent at ``t=1``).
        sat_min: Saturation multiplier at full night.
        noise_std: Gaussian sensor-noise std (in ``[0, 1]``) at full night.
        min_mean_luma: Mean-luma darkening cap (see :func:`coupled_day_night`).
        hard_floor: Per-pixel black floor at night.
        ir_prob: Probability of IR/mono night mode (deep night only).
        motion_blur_max: Max odd kernel for the occasional deep-night motion blur (``0`` disables).
    """

    def __init__(
        self,
        p: float = 0.6,
        night_bias: float = 0.65,
        gamma_max: float = 2.2,
        sat_min: float = 0.6,
        noise_std: float = 0.04,
        min_mean_luma: float = 0.06,
        hard_floor: float = 0.015,
        ir_prob: float = 0.10,
        motion_blur_max: int = 7,
    ) -> None:
        super().__init__(p=p)
        if not 0.0 <= night_bias <= 1.0:
            raise ValueError(f"night_bias must be in [0, 1], got {night_bias}")
        if gamma_max < 1.0:
            raise ValueError(f"gamma_max must be >= 1.0, got {gamma_max}")
        self.night_bias = night_bias
        self.gamma_max = gamma_max
        self.sat_min = sat_min
        self.noise_std = noise_std
        self.min_mean_luma = min_mean_luma
        self.hard_floor = hard_floor
        self.ir_prob = ir_prob
        self.motion_blur_max = motion_blur_max

    def get_params(self) -> Dict[str, Any]:
        """Sample the coupling parameters for one call (independent of the image)."""
        rng = np.random
        u = rng.random()
        s = rng.random() if u < self.night_bias else -rng.random()
        warm = 1.0 if rng.random() < 0.7 else -1.0
        ir = bool(s > 0.5 and rng.random() < self.ir_prob)
        mb = 0
        if s > 0.6 and self.motion_blur_max >= 3 and rng.random() < 0.3:
            k_max = self.motion_blur_max | 1
            mb = int(rng.choice(range(3, k_max + 1, 2)))
        return {"s": float(s), "warm": float(warm), "ir": ir, "mb_ksize": int(mb)}

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the coupled shift (and optional deep-night motion blur) to *img*."""
        out = coupled_day_night(
            img,
            params.get("s", 0.0),
            gamma_max=self.gamma_max,
            sat_min=self.sat_min,
            warm=params.get("warm", 1.0),
            noise_std=self.noise_std,
            min_mean_luma=self.min_mean_luma,
            hard_floor=self.hard_floor,
            ir=params.get("ir", False),
        )
        k = int(params.get("mb_ksize", 0))
        if k >= 3:
            import cv2

            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)
        return out

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Init args (excluding ``p``) for serialization."""
        return (
            "night_bias",
            "gamma_max",
            "sat_min",
            "noise_std",
            "min_mean_luma",
            "hard_floor",
            "ir_prob",
            "motion_blur_max",
        )


# ---------------------------------------------------------------------------
# Rare-class copy-paste (Tier 3). NOT an Albumentations transform: it ADDS boxes,
# so it runs as a (PIL image, target) hook in CocoDetection.__getitem__ BEFORE the
# Albumentations pipeline (so pasted boxes get warped by the geometry augs).
# ---------------------------------------------------------------------------

# Default rare contiguous labels for eccv-cross-city (CLASS_NAMES order): Single Truck(2),
# Combo Truck(3), Heavy Duty(4), Trailer(5), Motorcycle(6), Bicycle(7), Van(8). Car(0)/Pickup(1)/
# Person(9) are common and excluded.
DEFAULT_RARE_LABELS: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)


def _feather_alpha(h: int, w: int, border: int = 3) -> np.ndarray:
    """Feathered alpha matte: ``255`` interior, linearly fading over a ``border``-px ring.

    The soft edge prevents a hard rectangular seam that the model could latch onto as a paste cue.
    """
    a = np.full((h, w), 255, dtype=np.float32)
    b = max(1, int(border))
    for i in range(b):
        v = 255.0 * (i + 1) / (b + 1)
        a[i, :] = np.minimum(a[i, :], v)
        a[h - 1 - i, :] = np.minimum(a[h - 1 - i, :], v)
        a[:, i] = np.minimum(a[:, i], v)
        a[:, w - 1 - i] = np.minimum(a[:, w - 1 - i], v)
    return a.astype(np.uint8)


def _resize_rgba(crop: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Resize an ``H x W x 4`` RGBA crop to ``(th, tw)`` (area interpolation)."""
    import cv2

    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)


class RareInstanceBank:
    """A bank of rare-class instance crops harvested from a COCO dataset.

    Built from a pycocotools ``COCO`` object restricted to ``rare_labels`` (contiguous labels after
    ``cat2label`` remap). Crops are loaded lazily and cached. Building the bank from the *train-split*
    COCO is leakage-free by construction — held-out val cameras are physically absent from that file.

    Args:
        coco: pycocotools ``COCO`` API object (e.g. ``CocoDetection.coco``).
        img_folder: Directory holding the images referenced by ``coco`` (``CocoDetection.root``).
        rare_labels: Contiguous class labels to harvest.
        cat2label: Optional sparse-``category_id`` -> contiguous-label map (``CocoDetection.cat2label``).
        min_box: Skip instances smaller than this (px) on either side.
        max_per_class: Cap entries per class (bounds memory).
    """

    def __init__(
        self,
        coco: Any,
        img_folder: "str | Path",
        rare_labels: Any,
        cat2label: Optional[Dict[int, int]] = None,
        min_box: int = 12,
        max_per_class: int = 300,
    ) -> None:
        self.coco = coco
        self.img_folder = Path(img_folder)
        self.rare = {int(x) for x in rare_labels}
        self.cat2label = cat2label
        self.entries: List[Tuple[int, List[float], int]] = []
        self._crop_cache: Dict[int, Optional[np.ndarray]] = {}

        per_class: "Counter[int]" = Counter()
        for ann_id in coco.getAnnIds():
            ann = coco.loadAnns([ann_id])[0]
            if ann.get("iscrowd", 0) == 1:
                continue
            cat_id = ann["category_id"]
            label = int(cat2label.get(cat_id, cat_id)) if cat2label is not None else int(cat_id)
            if label not in self.rare:
                continue
            x, y, w, h = ann["bbox"]
            if w < min_box or h < min_box:
                continue
            if per_class[label] >= max_per_class:
                continue
            per_class[label] += 1
            self.entries.append((int(ann["image_id"]), [float(x), float(y), float(w), float(h)], label))

    def __len__(self) -> int:
        return len(self.entries)

    def _load_crop(self, idx: int) -> Optional[np.ndarray]:
        """Load (and cache) the feathered RGBA crop for entry ``idx``; ``None`` on any read failure."""
        if idx in self._crop_cache:
            return self._crop_cache[idx]
        from PIL import Image

        img_id, (x, y, w, h), _ = self.entries[idx]
        crop: Optional[np.ndarray] = None
        try:
            info = self.coco.loadImgs([img_id])[0]
            with Image.open(self.img_folder / info["file_name"]) as im:
                arr = np.asarray(im.convert("RGB"))
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(arr.shape[1], int(round(x + w))), min(arr.shape[0], int(round(y + h)))
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                rgb = arr[y0:y1, x0:x1]
                ch, cw = rgb.shape[:2]
                alpha = _feather_alpha(ch, cw, border=max(2, min(ch, cw) // 10))
                crop = np.dstack([rgb, alpha])
        except Exception:  # noqa: BLE001 - a single unreadable crop must never crash training
            crop = None
        self._crop_cache[idx] = crop
        return crop

    def sample(self, rng: "np.random.Generator") -> Optional[Tuple[np.ndarray, int]]:
        """Return ``(rgba_crop, label)`` for a random loadable rare instance, or ``None``."""
        if not self.entries:
            return None
        for _ in range(5):
            idx = int(rng.integers(0, len(self.entries)))
            crop = self._load_crop(idx)
            if crop is not None:
                return crop, self.entries[idx][2]
        return None


class CopyPaste:
    """Paste rare-class instances into a training image (and append their boxes).

    Operates on the ``(PIL image, target)`` pair produced by ``CocoDetection.prepare`` — boxes are
    absolute ``xyxy`` here, so the pasted boxes ride the same downstream geometry + normalization.

    Args:
        bank: A :class:`RareInstanceBank`.
        max_n: Max instances pasted per image (``k ~ U(1, max_n)`` when applied).
        p: Probability that an image receives any pastes.
        scale_frac: Pasted-object target height as a fraction of image height ``~U(lo, hi)``.
        seed: Optional RNG seed (tests); ``None`` -> fresh per-worker stream.
    """

    def __init__(
        self,
        bank: RareInstanceBank,
        max_n: int = 3,
        p: float = 0.5,
        scale_frac: Tuple[float, float] = (0.06, 0.25),
        seed: Optional[int] = None,
    ) -> None:
        self.bank = bank
        self.max_n = max_n
        self.p = p
        self.scale_frac = scale_frac
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: Any, target: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        import torch
        from PIL import Image

        if len(self.bank) == 0 or self.max_n <= 0 or self.rng.random() >= self.p:
            return img, target

        arr = np.asarray(img.convert("RGB")).copy()
        height, width = arr.shape[:2]
        n = int(self.rng.integers(1, self.max_n + 1))
        new_boxes: List[List[float]] = []
        new_labels: List[int] = []

        for _ in range(n):
            sampled = self.bank.sample(self.rng)
            if sampled is None:
                continue
            crop, label = sampled
            ch, cw = crop.shape[:2]
            th = int(self.rng.uniform(*self.scale_frac) * height)
            th = max(8, min(th, height - 2))
            tw = max(8, int(cw * (th / ch)))
            if tw > width - 2 or width - tw <= 0 or height - th <= 0:
                continue
            crop_r = _resize_rgba(crop, th, tw)
            x0 = int(self.rng.integers(0, width - tw))
            y_lo, y_hi = height // 3, height - th  # lower 2/3 (ground bias)
            y0 = int(self.rng.integers(y_lo, y_hi)) if y_hi > y_lo else max(0, y_hi)
            rgb = crop_r[..., :3].astype(np.float32)
            alpha = crop_r[..., 3:4].astype(np.float32) / 255.0
            region = arr[y0:y0 + th, x0:x0 + tw].astype(np.float32)
            arr[y0:y0 + th, x0:x0 + tw] = (rgb * alpha + region * (1.0 - alpha)).astype(np.uint8)
            new_boxes.append([float(x0), float(y0), float(x0 + tw), float(y0 + th)])
            new_labels.append(int(label))

        if not new_boxes:
            return img, target

        out = dict(target)
        nb = torch.as_tensor(new_boxes, dtype=torch.float32)
        boxes = target.get("boxes")
        out["boxes"] = torch.cat([boxes, nb], dim=0) if boxes is not None and len(boxes) else nb
        nl = torch.as_tensor(new_labels, dtype=torch.long)
        labels = target.get("labels")
        out["labels"] = torch.cat([labels, nl], dim=0) if labels is not None and len(labels) else nl
        if "area" in target:
            na = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
            out["area"] = torch.cat([target["area"], na], dim=0)
        if "iscrowd" in target:
            zeros = torch.zeros(len(new_boxes), dtype=target["iscrowd"].dtype)
            out["iscrowd"] = torch.cat([target["iscrowd"], zeros], dim=0)
        return Image.fromarray(arr), out


# ---------------------------------------------------------------------------
# FACT Fourier amplitude mix (Tier 4, flag-gated). Style randomization that swaps a reference
# image's LOW-FREQUENCY amplitude into the source while PRESERVING phase -> content/boxes unchanged.
# ---------------------------------------------------------------------------

_REF_DIR: Optional[str] = None
_REF_POOL: Optional[List[np.ndarray]] = None
_REF_LOCK = threading.Lock()


def set_reference_image_dir(path: Optional[str]) -> None:
    """Point the FACT reference pool at an image directory (resets the cached pool)."""
    global _REF_DIR, _REF_POOL
    _REF_DIR = path
    _REF_POOL = None


def _ensure_reference_image_dir(path: Optional[str]) -> None:
    """Set the FACT reference dir only if it changed (preserves the cached pool).

    Called from ``CocoDetection.__getitem__`` so EACH DataLoader worker (fork OR spawn) sets its own
    process-global reference dir before :class:`FourierAmplitudeMix` runs — making the reference pool
    worker-safe without resetting (and re-thumbnailing) the pool on every item.
    """
    if _REF_DIR != path:
        set_reference_image_dir(path)


def _get_reference_pool(cap: int = 64, thumb: int = 256) -> List[np.ndarray]:
    """Lazily build (once per worker) a small thumbnail pool from ``_REF_DIR``; ``[]`` if unset/empty.

    Network-free (reads local dataset images) and memory-cheap (cap thumbnails). Thread-safe. When the
    directory holds more than *cap* images the *cap* are sampled RANDOMLY across the whole set (not the
    alphabetically-first ones, which on a video dataset would all be one camera) so the pool represents
    the directory's full style range — important for FDA toward the target city.
    """
    global _REF_POOL
    if _REF_POOL is not None:
        return _REF_POOL
    with _REF_LOCK:
        if _REF_POOL is not None:
            return _REF_POOL
        pool: List[np.ndarray] = []
        if _REF_DIR:
            import glob
            import os

            from PIL import Image

            paths: List[str] = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                paths += glob.glob(os.path.join(_REF_DIR, "**", ext), recursive=True)
            paths = sorted(paths)
            if len(paths) > cap:
                idx = np.random.default_rng().choice(len(paths), size=cap, replace=False)
                paths = [paths[i] for i in sorted(idx.tolist())]
            for path in paths:
                try:
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((thumb, thumb))
                        pool.append(np.asarray(im))
                except Exception:  # noqa: BLE001 - skip unreadable reference images
                    continue
        _REF_POOL = pool
    return _REF_POOL


def _fourier_amplitude_mix(img: np.ndarray, ref: np.ndarray, beta: float = 0.05, lam: float = 0.5) -> np.ndarray:
    """Mix the low-frequency amplitude of *ref* into *img* while keeping *img*'s phase.

    Args:
        img: ``H x W x 3`` ``uint8`` source.
        ref: ``H x W x 3`` ``uint8`` reference (same size as *img*).
        beta: Half-size of the centered low-frequency window as a fraction of each side.
        lam: Mix weight; ``0`` keeps the source amplitude, ``1`` fully adopts the reference's.

    Returns:
        ``uint8`` image with content/edges (phase) intact and only low-freq style shifted.
    """
    img_f = img.astype(np.float32)
    ref_f = ref.astype(np.float32)
    h, w = img.shape[:2]
    bh, bw = max(1, int(round(h * beta))), max(1, int(round(w * beta)))
    cy, cx = h // 2, w // 2
    y0, y1, x0, x1 = cy - bh, cy + bh + 1, cx - bw, cx + bw + 1
    out = np.empty_like(img_f)
    for c in range(3):
        f_img = np.fft.fftshift(np.fft.fft2(img_f[..., c]))
        f_ref = np.fft.fftshift(np.fft.fft2(ref_f[..., c]))
        amp, phase = np.abs(f_img), np.angle(f_img)
        amp[y0:y1, x0:x1] = (1.0 - lam) * amp[y0:y1, x0:x1] + lam * np.abs(f_ref)[y0:y1, x0:x1]
        recon = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * phase)))
        out[..., c] = np.real(recon)
    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)


class FourierAmplitudeMix(ImageOnlyTransform):
    """FACT-style low-frequency amplitude randomization (phase-preserving, box-safe).

    Args:
        p: Probability of applying the transform.
        beta: Low-frequency window half-size as a fraction of each side (must be in ``(0, 0.2]``).
        lambda_max: Max mix weight; ``lam ~ U(0, lambda_max)`` per call (must be in ``[0, 1]``).
    """

    def __init__(self, p: float = 0.3, beta: float = 0.05, lambda_max: float = 0.3) -> None:
        super().__init__(p=p)
        if not 0.0 < beta <= 0.2:
            raise ValueError(f"beta must be in (0, 0.2], got {beta}")
        if not 0.0 <= lambda_max <= 1.0:
            raise ValueError(f"lambda_max must be in [0, 1], got {lambda_max}")
        self.beta = beta
        self.lambda_max = lambda_max

    def get_params(self) -> Dict[str, Any]:
        return {"lam": float(np.random.random() * self.lambda_max)}

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        pool = _get_reference_pool()
        if not pool:
            return img  # no reference available -> identity (network-isolated, dir unset)
        import cv2

        ref = pool[np.random.randint(len(pool))]
        ref_r = cv2.resize(ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        return _fourier_amplitude_mix(img, ref_r, beta=self.beta, lam=params.get("lam", 0.0))

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("beta", "lambda_max")


def _register() -> None:
    """Register the custom transforms into the ``albumentations`` namespace.

    ``AlbumentationsWrapper.from_config`` resolves transforms via ``getattr(albumentations, name)``,
    so custom names must live on the module. Idempotent and called at import, so DataLoader workers
    that re-import this module (fork or spawn) re-register before any ``from_config`` runs.
    """
    if alb is None:
        return
    setattr(alb, "CoupledDayNight", CoupledDayNight)
    setattr(alb, "FourierAmplitudeMix", FourierAmplitudeMix)


_register()
