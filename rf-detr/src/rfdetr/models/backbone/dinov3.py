# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""DINOv3 ViT backbone for RF-DETR (prototype).

Mirrors the DinoV2 encoder INTERFACE (``forward`` -> list of (B, C, H, W) feature maps at
``out_feature_indexes``; ``_out_feature_channels``) but uses transformers' ``DINOv3ViTBackbone``.

Differences from the DinoV2 path, on purpose:
  * NO windowed-attention variant. RF-DETR's windowing is tied to DINOv2's LEARNED positional
    embeddings; DINOv3 uses RoPE (rotary) positions, so window batch-reshaping would break
    positional coherence. We run vanilla full attention (more VRAM at high res — pair with a
    smaller variant or lower resolution on a 16 GB GPU).
  * NO positional-embedding interpolation / ``export()`` PE surgery. RoPE handles arbitrary
    resolution natively, so there is no learned PE grid to resize and ``num_windows`` is 1.

Weights: DINOv3 checkpoints on the HF Hub are LICENSE-GATED (accept the DINOv3 license + provide
an HF token). With ``load_weights=False`` the backbone is random-initialized — enough to verify the
RF-DETR plumbing (shapes), but not for real training.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DINOv3ViTBackbone, DINOv3ViTConfig

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# DINOv3 ViT sizes -> embedding width (same convention as DinoV2's size_to_width).
size_to_width = {"small": 384, "base": 768, "large": 1024}

# HF repo basenames (LVD-1689M pretraining). Gated on the Hub — but if a copy is bundled under the
# project's weights/<basename>/ (config.json + model.safetensors) we load that offline instead.
size_to_repo = {
    "small": "dinov3-vits16-pretrain-lvd1689m",
    "base": "dinov3-vitb16-pretrain-lvd1689m",
    "large": "dinov3-vitl16-pretrain-lvd1689m",
}


def _resolve_model_path(size: str) -> str:
    """Prefer a locally bundled HF model dir (offline); else the gated HF repo id."""
    basename = size_to_repo[size]
    env = os.environ.get("RFDETR_DINOV3_DIR")
    if env and (Path(env) / "config.json").exists():
        return env
    # weights/<basename> lives at the project root — same relative layout locally and on the
    # platform (/opt/recipe/weights). dinov3.py is rf-detr/src/rfdetr/models/backbone/dinov3.py.
    here = Path(__file__).resolve()
    for up in (5, 4, 6):
        cand = here.parents[up] / "weights" / basename
        if (cand / "config.json").exists():
            return str(cand)
    return f"facebook/{basename}"  # gated fallback

# Architecture presets for RANDOM-INIT (offline / no gated weights) — standard ViT-S/B/L @ patch16.
size_to_arch = {
    "small": dict(hidden_size=384, intermediate_size=1536, num_hidden_layers=12, num_attention_heads=6),
    "base": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12),
    "large": dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16),
}


class DinoV3(nn.Module):
    """DINOv3 ViT encoder exposing the DinoV2 backbone interface used by RF-DETR."""

    def __init__(
        self,
        shape: tuple[int, int] = (896, 896),
        out_feature_indexes: list[int] = [3, 6, 9, 12],
        size: str = "large",
        patch_size: int = 16,
        load_weights: bool = True,
        hf_name: str | None = None,
        **_ignored,  # swallow DinoV2-only kwargs (use_registers/use_windowed_attn/num_windows/...)
    ):
        super().__init__()
        if size not in size_to_width:
            raise ValueError(f"DinoV3 size must be one of {list(size_to_width)}, got {size!r}")
        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = 1  # RoPE: no windowing
        out_features = [f"stage{i}" for i in out_feature_indexes]
        name = hf_name or _resolve_model_path(size)

        if load_weights:
            # Use the explicit class (AutoBackbone mis-routes local dirs to its timm loader).
            # `name` is a local bundled dir when available, else the (gated) HF repo id.
            try:
                self.encoder = DINOv3ViTBackbone.from_pretrained(name, out_features=out_features)
                logger.info("DinoV3: loaded pretrained backbone %s (out_features=%s)", name, out_features)
            except Exception as exc:  # noqa: BLE001 — gated repo / offline / missing token
                logger.warning(
                    "DinoV3: could NOT load gated weights %s (%s: %s). Falling back to RANDOM-INIT. "
                    "Accept the DINOv3 license + set HF_TOKEN, or bundle the weights, before training.",
                    name, type(exc).__name__, str(exc).splitlines()[0][:120],
                )
                load_weights = False

        if not load_weights:
            cfg = DINOv3ViTConfig(
                patch_size=patch_size,
                image_size=shape[0],
                out_features=out_features,
                **size_to_arch[size],
            )
            self.encoder = DINOv3ViTBackbone(cfg)
            logger.warning("DinoV3: RANDOM-INIT %s backbone — plumbing-only, NOT for real training.", size)

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self):
        # RoPE has no learned positional-embedding grid to interpolate — nothing to do.
        self._export = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"DinoV3 backbone requires input divisible by {block_size}, got {tuple(x.shape)}"
        )
        out = self.encoder(x)
        feature_maps = out.feature_maps if hasattr(out, "feature_maps") else out[0]
        return list(feature_maps)
