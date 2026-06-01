"""Slim the BDD100K Cascade-R-CNN/ConvNeXt checkpoint for bundling into the trainer.

The original checkpoint (834 MB) carries optimizer state (~583 MB of AdamW moments)
that `load_from` does not need — only `state_dict` (the trained model weights) and
`meta` are required to initialize the detector. Stripping the optimizer brings it down
to ~291 MB (fp32) or ~146 MB (fp16), which makes the Hafnia trainer.zip upload + Docker
build far quicker.

Usage:
    python scripts/prepare_weights.py \
        --src ../weights/cascade_rcnn_convnext-s_fpn_fp16_3x_det_bdd100k.pth \
        --dst weights/cascade_convnext_bdd_slim.pth
    python scripts/prepare_weights.py ... --fp16     # half the size, loads fine into the fp32 model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT.parent / "weights" / "cascade_rcnn_convnext-s_fpn_fp16_3x_det_bdd100k.pth"
DEFAULT_DST = REPO_ROOT / "weights" / "cascade_convnext_bdd_slim.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    p.add_argument("--fp16", action="store_true", help="store weights as float16 (~146 MB vs ~291 MB)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"source checkpoint not found: {args.src}")

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)

    if args.fp16:
        state = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in state.items()}

    # Keep only what mmdet's load_from consumes: the weights + provenance meta.
    slim = {"state_dict": state, "meta": ck.get("meta", {})}

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, args.dst)

    src_mb = args.src.stat().st_size / 1e6
    dst_mb = args.dst.stat().st_size / 1e6
    print(f"[slim] {args.src.name} ({src_mb:.0f} MB) -> {args.dst.name} ({dst_mb:.0f} MB), "
          f"{len(state)} tensors, fp16={args.fp16}")


if __name__ == "__main__":
    main()
