"""Pull REAL frames from the cameras we called 'fisheye' (GRANDVIEW + the held-out set) and put a
real frame next to our SYNTHETIC OpticalDistortion(mode='fisheye') applied to a flat camera — so we
can judge whether the augmentation actually matches the real lens, instead of just asserting it.

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/real_fisheye_check.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rf-detr" / "src"))
import polars as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from rfdetr.datasets.transforms import AlbumentationsWrapper  # noqa: E402
from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402

OUT = REPO / ".debug_out"
OUT.mkdir(exist_ok=True)


def label(im: Image.Image, text: str) -> Image.Image:
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, max(120, 7 * len(text)), 16], fill=(0, 0, 0))
    d.text((3, 3), text, fill=(255, 255, 0))
    return im


def main() -> None:
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    cams = sorted([c for c in df["cam"].unique().to_list() if c])
    print("ALL CAMERAS:")
    for c in cams:
        print("  ", c)

    # 1) montage one REAL frame per camera that looks like a wide/fisheye candidate
    candidates = [c for c in cams if any(k in c.upper() for k in ["GRANDVIEW", "POINTS", "ARTERIAL", "CONNECTOR"])]
    h = 300
    cells = []
    for cam in candidates:
        r = df.filter(pl.col("cam") == cam).row(0, named=True)
        im = Image.open(r["file_path"]).convert("RGB")
        im = im.resize((int(im.width * h / im.height), h))
        cells.append(label(im, cam[:24]))
    if cells:
        cols = 2
        rows = (len(cells) + cols - 1) // cols
        cw = max(c.width for c in cells)
        canvas = Image.new("RGB", (cw * cols, h * rows), (255, 255, 255))
        for i, im in enumerate(cells):
            canvas.paste(im, ((i % cols) * cw, (i // cols) * h))
        canvas.save(OUT / "real_cams_montage.png")
        print("saved real_cams_montage.png with", len(cells), "cameras")

    # 2) direct A/B: a REAL GRANDVIEW frame  vs  our synthetic fisheye on a FLAT (HWY) frame
    def first(camsub):
        s = df.filter(pl.col("cam").str.contains(camsub))
        return s.row(0, named=True) if len(s) else None

    import cv2  # noqa: E402

    def synth(flat_im, params):
        w = AlbumentationsWrapper.from_config([{"OpticalDistortion": params}])[0]
        out, _ = w(flat_im, {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
        return out

    real = first("GRANDVIEW")
    flat = first("HWY 20")
    if real and flat:
        real_im = Image.open(real["file_path"]).convert("RGB")
        flat_im = Image.open(flat["file_path"]).convert("RGB")
        # current v10 aug (strong, default black border) vs corrected (mild, frame-filling reflect)
        cur = synth(flat_im, {"distort_limit": (0.4, 0.6), "mode": "fisheye", "p": 1.0})
        fixed = synth(
            flat_im,
            {"distort_limit": (0.05, 0.18), "mode": "fisheye", "border_mode": cv2.BORDER_REFLECT_101, "p": 1.0},
        )
        H = 300
        panels = [
            label(real_im.resize((int(real_im.width * H / real_im.height), H)), f"REAL {real['cam'][:16]}"),
            label(flat_im.resize((int(flat_im.width * H / flat_im.height), H)), "REAL flat HWY20"),
            label(cur.resize((int(cur.width * H / cur.height), H)), "v10 SYNTH strong+black"),
            label(fixed.resize((int(fixed.width * H / fixed.height), H)), "FIXED mild+filled"),
        ]
        w = sum(p.width for p in panels) + 30
        canvas = Image.new("RGB", (w, H), (255, 255, 255))
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + 10
        canvas.save(OUT / "real_vs_synth_fisheye.png")
        print("saved real_vs_synth_fisheye.png")


if __name__ == "__main__":
    main()
