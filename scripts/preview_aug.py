"""Render fisheye/night augmentation previews on REAL eccv-cross-city frames, with GT boxes drawn,
so we can eyeball (a) the fisheye warp, (b) that boxes track the warp, (c) night realism — BEFORE
spending any training credit.

Each output PNG is a 3-panel strip:  [ original + boxes | FISHEYE + boxes | NIGHT + boxes ].
The fisheye/night transforms are forced to p=1.0 here (showcase); in training they fire at the
configured probabilities (see AUG_FISHEYE_NIGHT in scripts/train.py).

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/preview_aug.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS opencv+torch OpenMP clash (local only)
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rf-detr" / "src"))
import polars as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from rfdetr.datasets.transforms import AlbumentationsWrapper  # noqa: E402
from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402

# Showcase strengths (p=1.0 so the effect is always visible in the preview).
FISHEYE = [{"OpticalDistortion": {"distort_limit": (0.4, 0.6), "mode": "fisheye", "p": 1.0}}]
NIGHT = [
    {"RandomBrightnessContrast": {"brightness_limit": (-0.45, -0.30), "contrast_limit": (-0.1, 0.1), "p": 1.0}},
    {"RandomGamma": {"gamma_limit": (160, 200), "p": 1.0}},
    {"ToGray": {"p": 1.0}},
    {"ISONoise": {"color_shift": (0.02, 0.05), "intensity": (0.3, 0.5), "p": 1.0}},
    {"RGBShift": {"r_shift_limit": 5, "g_shift_limit": 5, "b_shift_limit": 30, "p": 1.0}},
]


def boxes_of(row: dict) -> list[list[float]]:
    w, h = row["width"], row["height"]
    out = []
    for b in (row["bboxes"] or []):
        if b.get("task_name") != "object_detection":
            continue
        x1, y1 = b["top_left_x"] * w, b["top_left_y"] * h
        out.append([x1, y1, (b["top_left_x"] + b["width"]) * w, (b["top_left_y"] + b["height"]) * h])
    return out


def draw(img: Image.Image, boxes: list[list[float]], color=(255, 40, 40)) -> Image.Image:
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return img


def apply(wrappers: list, img: Image.Image, boxes: list[list[float]]):
    tgt = {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        "labels": torch.zeros((len(boxes),), dtype=torch.long),
    }
    out = img
    for w in wrappers:
        out, tgt = w(out, tgt)
    return out, tgt["boxes"].tolist()


def main() -> None:
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    picks = []
    for cam_filter in ["GRANDVIEW", "HWY 20", "5 POINTS", None]:
        sub = df if cam_filter is None else df.filter(pl.col("cam").str.contains(cam_filter))
        for i in range(len(sub)):
            r = sub.row(i, named=True)
            if len(boxes_of(r)) >= 4:
                picks.append(r)
                break
        if len(picks) >= 3:
            break

    fw = AlbumentationsWrapper.from_config(FISHEYE)
    nw = AlbumentationsWrapper.from_config(NIGHT)
    out_dir = REPO / ".debug_out"
    out_dir.mkdir(exist_ok=True)
    saved = []
    for k, r in enumerate(picks[:3]):
        img = Image.open(r["file_path"]).convert("RGB")
        bx = boxes_of(r)
        fimg, fbx = apply(fw, img, bx)
        nimg, nbx = apply(nw, img, bx)
        panels = [draw(img, bx), draw(fimg, fbx), draw(nimg, nbx)]
        h = 360
        panels = [p.resize((max(1, int(p.width * h / p.height)), h)) for p in panels]
        w_total = sum(p.width for p in panels) + 20
        canvas = Image.new("RGB", (w_total, h), (255, 255, 255))
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + 10
        cam = r["cam"][:12].replace(" ", "_").replace("/", "_")
        out = out_dir / f"aug_preview_{k}_{cam}.png"
        canvas.save(out)
        saved.append(str(out))
        print(f"saved {out.name} | cam={r['cam']} | boxes {len(bx)} -> fisheye {len(fbx)}, night {len(nbx)}")
    print("PREVIEWS:")
    for s in saved:
        print(s)


if __name__ == "__main__":
    main()
