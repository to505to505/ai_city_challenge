"""Run v28 (RF-DETR Large, R1280, best EMA epoch 9) over the LOCAL test split and draw
predicted boxes so a human can eyeball detection quality. The test split has NO ground
truth, so this is predictions-only (no error scoring).

Writes one annotated JPG per image + a montage to .data/viz/v28_test_preds/.

    PYTHONPATH=rf-detr/src python scripts/predict_v28_test.py [--threshold 0.3]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v28_failures import (  # noqa: E402  (also applies the transformers shim)
    CKPT, RES, DATASET_DIR, SHORT, load_font, draw, banner,
)

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from rfdetr import RFDETRLarge  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / ".data" / "viz" / "v28_test_preds"
# 10-class palette (RGB)
PALETTE = [
    (40, 120, 255), (0, 200, 120), (255, 150, 0), (200, 60, 220), (0, 200, 200),
    (255, 90, 90), (150, 110, 60), (120, 200, 40), (90, 90, 240), (235, 30, 30),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--montage", type=int, default=24, help="how many imgs in the montage")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[load] v28 RFDETRLarge(res={RES}) <- {Path(CKPT).name}")
    model = RFDETRLarge(num_classes=10, resolution=RES, pretrain_weights=CKPT)
    model.optimize_for_inference()

    df = pd.read_parquet(DATASET_DIR / "annotations.parquet")
    df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"[data] {len(df)} test images (no GT)")

    flbl = load_font(20)
    fbig = load_font(30)
    pc = Counter()
    per_img = []
    panels = []

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        path = DATASET_DIR / row["file_path"]
        img = Image.open(path).convert("RGB")
        det = model.predict(img, threshold=args.threshold)
        dets = sorted(
            [(tuple(float(v) for v in b), int(c), float(s)) for b, c, s in zip(det.xyxy, det.class_id, det.confidence)],
            key=lambda x: -x[2])
        for _b, c, _s in dets:
            pc[c] += 1
        per_img.append((row["file_path"], len(dets)))

        # draw per class group (color by class)
        canvas = img.copy()
        for c in range(10):
            boxes = [d[0] for d in dets if d[1] == c]
            labs = [f"{SHORT[c]} {d[2]:.2f}" for d in dets if d[1] == c]
            if boxes:
                draw(canvas, boxes, PALETTE[c], flbl, labels=labs, width=2)
        cam = (row.get("camera_info") or {}).get("name", "?")
        canvas = banner(canvas, f"v28 thr={args.threshold}  dets={len(dets)}  cam:{cam}  {Path(row['file_path']).name[-20:]}", fbig)
        sc = 1280 / canvas.width
        small = canvas.resize((int(canvas.width * sc), int(canvas.height * sc)))
        small.save(OUT / f"test_{i:03d}_{Path(row['file_path']).name}.jpg", quality=85)
        if i < args.montage:
            panels.append(small.resize((int(small.width * 0.34), int(small.height * 0.34))))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(df)}")

    total = sum(pc.values())
    print("\n" + "=" * 56)
    print(f"v28 on local TEST  (n={len(df)} imgs, threshold={args.threshold})")
    print("=" * 56)
    print(f"total detections={total}  ({total/max(1,len(df)):.1f} per image)")
    print("\nPer-class detection counts:")
    for c in range(10):
        print(f"  {SHORT[c]:10s} {pc[c]:5d}  ({pc[c]/max(1,total)*100:4.1f}%)")
    n0 = sum(1 for _f, n in per_img if n == 0)
    print(f"\nimages with 0 detections: {n0}/{len(df)}")
    busiest = sorted(per_img, key=lambda x: -x[1])[:8]
    print("busiest frames:", ", ".join(f"{Path(f).name[-12:]}={n}" for f, n in busiest))

    if panels:
        cols = 4
        nrows = (len(panels) + cols - 1) // cols
        cw = max(p.width for p in panels); ch = max(p.height for p in panels)
        grid = Image.new("RGB", (cols * cw, nrows * ch), (10, 10, 10))
        for idx, p in enumerate(panels):
            grid.paste(p, ((idx % cols) * cw, (idx // cols) * ch))
        grid.save(OUT / "montage_test.jpg", quality=85)
        print(f"[render] montage_test.jpg ({grid.width}x{grid.height})")
    print("\n[done]", OUT)


if __name__ == "__main__":
    main()
