"""Run the trained RF-DETR Large checkpoint on the local sample TEST split and
surface failure cases (missed objects, false positives, misclassifications).

Outputs side-by-side GT-vs-Prediction composites + a montage of the worst cases
to `visualization/failures/`.

NOTE: the local sample is from the SAME (source) city the model trained on, so
these are *in-domain* errors. The real cross-city test set is hidden — expect the
target-city numbers to be worse than what you see here.

Usage:
    python scripts/visualize_failures.py [--threshold 0.25] [--topk 9]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rf-detr" / "src"))

# transformers v4 -> v5 compat: expose backbone mixins at top level if needed.
import transformers  # noqa: E402
from transformers.utils import backbone_utils as _bu  # noqa: E402

for _n in ("BackboneConfigMixin", "BackboneMixin", "BackboneType"):
    if not hasattr(transformers, _n) and hasattr(_bu, _n):
        setattr(transformers, _n, getattr(_bu, _n))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402
from rfdetr import RFDETRLarge  # noqa: E402

CLASS_NAMES = [
    "Vehicle.Car", "Vehicle.Pickup Truck", "Vehicle.Single Truck", "Vehicle.Combo Truck",
    "Vehicle.Heavy Duty Vehicle", "Vehicle.Trailer", "Vehicle.Motorcycle", "Vehicle.Bicycle",
    "Vehicle.Van", "Person",
]
SHORT = ["Car", "Pickup", "SingleTrk", "ComboTrk", "HeavyDuty", "Trailer", "Moto", "Bicycle", "Van", "Person"]

CKPT = "/tmp/v3_ckpt/checkpoint_best_ema.pth"
OUT = REPO_ROOT / "visualization" / "failures"

# colors (RGB)
C_GT = (0, 200, 0)          # ground truth — green
C_TP = (40, 120, 255)       # correct prediction — blue
C_MIS = (255, 150, 0)       # misclassified (right place, wrong label) — orange
C_FP = (235, 30, 30)        # false positive — red
C_MISS = (255, 0, 200)      # missed GT highlight — magenta


def load_font(size: int):
    for p in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def gt_boxes(row):
    """Return list of (x1,y1,x2,y2,class_idx) in pixels from normalized GT bboxes."""
    W, H = row["width"], row["height"]
    out = []
    for b in (row["bboxes"] or []):
        if b.get("task_name") != "object_detection":
            continue
        x1 = b["top_left_x"] * W
        y1 = b["top_left_y"] * H
        x2 = (b["top_left_x"] + b["width"]) * W
        y2 = (b["top_left_y"] + b["height"]) * H
        out.append((x1, y1, x2, y2, int(b["class_idx"])))
    return out


def match(preds, gts, iou_thr=0.5):
    """Greedy IoU match. preds: list (box, cls, conf) sorted desc conf.
    Returns dict with categorized lists and counts."""
    matched_gt = [False] * len(gts)
    tp, mis, fp = [], [], []
    confusions = []  # (gt_cls, pred_cls)
    for box, cls, conf in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched_gt[j]:
                continue
            i = iou(box, g[:4])
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thr and best_j >= 0:
            matched_gt[best_j] = True
            if gts[best_j][4] == cls:
                tp.append((box, cls, conf))
            else:
                mis.append((box, cls, conf, gts[best_j][4]))
                confusions.append((gts[best_j][4], cls))
        else:
            fp.append((box, cls, conf))
    missed = [gts[j] for j in range(len(gts)) if not matched_gt[j]]
    return {"tp": tp, "mis": mis, "fp": fp, "missed": missed, "confusions": confusions}


def draw_boxes(img, boxes, color, font, labels=None, width=3, dashed=False):
    d = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[:4]
        if dashed:
            _dashed_rect(d, (x1, y1, x2, y2), color, width)
        else:
            d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels is not None and labels[i]:
            txt = labels[i]
            tb = d.textbbox((0, 0), txt, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 4)
            d.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
            d.text((x1 + 3, ty + 1), txt, fill=(255, 255, 255), font=font)


def _dashed_rect(d, box, color, width, dash=14, gap=8):
    x1, y1, x2, y2 = box
    def hline(y, xa, xb):
        x = xa
        while x < xb:
            d.line([x, y, min(x + dash, xb), y], fill=color, width=width)
            x += dash + gap
    def vline(x, ya, yb):
        y = ya
        while y < yb:
            d.line([x, y, x, min(y + dash, yb)], fill=color, width=width)
            y += dash + gap
    hline(y1, x1, x2); hline(y2, x1, x2); vline(x1, y1, y2); vline(x2, y1, y2)


def banner(img, text, font):
    """Add a title bar at the top."""
    d = ImageDraw.Draw(img)
    tb = d.textbbox((0, 0), text, font=font)
    h = tb[3] - tb[1] + 12
    bar = Image.new("RGB", (img.width, h), (20, 20, 20))
    ImageDraw.Draw(bar).text((8, 4), text, fill=(255, 255, 255), font=font)
    out = Image.new("RGB", (img.width, img.height + h), (20, 20, 20))
    out.paste(bar, (0, 0))
    out.paste(img, (0, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=9)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--split", default="validation",
                    help="validation (held-out, has GT) or train. 'test' GT is withheld.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] model ...")
    model = RFDETRLarge(num_classes=10, resolution=704, pretrain_weights=CKPT)
    model.optimize_for_inference()

    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    test = ds.samples.filter(ds.samples["split"] == args.split)
    print(f"[data] {len(test)} {args.split} images")

    font_big = load_font(34)
    font_lbl = load_font(20)

    results = []
    g_tp = g_fp = g_miss = g_mis = 0
    per_class_gt = Counter()
    per_class_missed = Counter()
    per_class_fp = Counter()
    confusion = Counter()

    for i in range(len(test)):
        row = test.row(i, named=True)
        img = Image.open(row["file_path"]).convert("RGB")
        det = model.predict(img, threshold=args.threshold)
        preds = []
        for box, cls, conf in zip(det.xyxy, det.class_id, det.confidence):
            preds.append((tuple(float(v) for v in box), int(cls), float(conf)))
        preds.sort(key=lambda x: -x[2])
        gts = gt_boxes(row)
        m = match(preds, gts, iou_thr=args.iou)

        g_tp += len(m["tp"]); g_fp += len(m["fp"]); g_miss += len(m["missed"]); g_mis += len(m["mis"])
        for g in gts:
            per_class_gt[g[4]] += 1
        for g in m["missed"]:
            per_class_missed[g[4]] += 1
        for (_b, c, _cf) in m["fp"]:
            per_class_fp[c] += 1
        for (gc, pc) in m["confusions"]:
            confusion[(gc, pc)] += 1

        # failure score: misses + FPs + 1.5*misclass, scaled to be readable
        score = len(m["missed"]) + len(m["fp"]) + 1.5 * len(m["mis"])
        results.append({"i": i, "row": row, "img_path": row["file_path"], "m": m,
                        "n_gt": len(gts), "score": score})
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(test)}")

    # ---------- global summary ----------
    print("\n================ GLOBAL (in-domain TEST, n={}) ================".format(len(test)))
    print(f"threshold={args.threshold}  IoU={args.iou}")
    print(f"TP={g_tp}  Missed(FN)={g_miss}  FalsePos(FP)={g_fp}  Misclassified={g_mis}")
    tot_gt = sum(per_class_gt.values())
    recall = g_tp / max(1, (g_tp + g_miss + g_mis))
    prec = g_tp / max(1, (g_tp + g_fp + g_mis))
    print(f"GT objects={tot_gt}  approx recall={recall:.3f}  approx precision={prec:.3f}")
    print("\nPer-class miss rate (missed / GT):")
    for c in range(10):
        if per_class_gt[c]:
            print(f"  {SHORT[c]:10s} GT={per_class_gt[c]:4d}  missed={per_class_missed[c]:4d}"
                  f"  ({per_class_missed[c]/per_class_gt[c]*100:5.1f}%)  FP={per_class_fp[c]}")
        else:
            print(f"  {SHORT[c]:10s} GT=   0  (no GT in test sample)  FP={per_class_fp[c]}")
    print("\nTop confusions (GT -> predicted):")
    for (gc, pc), n in confusion.most_common(8):
        print(f"  {SHORT[gc]} -> {SHORT[pc]}: {n}")

    # ---------- render worst cases ----------
    results.sort(key=lambda r: -r["score"])
    worst = [r for r in results if r["score"] > 0][:args.topk]
    print(f"\n[render] writing {len(worst)} worst cases to {OUT}")

    montage_panels = []
    for rank, r in enumerate(worst):
        row = r["row"]; m = r["m"]
        base = Image.open(r["img_path"]).convert("RGB")

        # GT panel
        gt_img = base.copy()
        gboxes = gt_boxes(row)
        draw_boxes(gt_img, [g[:4] for g in gboxes], C_GT, font_lbl,
                   labels=[SHORT[g[4]] for g in gboxes], width=3)
        gt_img = banner(gt_img, f"GROUND TRUTH  ({len(gboxes)} objects)", font_big)

        # Prediction panel
        pr_img = base.copy()
        # missed GT first (dashed magenta) so they stand out
        draw_boxes(pr_img, [g[:4] for g in m["missed"]], C_MISS, font_lbl,
                   labels=[f"MISSED:{SHORT[g[4]]}" for g in m["missed"]], width=3, dashed=True)
        draw_boxes(pr_img, [b[0] for b in m["tp"]], C_TP, font_lbl,
                   labels=[f"{SHORT[b[1]]} {b[2]:.2f}" for b in m["tp"]], width=2)
        draw_boxes(pr_img, [b[0] for b in m["mis"]], C_MIS, font_lbl,
                   labels=[f"{SHORT[b[1]]}? (gt:{SHORT[b[3]]}) {b[2]:.2f}" for b in m["mis"]], width=3)
        draw_boxes(pr_img, [b[0] for b in m["fp"]], C_FP, font_lbl,
                   labels=[f"FP:{SHORT[b[1]]} {b[2]:.2f}" for b in m["fp"]], width=3)
        cam = (row.get("camera_info") or {}).get("name", "?")
        pr_img = banner(pr_img,
                        f"PRED  miss={len(m['missed'])} FP={len(m['fp'])} misclass={len(m['mis'])}  | cam:{cam}",
                        font_big)

        # side by side
        gap = 12
        combo = Image.new("RGB", (gt_img.width + pr_img.width + gap, max(gt_img.height, pr_img.height)),
                          (20, 20, 20))
        combo.paste(gt_img, (0, 0))
        combo.paste(pr_img, (gt_img.width + gap, 0))
        # downscale for file size
        scale = 1600 / combo.width
        combo = combo.resize((int(combo.width * scale), int(combo.height * scale)))
        fn = OUT / f"worst_{rank+1:02d}_score{r['score']:.0f}.jpg"
        combo.save(fn, quality=88)
        montage_panels.append(pr_img.resize((int(pr_img.width * 0.33), int(pr_img.height * 0.33))))
        print(f"  {fn.name}  (gt={r['n_gt']} miss={len(m['missed'])} fp={len(m['fp'])} mis={len(m['mis'])})")

    # montage grid (3 cols)
    if montage_panels:
        cols = 3
        rows = (len(montage_panels) + cols - 1) // cols
        cw = max(p.width for p in montage_panels)
        ch = max(p.height for p in montage_panels)
        grid = Image.new("RGB", (cols * cw, rows * ch), (10, 10, 10))
        for idx, p in enumerate(montage_panels):
            grid.paste(p, ((idx % cols) * cw, (idx // cols) * ch))
        grid.save(OUT / "montage_worst.jpg", quality=85)
        print(f"  montage_worst.jpg ({grid.width}x{grid.height})")

    print("\n[done] open:", OUT)


if __name__ == "__main__":
    main()
