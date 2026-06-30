"""Find the specific local-sample images where v28 (RF-DETR Large, scratch+stratified,
R1280, best EMA epoch 9) performs worst.

Reads ground truth OFFLINE from the local parquet (.data/datasets/eccv-cross-city),
runs v28 at its native resolution (1280), matches predictions to GT (greedy IoU@0.5),
scores each image by errors (missed + false-pos + 1.5*misclassified), prints a ranked
table, and renders side-by-side GT-vs-PRED composites for the worst cases.

NOTE: the local sample is the SOURCE city and v28 used --split-mode stratified (all
cameras in train), so this is in-domain — the hidden cross-city target will be worse.

    PYTHONPATH=rf-detr/src python scripts/analyze_v28_failures.py [--threshold 0.3] [--topk 16]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rf-detr" / "src"))

# transformers v4 -> v5 compat (same shim as visualize_failures.py)
import transformers  # noqa: E402
from transformers.utils import backbone_utils as _bu  # noqa: E402

for _n in ("BackboneConfigMixin", "BackboneMixin", "BackboneType"):
    if not hasattr(transformers, _n) and hasattr(_bu, _n):
        setattr(transformers, _n, getattr(_bu, _n))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rfdetr import RFDETRLarge  # noqa: E402

DATASET_DIR = REPO / ".data" / "datasets" / "eccv-cross-city"
PARQUET = DATASET_DIR / "annotations.parquet"
CKPT = str(REPO / "weights" / "v28_best_ema.pth")
RES = 1280
OUT = REPO / ".data" / "viz" / "v28_failures"

CLASS_NAMES = [
    "Vehicle.Car", "Vehicle.Pickup Truck", "Vehicle.Single Truck", "Vehicle.Combo Truck",
    "Vehicle.Heavy Duty Vehicle", "Vehicle.Trailer", "Vehicle.Motorcycle", "Vehicle.Bicycle",
    "Vehicle.Van", "Person",
]
SHORT = ["Car", "Pickup", "SingleTrk", "ComboTrk", "HeavyDuty", "Trailer", "Moto", "Bicycle", "Van", "Person"]

C_GT = (0, 200, 0); C_TP = (40, 120, 255); C_MIS = (255, 150, 0); C_FP = (235, 30, 30); C_MISS = (255, 0, 200)


def load_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
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
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def gt_boxes(row):
    """Pixel (x1,y1,x2,y2,cls) from normalized object_detection GT bboxes."""
    W, H = row["width"], row["height"]
    out = []
    for b in (row["bboxes"] if row["bboxes"] is not None else []):
        if not isinstance(b, dict) or b.get("task_name") != "object_detection":
            continue
        x1 = b["top_left_x"] * W; y1 = b["top_left_y"] * H
        x2 = (b["top_left_x"] + b["width"]) * W; y2 = (b["top_left_y"] + b["height"]) * H
        out.append((x1, y1, x2, y2, int(b["class_idx"])))
    return out


def match(preds, gts, iou_thr=0.5):
    matched = [False] * len(gts)
    tp, mis, fp, conf = [], [], [], []
    for box, cls, cf in preds:
        bi, bj = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou(box, g[:4])
            if v > bi:
                bi, bj = v, j
        if bi >= iou_thr and bj >= 0:
            matched[bj] = True
            if gts[bj][4] == cls:
                tp.append((box, cls, cf))
            else:
                mis.append((box, cls, cf, gts[bj][4])); conf.append((gts[bj][4], cls))
        else:
            fp.append((box, cls, cf))
    missed = [gts[j] for j in range(len(gts)) if not matched[j]]
    return {"tp": tp, "mis": mis, "fp": fp, "missed": missed, "conf": conf}


def draw(img, boxes, color, font, labels=None, width=3, dashed=False):
    d = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[:4]
        if dashed:
            for (xa, ya, xb, yb) in [(x1, y1, x2, y1), (x1, y2, x2, y2), (x1, y1, x1, y2), (x2, y1, x2, y2)]:
                _dash(d, xa, ya, xb, yb, color, width)
        else:
            d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels and labels[i]:
            t = labels[i]; tb = d.textbbox((0, 0), t, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]; ty = max(0, y1 - th - 4)
            d.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
            d.text((x1 + 3, ty + 1), t, fill=(255, 255, 255), font=font)


def _dash(d, xa, ya, xb, yb, color, width, dash=14, gap=8):
    import math
    L = math.hypot(xb - xa, yb - ya)
    if L == 0:
        return
    ux, uy = (xb - xa) / L, (yb - ya) / L
    s = 0.0
    while s < L:
        e = min(s + dash, L)
        d.line([xa + ux * s, ya + uy * s, xa + ux * e, ya + uy * e], fill=color, width=width)
        s += dash + gap


def banner(img, text, font):
    d = ImageDraw.Draw(img); tb = d.textbbox((0, 0), text, font=font); h = tb[3] - tb[1] + 14
    bar = Image.new("RGB", (img.width, h), (20, 20, 20)); ImageDraw.Draw(bar).text((8, 5), text, fill=(255, 255, 255), font=font)
    out = Image.new("RGB", (img.width, img.height + h), (20, 20, 20)); out.paste(bar, (0, 0)); out.paste(img, (0, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=16)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[load] v28 RFDETRLarge(res={RES}) <- {Path(CKPT).name}")
    model = RFDETRLarge(num_classes=10, resolution=RES, pretrain_weights=CKPT)
    model.optimize_for_inference()

    df = pd.read_parquet(PARQUET)
    df = df[df["split"] != "test"].reset_index(drop=True)  # test has no GT
    print(f"[data] {len(df)} GT images (train+val)")

    fbig, flbl = load_font(32), load_font(20)
    rows = []
    g_tp = g_fp = g_miss = g_mis = 0
    pc_gt, pc_miss, pc_fp = Counter(), Counter(), Counter()
    confusion = Counter()

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        gts = gt_boxes(row)
        path = DATASET_DIR / row["file_path"]
        img = Image.open(path).convert("RGB")
        det = model.predict(img, threshold=args.threshold)
        preds = sorted(
            [(tuple(float(v) for v in b), int(c), float(s)) for b, c, s in zip(det.xyxy, det.class_id, det.confidence)],
            key=lambda x: -x[2])
        m = match(preds, gts, iou_thr=args.iou)

        g_tp += len(m["tp"]); g_fp += len(m["fp"]); g_miss += len(m["missed"]); g_mis += len(m["mis"])
        for g in gts:
            pc_gt[g[4]] += 1
        for g in m["missed"]:
            pc_miss[g[4]] += 1
        for (_b, c, _f) in m["fp"]:
            pc_fp[c] += 1
        for cpair in m["conf"]:
            confusion[cpair] += 1

        n_gt = len(gts); n_err = len(m["missed"]) + len(m["fp"]) + len(m["mis"])
        score = len(m["missed"]) + len(m["fp"]) + 1.5 * len(m["mis"])
        recall = len(m["tp"]) / max(1, n_gt)
        cam = (row.get("camera_info") or {}).get("name", "?")
        rows.append({"i": i, "row": row, "m": m, "file": Path(row["file_path"]).name,
                     "split": row["split"], "cam": cam, "n_gt": n_gt,
                     "miss": len(m["missed"]), "fp": len(m["fp"]), "mis": len(m["mis"]),
                     "tp": len(m["tp"]), "recall": recall, "n_err": n_err, "score": score})
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(df)}")

    # ---- global summary ----
    tot_gt = sum(pc_gt.values())
    recall = g_tp / max(1, g_tp + g_miss + g_mis)
    prec = g_tp / max(1, g_tp + g_fp + g_mis)
    print("\n" + "=" * 64)
    print(f"v28 on local sample  (n={len(df)} imgs, threshold={args.threshold}, IoU={args.iou})")
    print("=" * 64)
    print(f"GT objects={tot_gt}  TP={g_tp}  Missed={g_miss}  FalsePos={g_fp}  Misclass={g_mis}")
    print(f"approx recall={recall:.3f}  approx precision={prec:.3f}")
    print("\nPer-class  miss-rate (missed/GT)  and FP count:")
    print(f"  {'class':10s} {'GT':>5s} {'missed':>7s} {'miss%':>7s} {'FP':>5s}")
    for c in range(10):
        gt = pc_gt[c]
        mr = f"{pc_miss[c]/gt*100:5.1f}" if gt else "  n/a"
        print(f"  {SHORT[c]:10s} {gt:5d} {pc_miss[c]:7d} {mr:>7s} {pc_fp[c]:5d}")
    print("\nTop confusions (GT -> predicted):")
    for (gc, pc), n in confusion.most_common(8):
        print(f"  {SHORT[gc]} -> {SHORT[pc]}: {n}")

    # ---- ranked table (worst first) ----
    rows.sort(key=lambda r: (-r["score"], -r["n_err"]))
    print("\n" + "=" * 90)
    print("WORST IMAGES (ranked by error score = missed + FP + 1.5*misclass)")
    print("=" * 90)
    print(f"{'#':>3} {'score':>6} {'miss':>4} {'FP':>4} {'mis':>4} {'GT':>4} {'recall':>7} {'split':>5}  {'camera':<26} file")
    for rk, r in enumerate(rows[:40], 1):
        print(f"{rk:>3} {r['score']:6.1f} {r['miss']:>4} {r['fp']:>4} {r['mis']:>4} {r['n_gt']:>4} "
              f"{r['recall']*100:6.1f}% {r['split']:>5}  {r['cam'][:26]:<26} {r['file']}")

    # ---- save full ranking ----
    with open(OUT / "ranking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "miss", "fp", "misclass", "n_gt", "tp", "recall", "split", "camera", "file"])
        for rk, r in enumerate(rows, 1):
            w.writerow([rk, f"{r['score']:.1f}", r["miss"], r["fp"], r["mis"], r["n_gt"], r["tp"],
                        f"{r['recall']:.3f}", r["split"], r["cam"], r["file"]])
    print(f"\n[save] full ranking -> {OUT/'ranking.csv'}")

    # ---- render worst composites ----
    worst = [r for r in rows if r["score"] > 0][:args.topk]
    print(f"[render] {len(worst)} worst composites -> {OUT}")
    panels = []
    for rk, r in enumerate(worst, 1):
        row = r["row"]; m = r["m"]
        base = Image.open(DATASET_DIR / row["file_path"]).convert("RGB")
        gt_img = base.copy(); gb = gt_boxes(row)
        draw(gt_img, [g[:4] for g in gb], C_GT, flbl, labels=[SHORT[g[4]] for g in gb], width=3)
        gt_img = banner(gt_img, f"GT ({len(gb)} obj)  {r['file'][:24]}", fbig)
        pr = base.copy()
        draw(pr, [g[:4] for g in m["missed"]], C_MISS, flbl, labels=[f"MISS:{SHORT[g[4]]}" for g in m["missed"]], width=3, dashed=True)
        draw(pr, [b[0] for b in m["tp"]], C_TP, flbl, labels=[f"{SHORT[b[1]]} {b[2]:.2f}" for b in m["tp"]], width=2)
        draw(pr, [b[0] for b in m["mis"]], C_MIS, flbl, labels=[f"{SHORT[b[1]]}?(gt:{SHORT[b[3]]})" for b in m["mis"]], width=3)
        draw(pr, [b[0] for b in m["fp"]], C_FP, flbl, labels=[f"FP:{SHORT[b[1]]} {b[2]:.2f}" for b in m["fp"]], width=3)
        pr = banner(pr, f"PRED  miss={r['miss']} FP={r['fp']} misclass={r['mis']}  cam:{r['cam'][:22]}", fbig)
        gap = 12
        combo = Image.new("RGB", (gt_img.width + pr.width + gap, max(gt_img.height, pr.height)), (20, 20, 20))
        combo.paste(gt_img, (0, 0)); combo.paste(pr, (gt_img.width + gap, 0))
        sc = 1700 / combo.width
        combo = combo.resize((int(combo.width * sc), int(combo.height * sc)))
        combo.save(OUT / f"worst_{rk:02d}_score{r['score']:.0f}_{r['file']}.jpg", quality=88)
        panels.append(pr.resize((int(pr.width * 0.3), int(pr.height * 0.3))))

    if panels:
        cols = 4; nrows = (len(panels) + cols - 1) // cols
        cw = max(p.width for p in panels); ch = max(p.height for p in panels)
        grid = Image.new("RGB", (cols * cw, nrows * ch), (10, 10, 10))
        for idx, p in enumerate(panels):
            grid.paste(p, ((idx % cols) * cw, (idx // cols) * ch))
        grid.save(OUT / "montage_worst.jpg", quality=85)
        print(f"[render] montage_worst.jpg ({grid.width}x{grid.height})")
    print("\n[done]", OUT)


if __name__ == "__main__":
    main()
