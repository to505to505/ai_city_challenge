"""Diagnose WHERE/WHY the checkpoint fails: in-domain (seen cameras) vs cross-camera
(the 6 held-out cameras the model never trained on), on the local 300-image sample.

Splits the 146 GT-bearing local images (train+validation) into:
  * SEEN     — cameras present in the platform training set
  * HELDOUT  — the 6 leave-camera-out validation cameras (model never saw them)

For each group reports: COCO mAP (torchmetrics), recall/precision (greedy IoU@0.5),
per-class recall + FP, size-stratified recall (small/med/large), confusions, and
per-camera mAP. Goal: identify the dominant failure mode behind the cross-camera drop.

    python scripts/diagnose_checkpoint.py
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rf-detr" / "src"))

import transformers  # noqa: E402
from transformers.utils import backbone_utils as _bu  # noqa: E402

for _n in ("BackboneConfigMixin", "BackboneMixin", "BackboneType"):
    if not hasattr(transformers, _n) and hasattr(_bu, _n):
        setattr(transformers, _n, getattr(_bu, _n))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402
from rfdetr import RFDETRLarge  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "weights" / "v5_best_ema.pth")
CLASS_NAMES = ["Vehicle.Car", "Vehicle.Pickup Truck", "Vehicle.Single Truck", "Vehicle.Combo Truck",
               "Vehicle.Heavy Duty Vehicle", "Vehicle.Trailer", "Vehicle.Motorcycle", "Vehicle.Bicycle",
               "Vehicle.Van", "Person"]
SHORT = ["Car", "Pickup", "SingleTrk", "ComboTrk", "HeavyDuty", "Trailer", "Moto", "Bicycle", "Van", "Person"]
HELDOUT_CAMS = {"5 POINTS WB", "GRANDVIEW - DELHI INT", "HWY 20 - OLD HWY WBA",
                "LOCUST CONNECTOR NB", "NW ARTERIAL - CHAVENELLE INT", "US 61 - TWIN VALLEY INT"}

PRED_THR_MAP = 0.05   # low threshold so torchmetrics sees the full PR curve
PRED_THR_FAIL = 0.30  # operating point for the missed/FP/misclass breakdown
IOU_THR = 0.5


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def gt_boxes(row):
    W, H = row["width"], row["height"]
    out = []
    for b in (row["bboxes"] or []):
        if b.get("task_name") != "object_detection":
            continue
        x1, y1 = b["top_left_x"]*W, b["top_left_y"]*H
        x2, y2 = (b["top_left_x"]+b["width"])*W, (b["top_left_y"]+b["height"])*H
        out.append((x1, y1, x2, y2, int(b["class_idx"])))
    return out


def size_bin(box):
    side = ((box[2]-box[0]) * (box[3]-box[1])) ** 0.5
    return "small" if side < 32 else ("medium" if side < 96 else "large")


def match(preds, gts, iou_thr=IOU_THR):
    """preds: list (box, cls, conf) desc conf. Greedy IoU match."""
    matched = [False]*len(gts)
    tp, mis, fp = [], [], []
    confusions = []
    for box, cls, conf in preds:
        best_i, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            i = iou(box, g[:4])
            if i > best_i:
                best_i, best_j = i, j
        if best_i >= iou_thr and best_j >= 0:
            matched[best_j] = True
            if gts[best_j][4] == cls:
                tp.append((box, cls, conf))
            else:
                mis.append((box, cls, conf, gts[best_j][4]))
                confusions.append((gts[best_j][4], cls))
        else:
            fp.append((box, cls, conf))
    missed = [g for j, g in enumerate(gts) if not matched[j]]
    return tp, mis, fp, missed, confusions


def main():
    print(f"[load] {CKPT}")
    model = RFDETRLarge(num_classes=10, resolution=704, pretrain_weights=CKPT)
    model.optimize_for_inference()
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    gt = df.filter(pl.col("split").is_in(["train", "validation"]))
    print(f"[data] {len(gt)} GT images")

    from torchmetrics.detection import MeanAveragePrecision
    maps = {"SEEN": MeanAveragePrecision(box_format="xyxy", class_metrics=True),
            "HELDOUT": MeanAveragePrecision(box_format="xyxy", class_metrics=True)}

    # accumulators per group
    agg = {g: dict(tp=0, mis=0, fp=0, missed=0, gt=0,
                   cls_gt=Counter(), cls_missed=Counter(), cls_fp=Counter(),
                   size_gt=Counter(), size_missed=Counter(), conf=Counter(),
                   per_cam=defaultdict(lambda: [0, 0]))  # cam -> [tp, gt]
           for g in ("SEEN", "HELDOUT")}

    n_img = {"SEEN": 0, "HELDOUT": 0}
    for i in range(len(gt)):
        row = gt.row(i, named=True)
        grp = "HELDOUT" if row["cam"] in HELDOUT_CAMS else "SEEN"
        n_img[grp] += 1
        img = Image.open(row["file_path"]).convert("RGB")
        det = model.predict(img, threshold=PRED_THR_MAP)
        gts = gt_boxes(row)

        # torchmetrics update (full preds)
        maps[grp].update(
            [{"boxes": torch.tensor([list(b) for b in det.xyxy], dtype=torch.float32).reshape(-1, 4),
              "scores": torch.tensor(list(det.confidence), dtype=torch.float32),
              "labels": torch.tensor([int(c) for c in det.class_id], dtype=torch.long)}],
            [{"boxes": torch.tensor([list(g[:4]) for g in gts], dtype=torch.float32).reshape(-1, 4),
              "labels": torch.tensor([g[4] for g in gts], dtype=torch.long)}],
        )

        # failure breakdown at operating threshold
        preds = sorted([(tuple(float(v) for v in b), int(c), float(s))
                        for b, c, s in zip(det.xyxy, det.class_id, det.confidence) if s >= PRED_THR_FAIL],
                       key=lambda x: -x[2])
        tp, mis, fp, missed, confs = match(preds, gts)
        a = agg[grp]
        a["tp"] += len(tp); a["mis"] += len(mis); a["fp"] += len(fp); a["missed"] += len(missed); a["gt"] += len(gts)
        a["per_cam"][row["cam"]][0] += len(tp)
        a["per_cam"][row["cam"]][1] += len(gts)
        for g in gts:
            a["cls_gt"][g[4]] += 1; a["size_gt"][size_bin(g)] += 1
        for g in missed:
            a["cls_missed"][g[4]] += 1; a["size_missed"][size_bin(g)] += 1
        for (_b, c, _s) in fp:
            a["cls_fp"][c] += 1
        for gc, pc in confs:
            a["conf"][(gc, pc)] += 1

    # ---------- report ----------
    print("\n" + "=" * 70)
    print(f"DIAGNOSIS — {CKPT.split('/')[-1]}  (IoU={IOU_THR}, op-thr={PRED_THR_FAIL})")
    print("=" * 70)
    for grp in ("SEEN", "HELDOUT"):
        a = agg[grp]
        m = maps[grp].compute()
        denom_rec = a["tp"] + a["missed"] + a["mis"]
        denom_prec = a["tp"] + a["fp"] + a["mis"]
        rec = a["tp"]/max(1, denom_rec)
        prec = a["tp"]/max(1, denom_prec)
        print(f"\n### {grp}  ({n_img[grp]} imgs, {a['gt']} GT objs)")
        print(f"  COCO mAP@50:95={float(m['map']):.3f}  mAP@50={float(m['map_50']):.3f}  mAR@100={float(m['mar_100']):.3f}")
        print(f"  greedy recall={rec:.3f}  precision={prec:.3f}   "
              f"(TP={a['tp']} missed={a['missed']} FP={a['fp']} misclass={a['mis']})")
        print(f"  size-stratified MISS RATE (missed/GT):")
        for sz in ("small", "medium", "large"):
            g_ = a["size_gt"][sz]; mm = a["size_missed"][sz]
            if g_:
                print(f"     {sz:7s}: {mm:4d}/{g_:4d}  ({mm/g_*100:5.1f}%)")
        print(f"  per-class recall (TP-ish via miss) / FP:")
        for c in range(10):
            if a["cls_gt"][c]:
                miss = a["cls_missed"][c]; gtc = a["cls_gt"][c]
                print(f"     {SHORT[c]:10s} GT={gtc:4d} missed={miss:4d} ({miss/gtc*100:5.1f}%)  FP={a['cls_fp'][c]}")
        if a["conf"]:
            print("  top confusions (GT→pred):", ", ".join(f"{SHORT[gc]}→{SHORT[pc]}:{n}" for (gc, pc), n in a["conf"].most_common(5)))

    print("\n### per-HELDOUT-camera recall")
    for cam, (tp, g_) in sorted(agg["HELDOUT"]["per_cam"].items()):
        print(f"  {cam:32s} recall(approx)={tp/max(1,g_):.3f}  (TP {tp}/{g_} GT)")

    print("\n[done]")


if __name__ == "__main__":
    main()
