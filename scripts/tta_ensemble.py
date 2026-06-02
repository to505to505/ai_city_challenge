"""TTA + ensemble harness (mAP is the Track-6 metric, so we optimize mAP, not recall).

Runs each model ONCE over the held-out cameras, caches raw predictions, then merges offline (instant)
so every TTA / ensemble combination is comparable to the v7 baseline (0.380 on these 36 imgs):
  - baseline      : v7 @896, full frame
  - flip_tta      : v7 @896 + horizontal-flip
  - ms_tta        : v7 @896 + v7 @1024
  - full_tta      : v7 @896 + flip + @1024
  - ens_v7v6      : v7 @896 + v6 @704 (architecture-lineage ensemble)
  - tta+ens       : everything

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/tta_ensemble.py
"""
from __future__ import annotations

import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rf-detr" / "src"))
import transformers  # noqa: E402
from transformers.utils import backbone_utils as _bu  # noqa: E402

for _n in ("BackboneConfigMixin", "BackboneMixin", "BackboneType"):
    if not hasattr(transformers, _n) and hasattr(_bu, _n):
        setattr(transformers, _n, getattr(_bu, _n))
import polars as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402

HELDOUT = {"5 POINTS WB", "GRANDVIEW - DELHI INT", "HWY 20 - OLD HWY WBA",
           "LOCUST CONNECTOR NB", "NW ARTERIAL - CHAVENELLE INT", "US 61 - TWIN VALLEY INT"}
CACHE = REPO / ".data" / "tta_ens_cache.pkl"
FULL_THR = 0.05
# (cache-name, checkpoint, resolution, also-run-flipped)
MODELS = [
    ("v7_896", REPO / "weights" / "v7_best_ema.pth", 896, True),
    ("v7_1024", REPO / "weights" / "v7_best_ema.pth", 1024, False),
    ("v6_704", REPO / "weights" / "v6_best_ema.pth", 704, False),
]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def size_bin(box):
    s = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
    return "small" if s < 32 else ("medium" if s < 96 else "large")


def gt_boxes(row):
    w, h = row["width"], row["height"]
    return [((b["top_left_x"] * w, b["top_left_y"] * h, (b["top_left_x"] + b["width"]) * w,
             (b["top_left_y"] + b["height"]) * h), int(b["class_idx"]))
            for b in (row["bboxes"] or []) if b.get("task_name") == "object_detection"]


def predict_boxes(model, img, thr=FULL_THR):
    det = model.predict(img, threshold=thr)
    return [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), int(c), float(s))
            for b, c, s in zip(det.xyxy, det.class_id, det.confidence)]


def wbf(dets, iou_thr=0.55):
    by_cls = defaultdict(list)
    for d in dets:
        by_cls[d[1]].append(d)
    out = []
    for cls, ds in by_cls.items():
        clusters = []
        for box, _c, sc in sorted(ds, key=lambda x: -x[2]):
            bi, bk = iou_thr, -1
            for k, cl in enumerate(clusters):
                v = iou(box, cl["fused"])
                if v > bi:
                    bi, bk = v, k
            if bk >= 0:
                cl = clusters[bk]
                cl["boxes"].append(box)
                cl["scores"].append(sc)
                wt = sum(cl["scores"])
                cl["fused"] = tuple(sum(b[j] * s for b, s in zip(cl["boxes"], cl["scores"])) / wt for j in range(4))
                cl["n"] += 1
            else:
                clusters.append({"boxes": [box], "scores": [sc], "fused": box, "n": 1})
        for cl in clusters:
            # average confidence over the cluster (standard WBF) — rewards boxes seen by multiple views
            out.append((cl["fused"], cls, sum(cl["scores"]) / len(cl["scores"])))
    return out


def to_tm(preds):
    if not preds:
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)}
    return {"boxes": torch.tensor([list(p[0]) for p in preds]), "scores": torch.tensor([p[2] for p in preds]),
            "labels": torch.tensor([p[1] for p in preds], dtype=torch.long)}


def recall_by_size(preds, gts, op=0.30):
    preds = sorted([p for p in preds if p[2] >= op], key=lambda x: -x[2])
    matched = [False] * len(gts)
    res = {s: [0, 0] for s in ("small", "medium", "large")}
    for g in gts:
        res[size_bin(g[0])][1] += 1
    for box, cls, _ in preds:
        bi, bj = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou(box, g[0])
            if v > bi:
                bi, bj = v, j
        if bi >= 0.5 and bj >= 0 and gts[bj][1] == cls:
            matched[bj] = True
            res[size_bin(gts[bj][0])][0] += 1
    return res


def build_cache():
    from rfdetr import RFDETRLarge
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    held = df.filter(pl.col("split").is_in(["train", "validation"]) & pl.col("cam").is_in(list(HELDOUT)))
    rows = [held.row(i, named=True) for i in range(len(held))]
    print(f"[cache] {len(rows)} held-out images")
    cache = [{"w": r["width"], "h": r["height"], "gts": gt_boxes(r), "preds": {}} for r in rows]
    for name, ckpt, res, do_flip in MODELS:
        print(f"[cache] {name}: loading {Path(ckpt).name} @ {res} ...")
        model = RFDETRLarge(num_classes=10, resolution=res, pretrain_weights=str(ckpt))
        model.optimize_for_inference()
        for i, r in enumerate(rows):
            img = Image.open(r["file_path"]).convert("RGB")
            w = r["width"]
            cache[i]["preds"][name] = predict_boxes(model, img)
            if do_flip:
                fimg = img.transpose(Image.FLIP_LEFT_RIGHT)
                cache[i]["preds"][name + "_flip"] = [((w - b[2], b[1], w - b[0], b[3]), c, s)
                                                     for b, c, s in predict_boxes(model, fimg)]
            if (i + 1) % 12 == 0:
                print(f"    ...{i + 1}/{len(rows)}")
        del model
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"[cache] saved {CACHE}")
    return cache


STRATEGIES = {
    "baseline":   ["v7_896"],
    "flip_tta":   ["v7_896", "v7_896_flip"],
    "ms_tta":     ["v7_896", "v7_1024"],
    "full_tta":   ["v7_896", "v7_896_flip", "v7_1024"],
    "ens_v7v6":   ["v7_896", "v6_704"],
    "tta_ens":    ["v7_896", "v7_896_flip", "v7_1024", "v6_704"],
}


def main():
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else build_cache()
    print(f"\n[eval] {len(cache)} imgs, {sum(len(r['gts']) for r in cache)} GT")
    names = list(STRATEGIES)
    mp = {k: MeanAveragePrecision(box_format="xyxy") for k in names}
    rs = {k: {s: [0, 0] for s in ("small", "medium", "large")} for k in names}
    for rec in cache:
        tgt = [{"boxes": torch.tensor([list(g[0]) for g in rec["gts"]]).reshape(-1, 4),
                "labels": torch.tensor([g[1] for g in rec["gts"]], dtype=torch.long)}]
        for k in names:
            dets = []
            for src in STRATEGIES[k]:
                dets += rec["preds"].get(src, [])
            preds = rec["preds"]["v7_896"] if k == "baseline" else wbf(dets)
            mp[k].update([to_tm(preds)], tgt)
            for s2, (m2, t2) in recall_by_size(preds, rec["gts"]).items():
                rs[k][s2][0] += m2
                rs[k][s2][1] += t2
    res = {k: mp[k].compute() for k in names}
    base = float(res["baseline"]["map"])
    print("\n" + "=" * 80)
    print(f"{'strategy':12s} {'mAP50:95':>9s} {'mAP50':>8s} {'mAR100':>8s}   {'small':>6s} {'med':>6s} {'large':>6s}")
    print("=" * 80)
    for k in names:
        r = res[k]
        rec3 = "  ".join(f"{rs[k][s][0]/max(1,rs[k][s][1])*100:5.1f}%" for s in ("small", "medium", "large"))
        d = float(r["map"]) - base
        flag = "  <-- base" if k == "baseline" else f"  ({'+' if d >= 0 else ''}{d:.3f})"
        print(f"{k:12s} {float(r['map']):9.3f} {float(r['map_50']):8.3f} {float(r['mar_100']):8.3f}   {rec3}{flag}")
    print("\n[done] best mAP:", max(names, key=lambda k: float(res[k]["map"])))


if __name__ == "__main__":
    main()
