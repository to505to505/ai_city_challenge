"""SAHI tuning harness: run v7 inference ONCE (full frame + tiles), cache the raw predictions to disk,
then evaluate many merge strategies OFFLINE (instant) — so we can convert SAHI's small-object recall
gain into an actual mAP gain without re-running 35-min inference per idea.

Out-of-the-box SAHI (test_sahi_wbf.py) gave +28pp small-object recall but -0.03 mAP (tile false
positives on medium/large objects cost precision). The fix the data points to: keep baseline
detections for large objects, add tile detections ONLY where SAHI helps (small). This harness
sweeps that.

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/sahi_tune.py [weights/v7_best_ema.pth] [896]
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
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "weights" / "v7_best_ema.pth")
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 896
CACHE = REPO / ".data" / f"sahi_cache_{Path(CKPT).stem}_{RES}.pkl"
COLS, ROWS, OV = 3, 2, 0.2
FULL_THR, TILE_THR, EDGE_MARGIN = 0.05, 0.10, 4  # low thresholds at cache time; filter conf offline


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def sz(box):
    return ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5


def size_bin(box):
    s = sz(box)
    return "small" if s < 32 else ("medium" if s < 96 else "large")


def gt_boxes(row):
    w, h = row["width"], row["height"]
    return [((b["top_left_x"] * w, b["top_left_y"] * h, (b["top_left_x"] + b["width"]) * w,
             (b["top_left_y"] + b["height"]) * h), int(b["class_idx"]))
            for b in (row["bboxes"] or []) if b.get("task_name") == "object_detection"]


def tile_rects(w, h):
    tw = w / (COLS - (COLS - 1) * OV)
    th = h / (ROWS - (ROWS - 1) * OV)
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            x = round(c * tw * (1 - OV))
            y = round(r * th * (1 - OV))
            out.append((x, y, min(w, round(x + tw)), min(h, round(y + th))))
    return out


def predict_boxes(model, img, thr, off=(0, 0)):
    det = model.predict(img, threshold=thr)
    ox, oy = off
    return [((float(b[0]) + ox, float(b[1]) + oy, float(b[2]) + ox, float(b[3]) + oy), int(c), float(s))
            for b, c, s in zip(det.xyxy, det.class_id, det.confidence)]


def cut_by_seam(box, tile, w, h, m=EDGE_MARGIN):
    bx1, by1, bx2, by2 = box
    tx1, ty1, tx2, ty2 = tile
    return ((tx1 > 0 and bx1 <= tx1 + m) or (ty1 > 0 and by1 <= ty1 + m) or
            (tx2 < w and bx2 >= tx2 - m) or (ty2 < h and by2 >= ty2 - m))


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
            else:
                clusters.append({"boxes": [box], "scores": [sc], "fused": box})
        for cl in clusters:
            out.append((cl["fused"], cls, max(cl["scores"])))
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
    print(f"[cache] loading {CKPT} @ {RES} ...")
    model = RFDETRLarge(num_classes=10, resolution=RES, pretrain_weights=CKPT)
    model.optimize_for_inference()
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    held = df.filter(pl.col("split").is_in(["train", "validation"]) & pl.col("cam").is_in(list(HELDOUT)))
    print(f"[cache] {len(held)} held-out images — running inference (full + tiles) once ...")
    cache = []
    for i in range(len(held)):
        row = held.row(i, named=True)
        img = Image.open(row["file_path"]).convert("RGB")
        w, h = img.size
        full = predict_boxes(model, img, FULL_THR)
        tiles = []
        for tr in tile_rects(w, h):
            for d in predict_boxes(model, img.crop(tr), TILE_THR, off=(tr[0], tr[1])):
                if not cut_by_seam(d[0], tr, w, h):
                    tiles.append(d)
        cache.append({"w": w, "h": h, "gts": gt_boxes(row), "full": full, "tiles": tiles})
        if (i + 1) % 12 == 0:
            print(f"  ...{i + 1}/{len(held)}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"[cache] saved {CACHE}")
    return cache


# Merge strategies. mode:
#   raw  = full predictions untouched (no merge) — the true baseline
#   wbf  = weighted-box-fusion over full + gated tiles (re-clusters everything)
#   add  = keep full UNTOUCHED, only APPEND gated tile boxes that don't duplicate a full box
#          (same class, IoU >= dup_iou) — preserves baseline precision, adds small-object recall
STRATEGIES = {
    "baseline_raw":   dict(mode="raw"),
    "wbf_all":        dict(mode="wbf", tile_conf=0.25, size_max=None),
    "add_small64":    dict(mode="add", tile_conf=0.30, size_max=64),
    "add_small48_t30": dict(mode="add", tile_conf=0.30, size_max=48),
    "add_small48_t40": dict(mode="add", tile_conf=0.40, size_max=48),
    "add_small48_t50": dict(mode="add", tile_conf=0.50, size_max=48),
    "boost_s48_t30":  dict(mode="boost", tile_conf=0.30, size_max=48),
    "boost_s48_t40":  dict(mode="boost", tile_conf=0.40, size_max=48),
    "boost_s64_t30":  dict(mode="boost", tile_conf=0.30, size_max=64),
}


def apply_strategy(rec, cfg):
    full = list(rec["full"])
    mode = cfg.get("mode", "raw")
    if mode == "raw":
        return full
    tc = cfg.get("tile_conf", 0.25)
    smax = cfg.get("size_max", None)
    tiles = [d for d in rec["tiles"] if d[2] >= tc and (smax is None or sz(d[0]) < smax)]
    if mode == "wbf":
        return wbf(full + tiles)
    dup_iou = cfg.get("dup_iou", 0.5)
    if mode == "add":
        # append only non-duplicate tile boxes; full kept exactly as-is
        kept = list(full)
        for d in tiles:
            if all(d[1] != k[1] or iou(d[0], k[0]) < dup_iou for k in kept):
                kept.append(d)
        return kept
    # mode == "boost": keep full box positions; raise a full box's confidence to the tile's when a
    # tile confirms it (small object weakly seen full + strongly in tile), and append genuinely-new ones.
    kept = [[b, c, s] for (b, c, s) in full]
    for d in tiles:
        best, bi = -1, dup_iou
        for k, kk in enumerate(kept):
            if kk[1] == d[1]:
                v = iou(d[0], kk[0])
                if v >= bi:
                    bi, best = v, k
        if best >= 0:
            kept[best][2] = max(kept[best][2], d[2])
        else:
            kept.append([d[0], d[1], d[2]])
    return [tuple(x) for x in kept]


def main():
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else build_cache()
    print(f"\n[eval] {len(cache)} images, {sum(len(r['gts']) for r in cache)} GT boxes")
    names = list(STRATEGIES)
    mp = {k: MeanAveragePrecision(box_format="xyxy") for k in names}
    rs = {k: {s: [0, 0] for s in ("small", "medium", "large")} for k in names}
    for rec in cache:
        tgt = [{"boxes": torch.tensor([list(g[0]) for g in rec["gts"]]).reshape(-1, 4),
                "labels": torch.tensor([g[1] for g in rec["gts"]], dtype=torch.long)}]
        for k in names:
            preds = apply_strategy(rec, STRATEGIES[k])
            mp[k].update([to_tm(preds)], tgt)
            for s2, (m2, t2) in recall_by_size(preds, rec["gts"]).items():
                rs[k][s2][0] += m2
                rs[k][s2][1] += t2
    res = {k: mp[k].compute() for k in names}
    print("\n" + "=" * 86)
    print(f"{'strategy':14s} {'mAP50:95':>9s} {'mAP50':>8s} {'mAR100':>8s}   {'small-rec':>9s} {'med-rec':>8s} {'large':>7s}")
    print("=" * 86)
    base = float(res["baseline_raw"]["map"])
    for k in names:
        r = res[k]
        sm = rs[k]["small"][0] / max(1, rs[k]["small"][1]) * 100
        md = rs[k]["medium"][0] / max(1, rs[k]["medium"][1]) * 100
        lg = rs[k]["large"][0] / max(1, rs[k]["large"][1]) * 100
        d = float(r["map"]) - base
        flag = "  <-- baseline" if k == "baseline_raw" else (f"  ({'+' if d >= 0 else ''}{d:.3f} vs base)")
        print(f"{k:14s} {float(r['map']):9.3f} {float(r['map_50']):8.3f} {float(r['mar_100']):8.3f}   "
              f"{sm:8.1f}% {md:7.1f}% {lg:6.1f}%{flag}")
    print("\n[done] best mAP strategy:", max(names, key=lambda k: float(res[k]["map"])))


if __name__ == "__main__":
    main()
