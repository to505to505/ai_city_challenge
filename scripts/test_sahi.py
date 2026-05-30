"""Test tiled (SAHI-style) inference vs plain full-image inference on the 36 held-out-camera
images, to see if slicing recovers the small/medium objects the model misses cross-camera.

Tiling: full image + a cols×rows grid of overlapping tiles. Each tile is predicted at the
model's native resolution (so a small distant object gets many more effective pixels than in
the downscaled full frame), boxes are offset back to full-image coords, then everything is
merged with class-aware greedy NMS.

    python scripts/test_sahi.py [weights/v5_best_ema.pth]
"""
from __future__ import annotations
import sys, warnings
from collections import Counter
from pathlib import Path
warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rf-detr" / "src"))
import transformers  # noqa: E402
from transformers.utils import backbone_utils as _bu  # noqa: E402
for _n in ("BackboneConfigMixin", "BackboneMixin", "BackboneType"):
    if not hasattr(transformers, _n) and hasattr(_bu, _n):
        setattr(transformers, _n, getattr(_bu, _n))
import polars as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402
from rfdetr import RFDETRLarge  # noqa: E402

SHORT = ["Car", "Pickup", "SingleTrk", "ComboTrk", "HeavyDuty", "Trailer", "Moto", "Bicycle", "Van", "Person"]
HELDOUT_CAMS = {"5 POINTS WB", "GRANDVIEW - DELHI INT", "HWY 20 - OLD HWY WBA",
                "LOCUST CONNECTOR NB", "NW ARTERIAL - CHAVENELLE INT", "US 61 - TWIN VALLEY INT"}
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "weights" / "v5_best_ema.pth")
COLS, ROWS, OV = 3, 2, 0.2
TILE_THR, FULL_THR_MAP, OP_THR, NMS_IOU = 0.2, 0.05, 0.30, 0.55


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1)
    inter = iw*ih
    return 0. if inter <= 0 else inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9)


def gt_boxes(row):
    W, H = row["width"], row["height"]
    return [((b["top_left_x"]*W, b["top_left_y"]*H, (b["top_left_x"]+b["width"])*W, (b["top_left_y"]+b["height"])*H), int(b["class_idx"]))
            for b in (row["bboxes"] or []) if b.get("task_name") == "object_detection"]


def size_bin(box):
    side = ((box[2]-box[0])*(box[3]-box[1]))**0.5
    return "small" if side < 32 else ("medium" if side < 96 else "large")


def tile_rects(W, H):
    tw = W / (COLS - (COLS-1)*OV); th = H / (ROWS - (ROWS-1)*OV)
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            x = round(c*tw*(1-OV)); y = round(r*th*(1-OV))
            out.append((x, y, min(W, round(x+tw)), min(H, round(y+th))))
    return out


def predict_boxes(model, img, thr, off=(0, 0)):
    det = model.predict(img, threshold=thr)
    ox, oy = off
    return [((float(b[0])+ox, float(b[1])+oy, float(b[2])+ox, float(b[3])+oy), int(c), float(s))
            for b, c, s in zip(det.xyxy, det.class_id, det.confidence)]


def nms(dets, iou_thr=NMS_IOU):
    dets = sorted(dets, key=lambda d: -d[2]); keep = []
    for d in dets:
        if all(d[1] != k[1] or iou(d[0], k[0]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def recall_by_size(preds, gts):
    """greedy match @0.5; returns dict size-> [matched, total]."""
    preds = sorted(preds, key=lambda x: -x[2]); matched = [False]*len(gts)
    res = {s: [0, 0] for s in ("small", "medium", "large")}
    for g in gts:
        res[size_bin(g[0])][1] += 1
    for box, cls, conf in preds:
        bi, bj = 0., -1
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


def to_tm(preds):
    if not preds:
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)}
    return {"boxes": torch.tensor([list(p[0]) for p in preds]), "scores": torch.tensor([p[2] for p in preds]),
            "labels": torch.tensor([p[1] for p in preds], dtype=torch.long)}


def main():
    print(f"[load] {CKPT}  | tiling {COLS}x{ROWS} ov={OV}")
    model = RFDETRLarge(num_classes=10, resolution=704, pretrain_weights=CKPT); model.optimize_for_inference()
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    held = df.filter(pl.col("split").is_in(["train", "validation"]) & pl.col("cam").is_in(list(HELDOUT_CAMS)))
    print(f"[data] {len(held)} held-out images")
    from torchmetrics.detection import MeanAveragePrecision
    m_base, m_sahi = MeanAveragePrecision(box_format="xyxy"), MeanAveragePrecision(box_format="xyxy")
    rs_base = {s: [0, 0] for s in ("small", "medium", "large")}
    rs_sahi = {s: [0, 0] for s in ("small", "medium", "large")}
    for i in range(len(held)):
        row = held.row(i, named=True)
        img = Image.open(row["file_path"]).convert("RGB")
        W, H = img.size
        gts = gt_boxes(row)
        tgt = [{"boxes": torch.tensor([list(g[0]) for g in gts]).reshape(-1, 4),
                "labels": torch.tensor([g[1] for g in gts], dtype=torch.long)}]
        # baseline: full only
        full = predict_boxes(model, img, FULL_THR_MAP)
        m_base.update([to_tm(full)], tgt)
        for sz, (mt, _t) in recall_by_size([p for p in full if p[2] >= OP_THR], gts).items():
            rs_base[sz][0] += mt; rs_base[sz][1] += _t
        # SAHI: full + tiles, merged
        alld = list(full)
        for (x1, y1, x2, y2) in tile_rects(W, H):
            alld += predict_boxes(model, img.crop((x1, y1, x2, y2)), TILE_THR, off=(x1, y1))
        merged = nms(alld)
        m_sahi.update([to_tm(merged)], tgt)
        for sz, (mt, t) in recall_by_size([p for p in merged if p[2] >= OP_THR], gts).items():
            rs_sahi[sz][0] += mt; rs_sahi[sz][1] += t
        if (i+1) % 12 == 0:
            print(f"  ...{i+1}/{len(held)}")

    rb, rsa = m_base.compute(), m_sahi.compute()
    print("\n" + "="*60)
    print("HELD-OUT (36 imgs):  BASELINE (full)   vs   SAHI (full+3x2 tiles)")
    print("="*60)
    print(f"  mAP@50:95   {float(rb['map']):.3f}   ->   {float(rsa['map']):.3f}")
    print(f"  mAP@50      {float(rb['map_50']):.3f}   ->   {float(rsa['map_50']):.3f}")
    print(f"  mAR@100     {float(rb['mar_100']):.3f}   ->   {float(rsa['mar_100']):.3f}")
    print(f"\n  recall@0.5 by size (op-thr {OP_THR}):  baseline -> SAHI")
    for sz in ("small", "medium", "large"):
        b = rs_base[sz][0]/max(1, rs_base[sz][1]); s = rs_sahi[sz][0]/max(1, rs_sahi[sz][1])
        print(f"     {sz:7s} ({rs_base[sz][1]:4d} GT):  {b*100:5.1f}%  ->  {s*100:5.1f}%")
    print("\n[done]")


if __name__ == "__main__":
    main()
