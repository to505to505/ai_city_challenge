"""Tuned tiled inference: full frame + overlapping tiles, merged with Weighted Box Fusion
(not crude NMS), with tile-edge-box filtering and a higher tile threshold — to recover the
precision/mAP that naive SAHI lost while keeping the small-object recall gain.

Compares, on the 36 held-out-camera images:
  baseline (full)  vs  SAHI-NMS  vs  SAHI-WBF(+edge filter)

    python scripts/test_sahi_wbf.py [weights/v5_best_ema.pth]
"""
from __future__ import annotations
import sys, warnings
from collections import defaultdict
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

HELDOUT_CAMS = {"5 POINTS WB", "GRANDVIEW - DELHI INT", "HWY 20 - OLD HWY WBA",
                "LOCUST CONNECTOR NB", "NW ARTERIAL - CHAVENELLE INT", "US 61 - TWIN VALLEY INT"}
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "weights" / "v5_best_ema.pth")
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 704  # must match the checkpoint's train resolution
COLS, ROWS, OV = 3, 2, 0.2
FULL_THR, TILE_THR, OP_THR = 0.05, 0.25, 0.30
NMS_IOU, WBF_IOU, EDGE_MARGIN = 0.55, 0.55, 4


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
    s = ((box[2]-box[0])*(box[3]-box[1]))**0.5
    return "small" if s < 32 else ("medium" if s < 96 else "large")


def tile_rects(W, H):
    tw = W/(COLS-(COLS-1)*OV); th = H/(ROWS-(ROWS-1)*OV); out = []
    for r in range(ROWS):
        for c in range(COLS):
            x = round(c*tw*(1-OV)); y = round(r*th*(1-OV))
            out.append((x, y, min(W, round(x+tw)), min(H, round(y+th))))
    return out


def predict_boxes(model, img, thr, off=(0, 0)):
    det = model.predict(img, threshold=thr); ox, oy = off
    return [((float(b[0])+ox, float(b[1])+oy, float(b[2])+ox, float(b[3])+oy), int(c), float(s))
            for b, c, s in zip(det.xyxy, det.class_id, det.confidence)]


def cut_by_seam(box, tile, W, H, m=EDGE_MARGIN):
    bx1, by1, bx2, by2 = box; tx1, ty1, tx2, ty2 = tile
    return ((tx1 > 0 and bx1 <= tx1+m) or (ty1 > 0 and by1 <= ty1+m) or
            (tx2 < W and bx2 >= tx2-m) or (ty2 < H and by2 >= ty2-m))


def nms(dets, iou_thr=NMS_IOU):
    dets = sorted(dets, key=lambda d: -d[2]); keep = []
    for d in dets:
        if all(d[1] != k[1] or iou(d[0], k[0]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def wbf(dets, iou_thr=WBF_IOU):
    by_cls = defaultdict(list)
    for d in dets:
        by_cls[d[1]].append(d)
    out = []
    for cls, ds in by_cls.items():
        clusters = []  # {'boxes','scores','fused'}
        for box, _c, sc in sorted(ds, key=lambda x: -x[2]):
            bi, bk = iou_thr, -1
            for k, cl in enumerate(clusters):
                v = iou(box, cl["fused"])
                if v > bi:
                    bi, bk = v, k
            if bk >= 0:
                cl = clusters[bk]; cl["boxes"].append(box); cl["scores"].append(sc)
                Wt = sum(cl["scores"])
                cl["fused"] = tuple(sum(b[j]*s for b, s in zip(cl["boxes"], cl["scores"]))/Wt for j in range(4))
            else:
                clusters.append({"boxes": [box], "scores": [sc], "fused": box})
        for cl in clusters:
            out.append((cl["fused"], cls, max(cl["scores"])))  # keep peak conf (don't penalize single-tile small objs)
    return out


def to_tm(preds):
    if not preds:
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)}
    return {"boxes": torch.tensor([list(p[0]) for p in preds]), "scores": torch.tensor([p[2] for p in preds]),
            "labels": torch.tensor([p[1] for p in preds], dtype=torch.long)}


def recall_by_size(preds, gts):
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
            matched[bj] = True; res[size_bin(gts[bj][0])][0] += 1
    return res


def main():
    print(f"[load] {CKPT} | {COLS}x{ROWS} ov={OV} tile_thr={TILE_THR} edge_margin={EDGE_MARGIN}")
    print(f"[load] {CKPT} resolution={RES}")
    model = RFDETRLarge(num_classes=10, resolution=RES, pretrain_weights=CKPT); model.optimize_for_inference()
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    held = df.filter(pl.col("split").is_in(["train", "validation"]) & pl.col("cam").is_in(list(HELDOUT_CAMS)))
    print(f"[data] {len(held)} held-out images")
    from torchmetrics.detection import MeanAveragePrecision
    METH = ["baseline", "sahi_nms", "sahi_wbf"]
    mp = {k: MeanAveragePrecision(box_format="xyxy") for k in METH}
    rs = {k: {s: [0, 0] for s in ("small", "medium", "large")} for k in METH}
    for i in range(len(held)):
        row = held.row(i, named=True); img = Image.open(row["file_path"]).convert("RGB"); W, H = img.size
        gts = gt_boxes(row)
        tgt = [{"boxes": torch.tensor([list(g[0]) for g in gts]).reshape(-1, 4),
                "labels": torch.tensor([g[1] for g in gts], dtype=torch.long)}]
        full = predict_boxes(model, img, FULL_THR)
        tiled = []
        for tr in tile_rects(W, H):
            for d in predict_boxes(model, img.crop(tr), TILE_THR, off=(tr[0], tr[1])):
                if not cut_by_seam(d[0], tr, W, H):
                    tiled.append(d)
        cand = {"baseline": full, "sahi_nms": nms(full+tiled), "sahi_wbf": wbf(full+tiled)}
        for k in METH:
            mp[k].update([to_tm(cand[k])], tgt)
            for sz, (mt, t) in recall_by_size([p for p in cand[k] if p[2] >= OP_THR], gts).items():
                rs[k][sz][0] += mt; rs[k][sz][1] += t
        if (i+1) % 12 == 0:
            print(f"  ...{i+1}/{len(held)}")
    res = {k: mp[k].compute() for k in METH}
    print("\n" + "="*64)
    print("HELD-OUT (36 imgs)         baseline   sahi_nms   sahi_wbf")
    print("="*64)
    print(f"  mAP@50:95              {float(res['baseline']['map']):7.3f}   {float(res['sahi_nms']['map']):7.3f}   {float(res['sahi_wbf']['map']):7.3f}")
    print(f"  mAP@50                 {float(res['baseline']['map_50']):7.3f}   {float(res['sahi_nms']['map_50']):7.3f}   {float(res['sahi_wbf']['map_50']):7.3f}")
    print(f"  mAR@100                {float(res['baseline']['mar_100']):7.3f}   {float(res['sahi_nms']['mar_100']):7.3f}   {float(res['sahi_wbf']['mar_100']):7.3f}")
    print(f"\n  recall@0.5 by size (op-thr {OP_THR}):")
    for sz in ("small", "medium", "large"):
        vals = "   ".join(f"{rs[k][sz][0]/max(1,rs[k][sz][1])*100:6.1f}%" for k in METH)
        print(f"     {sz:7s} ({rs['baseline'][sz][1]:4d} GT):   {vals}")
    print("\n[done]")


if __name__ == "__main__":
    main()
