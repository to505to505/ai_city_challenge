"""Quick architectural probe: does a high-res CNN detector (YOLO11x, zero-shot COCO) catch
the small objects that RF-DETR (704) misses on the held-out cameras?

NOT a fair task benchmark — YOLO is zero-shot and COCO has no vehicle subtypes — so we compare
at a COARSE level (vehicle / person) and focus on RECALL BY OBJECT SIZE, which is the
architectural question. A trained YOLO would only do better than this zero-shot lower bound.

    python scripts/compare_yolo.py [--imgsz 1280]
"""
from __future__ import annotations
import sys, warnings, argparse
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
from PIL import Image  # noqa: E402
from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402

HELDOUT_CAMS = {"5 POINTS WB", "GRANDVIEW - DELHI INT", "HWY 20 - OLD HWY WBA",
                "LOCUST CONNECTOR NB", "NW ARTERIAL - CHAVENELLE INT", "US 61 - TWIN VALLEY INT"}
RFDETR_CKPT = str(REPO_ROOT / "weights" / "v5_best_ema.pth")
# coarse buckets
GT_COARSE = {0: "veh", 1: "veh", 2: "veh", 3: "veh", 4: "veh", 5: "veh", 6: "two", 7: "two", 8: "veh", 9: "person"}
COCO_COARSE = {2: "veh", 5: "veh", 7: "veh", 0: "person", 1: "two", 3: "two"}  # car/bus/truck, person, bike/moto


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1)
    inter = iw*ih
    return 0. if inter <= 0 else inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9)


def size_bin(box):
    s = ((box[2]-box[0])*(box[3]-box[1]))**0.5
    return "small" if s < 32 else ("medium" if s < 96 else "large")


def gt_coarse(row):
    W, H = row["width"], row["height"]
    out = []
    for b in (row["bboxes"] or []):
        if b.get("task_name") != "object_detection":
            continue
        c = GT_COARSE.get(int(b["class_idx"]))
        if c:
            out.append(((b["top_left_x"]*W, b["top_left_y"]*H, (b["top_left_x"]+b["width"])*W, (b["top_left_y"]+b["height"])*H), c))
    return out


def recall_by_size(preds, gts):
    """preds: list (box, coarse, conf) ; gts: list (box, coarse). Greedy coarse match @0.5."""
    preds = sorted(preds, key=lambda x: -x[2]); matched = [False]*len(gts)
    res = {(sz, cl): [0, 0] for sz in ("small", "medium", "large") for cl in ("veh", "person", "two")}
    for box, c in gts:
        res[(size_bin(box), c)][1] += 1
    for box, c, _s in preds:
        bi, bj = 0., -1
        for j, g in enumerate(gts):
            if matched[j] or g[1] != c:
                continue
            v = iou(box, g[0])
            if v > bi:
                bi, bj = v, j
        if bi >= 0.5 and bj >= 0:
            matched[bj] = True; res[(size_bin(gts[bj][0]), c)][0] += 1
    return res


def add(acc, r):
    for k, (m, t) in r.items():
        acc[k][0] += m; acc[k][1] += t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    held = df.filter(pl.col("split").is_in(["train", "validation"]) & pl.col("cam").is_in(list(HELDOUT_CAMS)))
    print(f"[data] {len(held)} held-out images")

    from rfdetr import RFDETRLarge
    rf = RFDETRLarge(num_classes=10, resolution=704, pretrain_weights=RFDETR_CKPT); rf.optimize_for_inference()
    from ultralytics import YOLO
    yolo = YOLO("yolo11x.pt")

    acc = {m: {(sz, cl): [0, 0] for sz in ("small", "medium", "large") for cl in ("veh", "person", "two")}
           for m in ("rfdetr704", f"yolo11x{args.imgsz}")}
    for i in range(len(held)):
        row = held.row(i, named=True); img = Image.open(row["file_path"]).convert("RGB")
        gts = gt_coarse(row)
        det = rf.predict(img, threshold=args.conf)
        rfp = [(tuple(float(v) for v in b), GT_COARSE.get(int(c)), float(s)) for b, c, s in zip(det.xyxy, det.class_id, det.confidence) if GT_COARSE.get(int(c))]
        add(acc["rfdetr704"], recall_by_size(rfp, gts))
        yr = yolo.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        yp = []
        for b in yr.boxes:
            cl = COCO_COARSE.get(int(b.cls[0]))
            if cl:
                xy = b.xyxy[0].tolist(); yp.append((tuple(xy), cl, float(b.conf[0])))
        add(acc[f"yolo11x{args.imgsz}"], recall_by_size(yp, gts))
        if (i+1) % 12 == 0:
            print(f"  ...{i+1}/{len(held)}")

    print("\n" + "="*60)
    print("COARSE recall@0.5 on HELD-OUT cameras (vehicle / person)")
    print("RF-DETR is trained on this data @704; YOLO is ZERO-SHOT COCO @%d" % args.imgsz)
    print("="*60)
    models = list(acc.keys())
    for cl in ("veh", "person"):
        print(f"\n  {cl.upper()}")
        for sz in ("small", "medium", "large"):
            tot = acc[models[0]][(sz, cl)][1]
            if not tot:
                continue
            vals = "   ".join(f"{m}: {acc[m][(sz,cl)][0]/max(1,acc[m][(sz,cl)][1])*100:5.1f}%" for m in models)
            print(f"     {sz:7s} ({tot:4d} GT):   {vals}")
    print("\n[done]")


if __name__ == "__main__":
    main()
