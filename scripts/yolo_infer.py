"""Run our trained YOLO26-L (v8) on the held-out images and dump predictions.

Runs in the MAIN env (Ultralytics is pure-Python, no Docker needed unlike mmdet). Reads the same
ordered held-out image paths as the v7 cache and writes per-image predictions (box xyxy in original
image pixels, class 0-9, score) — index-aligned with `.data/tta_ens_cache.pkl` and
`.data/convnext_preds.pkl` so the main env can WBF-ensemble all three.

v8's class order is the dataset's canonical order (same as v7/GT/ConvNeXt) — VERIFIED downstream by
its standalone mAP in ensemble_eval.py (a class permutation would drop solo mAP to ~0; it scores ~0.25).

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/yolo_infer.py
"""
import json
import pickle
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
CKPT = str(REPO / "weights" / "v8_best.pt")
IMGSZ = 1280  # v8 trained at 1280
THR = 0.05

print(f"[yolo26] loading {Path(CKPT).name} ...")
model = YOLO(CKPT)
paths = json.load(open(REPO / ".data" / "heldout_paths.json"))
print(f"[yolo26] {len(paths)} images @ imgsz={IMGSZ}, conf>={THR}, classes={len(model.names)}")
out = []
for i, p in enumerate(paths):
    r = model.predict(p, imgsz=IMGSZ, conf=THR, verbose=False)[0]
    b = r.boxes
    preds = [((float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])), int(c), float(s))
             for xy, c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist())]
    out.append(preds)
    if (i + 1) % 12 == 0:
        print(f"   ...{i + 1}/{len(paths)}")
with open(REPO / ".data" / "yolo26_preds.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"[yolo26] saved {len(out)} images -> .data/yolo26_preds.pkl "
      f"({sum(len(x) for x in out)} total boxes)")
