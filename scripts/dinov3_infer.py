"""Run our trained DINOv3-RF-DETR (v9) on the held-out images and dump predictions.

Runs in the MAIN env with the rf-detr source on PYTHONPATH:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=rf-detr/src python scripts/dinov3_infer.py

v9 is RF-DETR with the DINOv3-S backbone (RoPE, non-windowed), trained at 704. The trained checkpoint
already contains the DINOv3 backbone weights, so no warm-start surgery is needed — we just build the
architecture with encoder='dinov3_small' and load the checkpoint. Writes per-image predictions
(box xyxy in original-image pixels, class 0-9, score), index-aligned with the v7/ConvNeXt/YOLO caches.

Class order is the canonical dataset order (same as v7/GT) — VERIFIED by its non-zero solo mAP in
ensemble_eval.py (a permutation would drop solo mAP to ~0).
"""
import json
import pickle
from pathlib import Path

from rfdetr import RFDETRLarge

REPO = Path(__file__).resolve().parent.parent
CKPT = str(REPO / "weights" / "v9_best_ema.pth")
RES = 704  # v9's native trained resolution
THR = 0.05

print(f"[dinov3] building RFDETRLarge(encoder=dinov3_small, res={RES}) + loading {Path(CKPT).name} ...")
model = RFDETRLarge(num_classes=10, resolution=RES, encoder="dinov3_small", pretrain_weights=CKPT)
paths = json.load(open(REPO / ".data" / "heldout_paths.json"))
print(f"[dinov3] {len(paths)} images, conf>={THR}")
out = []
for i, p in enumerate(paths):
    det = model.predict(p, threshold=THR)
    preds = [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), int(c), float(s))
             for b, c, s in zip(det.xyxy, det.class_id, det.confidence)]
    out.append(preds)
    if (i + 1) % 12 == 0:
        print(f"   ...{i + 1}/{len(paths)}")
with open(REPO / ".data" / "dinov3_preds.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"[dinov3] saved {len(out)} images -> .data/dinov3_preds.pkl "
      f"({sum(len(x) for x in out)} total boxes)")
