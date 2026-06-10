"""Run Dima's v16 (RF-DETR @1120, best single model: EMA 0.3886) on the held-out images.

Local main env, rf-detr source on PYTHONPATH. Produces the v16 prediction pool for the ensemble:
@1120 (native) and @1120 horizontally flipped — index-aligned with all other caches. Flip boxes are
mirrored back to original coords. Same canonical class order as v7 (same RFDETRLarge head).

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 PYTHONPATH=rf-detr/src python scripts/v16_infer.py
"""
import json
import pickle
from pathlib import Path

from PIL import Image, ImageOps

from rfdetr import RFDETRLarge

REPO = Path(__file__).resolve().parent.parent
CKPT = str(REPO / "weights" / "v16_best_ema.pth")
RES = 1120  # v16's native trained resolution
THR = 0.05

print(f"[v16] building RFDETRLarge(res={RES}) + loading {Path(CKPT).name} ...")
model = RFDETRLarge(num_classes=10, resolution=RES, pretrain_weights=CKPT)
paths = json.load(open(REPO / ".data" / "heldout_paths.json"))
print(f"[v16] {len(paths)} images, conf>={THR}, passes: native + hflip")
out = {"v16_1120": [], "v16_1120_flip": []}
for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    w = img.width
    det = model.predict(img, threshold=THR)
    out["v16_1120"].append([
        ((float(b[0]), float(b[1]), float(b[2]), float(b[3])), int(c), float(s))
        for b, c, s in zip(det.xyxy, det.class_id, det.confidence)])
    detf = model.predict(ImageOps.mirror(img), threshold=THR)
    out["v16_1120_flip"].append([
        ((float(w - b[2]), float(b[1]), float(w - b[0]), float(b[3])), int(c), float(s))
        for b, c, s in zip(detf.xyxy, detf.class_id, detf.confidence)])
    if (i + 1) % 6 == 0:
        print(f"   ...{i + 1}/{len(paths)}")
with open(REPO / ".data" / "v16_preds.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"[v16] saved -> .data/v16_preds.pkl "
      f"(native {sum(len(x) for x in out['v16_1120'])} boxes, "
      f"flip {sum(len(x) for x in out['v16_1120_flip'])} boxes)")
