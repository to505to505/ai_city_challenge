"""Run our trained VFNet R-50 (mmdet) on the held-out images and dump predictions.

Runs inside the linux/amd64 Docker container (same OpenMMLab CPU stack as convnext_infer.py — mmcv's
prebuilt CPU wheels include the compiled deform_conv2d op the VFNet head needs). Reads the ordered
held-out image paths, writes per-image predictions (box xyxy in original-image coords, class 0-9,
score), index-aligned with the v7/ConvNeXt/YOLO/DINOv3 caches.

Class order is the canonical dataset order (the config's `classes` tuple) — verified downstream by
its non-zero solo mAP in ensemble_eval.py (a permutation would collapse it to ~0; expect ~0.29).

    python scripts/vfnet_infer.py   # inside the container, repo mounted at the same abs path
"""
import json
import pickle
from pathlib import Path

from mmdet.apis import inference_detector, init_detector

REPO = Path(__file__).resolve().parent.parent
CONFIG = str(REPO / "trainer-vfnet" / "configs" / "vfnet_eccv.py")
CKPT = str(REPO / "weights" / "vfnet_eccv_best.pth")
THR = 0.05

print(f"[vfnet] init_detector({Path(CONFIG).name}, {Path(CKPT).name}) on cpu ...")
model = init_detector(CONFIG, CKPT, device="cpu")
paths = json.load(open(REPO / ".data" / "heldout_paths.json"))
print(f"[vfnet] {len(paths)} images")
out = []
for i, p in enumerate(paths):
    res = inference_detector(model, p)
    inst = res.pred_instances
    boxes = inst.bboxes.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    labels = inst.labels.cpu().numpy()
    preds = [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), int(l), float(s))
             for b, l, s in zip(boxes, labels, scores) if s >= THR]
    out.append(preds)
    if (i + 1) % 12 == 0:
        print(f"   ...{i + 1}/{len(paths)}")
with open(REPO / ".data" / "vfnet_preds.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"[vfnet] saved {len(out)} images -> .data/vfnet_preds.pkl "
      f"({sum(len(x) for x in out)} total boxes)")
