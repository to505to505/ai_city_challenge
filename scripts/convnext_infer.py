"""Run Dima's mmdet Cascade-R-CNN/ConvNeXt on the held-out images and dump predictions.

Runs in the dedicated `mmdet` conda env (torch 2.1 + OpenMMLab), NOT the main env. Reads the ordered
held-out image paths exported by the main env, runs inference, and writes per-image predictions
(box xyxy in original-image coords, class 0-9, score) — index-aligned with the v7 cache so the main
env can WBF-ensemble them. His config's class order is identical to ours, so labels need no remapping.

    ~/miniconda3/envs/mmdet/bin/python scripts/convnext_infer.py
"""
import json
import pickle
from pathlib import Path

from mmdet.apis import inference_detector, init_detector

REPO = Path(__file__).resolve().parent.parent
CONFIG = str(REPO / "trainer-convnext" / "configs" / "cascade_convnext_eccv.py")
CKPT = str(REPO / "weights" / "convnext_dima.pth")
THR = 0.05

print(f"[convnext] init_detector({Path(CONFIG).name}, {Path(CKPT).name}) on cpu ...")
model = init_detector(CONFIG, CKPT, device="cpu")
paths = json.load(open(REPO / ".data" / "heldout_paths.json"))
print(f"[convnext] {len(paths)} images")
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
with open(REPO / ".data" / "convnext_preds.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"[convnext] saved {len(out)} images -> .data/convnext_preds.pkl "
      f"({sum(len(x) for x in out)} total boxes)")
