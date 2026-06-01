"""Training-free backbone comparison: DINOv2-S (what RF-DETR Large uses) vs DINOv3-S (bundled),
both FROZEN, on the local 300-sample GT boxes.

We can't compare DETECTION yet (the DINOv3 RF-DETR head is untrained), so instead we probe the
*backbone features* directly: crop each GT box, extract a frozen feature from each backbone, and
run a parameter-free kNN classifier trained on SEEN-camera boxes, evaluated on HELD-OUT-camera
boxes (10-class + coarse). Higher held-out accuracy = features that transfer better cross-camera —
the property that should help detection. It's a PROXY (real answer needs training), but a fair one.

    python scripts/probe_backbones.py
"""
from __future__ import annotations
import sys, warnings
from collections import Counter
from pathlib import Path
warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rf-detr" / "src"))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModel, DINOv3ViTModel  # noqa: E402
from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402

SHORT = ["Car", "Pickup", "SingleTrk", "ComboTrk", "HeavyDuty", "Trailer", "Moto", "Bicycle", "Van", "Person"]
COARSE = {0: "veh", 1: "veh", 2: "veh", 3: "veh", 4: "veh", 5: "veh", 6: "two", 7: "two", 8: "veh", 9: "person"}
HELDOUT = {"5 POINTS WB", "GRANDVIEW - DELHI INT", "HWY 20 - OLD HWY WBA",
           "LOCUST CONNECTOR NB", "NW ARTERIAL - CHAVENELLE INT", "US 61 - TWIN VALLEY INT"}
DINOV2_NAME = "facebook/dinov2-with-registers-small"   # the backbone RF-DETR Large uses (ViT-S, 384)
DINOV3_DIR = str(REPO_ROOT / "weights" / "dinov3-vits16-pretrain-lvd1689m")
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CROP = 224
MIN_BOX_PX = 8  # skip degenerate tiny boxes


def gt_boxes(row):
    W, H = row["width"], row["height"]
    out = []
    for b in (row["bboxes"] or []):
        if b.get("task_name") != "object_detection":
            continue
        x1, y1 = b["top_left_x"] * W, b["top_left_y"] * H
        x2, y2 = (b["top_left_x"] + b["width"]) * W, (b["top_left_y"] + b["height"]) * H
        if x2 - x1 >= MIN_BOX_PX and y2 - y1 >= MIN_BOX_PX:
            out.append((int(x1), int(y1), int(x2), int(y2), int(b["class_idx"])))
    return out


@torch.no_grad()
def extract(model, crops):
    """crops: (N,3,224,224) normalized. Returns (N, D) mean-pooled last-hidden features."""
    feats = []
    for i in range(0, len(crops), 32):
        out = model(pixel_values=crops[i:i + 32])
        h = out.last_hidden_state  # (b, tokens, D)
        feats.append(h.mean(dim=1))  # mean over all tokens (global descriptor)
    return torch.cat(feats).float()


def knn_eval(train_f, train_y, test_f, test_y, k=5):
    train_f = F.normalize(train_f, dim=1); test_f = F.normalize(test_f, dim=1)
    sim = test_f @ train_f.T  # cosine
    idx = sim.topk(k, dim=1).indices
    preds = []
    for row in idx:
        preds.append(Counter(train_y[j].item() for j in row).most_common(1)[0][0])
    preds = np.array(preds); test_y = test_y.numpy()
    acc = (preds == test_y).mean()
    coarse_ok = np.array([COARSE[p] == COARSE[t] for p, t in zip(preds, test_y)]).mean()
    return acc, coarse_ok


def main():
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    gt = df.filter(pl.col("split").is_in(["train", "validation"]))
    print(f"[data] {len(gt)} GT images — cropping boxes ...")

    crops, labels, groups = [], [], []
    for i in range(len(gt)):
        row = gt.row(i, named=True)
        held = row["cam"] in HELDOUT
        img = Image.open(row["file_path"]).convert("RGB")
        for (x1, y1, x2, y2, c) in gt_boxes(row):
            crop = img.crop((x1, y1, x2, y2)).resize((CROP, CROP))
            t = torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255.0
            crops.append((t.unsqueeze(0) - IMN_MEAN) / IMN_STD)
            labels.append(c); groups.append("HELDOUT" if held else "SEEN")
    crops = torch.cat(crops); labels = torch.tensor(labels)
    groups = np.array(groups)
    seen_m, held_m = groups == "SEEN", groups == "HELDOUT"
    print(f"[data] {len(crops)} boxes | SEEN={seen_m.sum()} HELDOUT={held_m.sum()}")

    results = {}
    for tag, loader in [("DINOv2-S", lambda: AutoModel.from_pretrained(DINOV2_NAME)),
                        ("DINOv3-S", lambda: DINOv3ViTModel.from_pretrained(DINOV3_DIR))]:
        print(f"[extract] {tag} ...")
        model = loader().eval()
        feats = extract(model, crops)
        acc, coarse = knn_eval(feats[seen_m], labels[seen_m], feats[held_m], labels[held_m])
        # also seen->seen (in-domain, via simple split) for reference
        results[tag] = (acc, coarse)
        del model

    print("\n" + "=" * 58)
    print("FROZEN BACKBONE kNN PROBE — train SEEN cams, test HELD-OUT cams")
    print("=" * 58)
    print(f"  {'backbone':10s}  {'10-class acc':>12s}  {'coarse acc':>11s}")
    for tag, (acc, coarse) in results.items():
        print(f"  {tag:10s}  {acc*100:11.1f}%  {coarse*100:10.1f}%")
    d = results["DINOv3-S"][0] - results["DINOv2-S"][0]
    print(f"\n  -> DINOv3-S vs DINOv2-S (10-class held-out): {d*100:+.1f} pts")
    print("  (proxy for backbone transfer quality; real detection needs training the head.)")


if __name__ == "__main__":
    main()
