"""Ensemble v7 (RF-DETR) + Dima's ConvNeXt via MODEL-WEIGHTED WBF, evaluated offline from caches.

Runs in the main env. Needs:
  - .data/tta_ens_cache.pkl  (v7 @896 / @1024 / flip predictions + GT, from tta_ensemble.py)
  - .data/convnext_preds.pkl (ConvNeXt predictions, produced by the Docker mmdet run)
both index-aligned over the same 36 held-out images.

Standard WBF penalizes a box seen by few models (conf = Σ wᵢ·confᵢ / Σ w_models), so a weaker model's
*unique* boxes are down-weighted automatically; giving ConvNeXt a model-weight < 1 down-weights it
further. That's how a weaker-but-diverse model can ADD agreed true-positives without dragging v7 down
(the failure mode of the v6 ensemble). We sweep the ConvNeXt weight.

    python scripts/ensemble_eval.py
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision

REPO = Path(__file__).resolve().parent.parent
V7_CACHE = REPO / ".data" / "tta_ens_cache.pkl"
CN_CACHE = REPO / ".data" / "convnext_preds.pkl"


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def size_bin(box):
    s = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
    return "small" if s < 32 else ("medium" if s < 96 else "large")


def wbf_weighted(model_dets, weights, iou_thr=0.55):
    """Model-weighted Weighted Box Fusion. model_dets: list per model of [(box,cls,score)]; weights aligned."""
    total_w = sum(weights)
    by_cls = defaultdict(list)  # cls -> [(box, score, model_weight)]
    for dets, w in zip(model_dets, weights):
        for box, cls, sc in dets:
            by_cls[cls].append((box, sc, w))
    out = []
    for cls, items in by_cls.items():
        clusters = []  # each: {sw_box (Σ w·s·box), sw (Σ w·s), conf_sum (Σ w·s), fused}
        for box, sc, w in sorted(items, key=lambda x: -x[1] * x[2]):
            best, bi = -1, iou_thr
            for k, cl in enumerate(clusters):
                v = iou(box, cl["fused"])
                if v > bi:
                    bi, best = v, k
            if best >= 0:
                cl = clusters[best]
                cl["sw"] += w * sc
                cl["sw_box"] = tuple(cl["sw_box"][j] + w * sc * box[j] for j in range(4))
                cl["fused"] = tuple(cl["sw_box"][j] / cl["sw"] for j in range(4))
            else:
                clusters.append({"sw": w * sc, "sw_box": tuple(w * sc * box[j] for j in range(4)), "fused": box})
        for cl in clusters:
            out.append((cl["fused"], cls, cl["sw"] / total_w))  # WBF conf: penalizes few-model boxes
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


def main():
    if not CN_CACHE.exists():
        print(f"[wait] {CN_CACHE} not present yet — the Docker ConvNeXt run hasn't finished.")
        return
    v7 = pickle.load(open(V7_CACHE, "rb"))
    cn = pickle.load(open(CN_CACHE, "rb"))
    assert len(v7) == len(cn), f"misaligned: v7 {len(v7)} vs convnext {len(cn)}"
    print(f"[eval] {len(v7)} imgs | v7 boxes {sum(len(r['preds']['v7_896']) for r in v7)} | "
          f"convnext boxes {sum(len(c) for c in cn)}")

    def v7p(i, k):
        return v7[i]["preds"].get(k, [])

    STRATS = {
        "v7_baseline":      lambda i: v7p(i, "v7_896"),
        "v7_full_tta":      lambda i: wbf_weighted([v7p(i, "v7_896"), v7p(i, "v7_896_flip"), v7p(i, "v7_1024")], [1, 1, 1]),
        "v7+cn_w1.0":       lambda i: wbf_weighted([v7p(i, "v7_896"), cn[i]], [1.0, 1.0]),
        "v7+cn_w0.5":       lambda i: wbf_weighted([v7p(i, "v7_896"), cn[i]], [1.0, 0.5]),
        "v7+cn_w0.3":       lambda i: wbf_weighted([v7p(i, "v7_896"), cn[i]], [1.0, 0.3]),
        "v7tta+cn_w0.3":    lambda i: wbf_weighted([v7p(i, "v7_896"), v7p(i, "v7_896_flip"), v7p(i, "v7_1024"), cn[i]], [1, 1, 1, 0.3]),
        "v7tta+cn_w0.5":    lambda i: wbf_weighted([v7p(i, "v7_896"), v7p(i, "v7_896_flip"), v7p(i, "v7_1024"), cn[i]], [1, 1, 1, 0.5]),
        "v7tta+cn_w0.7":    lambda i: wbf_weighted([v7p(i, "v7_896"), v7p(i, "v7_896_flip"), v7p(i, "v7_1024"), cn[i]], [1, 1, 1, 0.7]),
        "v7tta+cn_w1.0":    lambda i: wbf_weighted([v7p(i, "v7_896"), v7p(i, "v7_896_flip"), v7p(i, "v7_1024"), cn[i]], [1, 1, 1, 1.0]),
    }
    names = list(STRATS)
    mp = {k: MeanAveragePrecision(box_format="xyxy") for k in names}
    rs = {k: {s: [0, 0] for s in ("small", "medium", "large")} for k in names}
    for i, rec in enumerate(v7):
        tgt = [{"boxes": torch.tensor([list(g[0]) for g in rec["gts"]]).reshape(-1, 4),
                "labels": torch.tensor([g[1] for g in rec["gts"]], dtype=torch.long)}]
        for k in names:
            preds = STRATS[k](i)
            mp[k].update([to_tm(preds)], tgt)
            for s2, (m2, t2) in recall_by_size(preds, rec["gts"]).items():
                rs[k][s2][0] += m2
                rs[k][s2][1] += t2
    res = {k: mp[k].compute() for k in names}
    base = float(res["v7_baseline"]["map"])
    print("\n" + "=" * 80)
    print(f"{'strategy':16s} {'mAP50:95':>9s} {'mAP50':>8s} {'mAR100':>8s}   {'small':>6s} {'med':>6s} {'large':>6s}")
    print("=" * 80)
    for k in names:
        r = res[k]
        rec3 = "  ".join(f"{rs[k][s][0] / max(1, rs[k][s][1]) * 100:5.1f}%" for s in ("small", "medium", "large"))
        d = float(r["map"]) - base
        flag = "  <-- base" if k == "v7_baseline" else f"  ({'+' if d >= 0 else ''}{d:.3f})"
        print(f"{k:16s} {float(r['map']):9.3f} {float(r['map_50']):8.3f} {float(r['mar_100']):8.3f}   {rec3}{flag}")
    print("\n[done] best mAP:", max(names, key=lambda k: float(res[k]["map"])))


if __name__ == "__main__":
    main()
