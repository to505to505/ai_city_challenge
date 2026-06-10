"""Empirical ensemble-diversity audit over our existing members' prediction caches.

Answers "which models actually complement each other" with DATA, not theory: for every ground-truth
object it records WHICH models detect it (IoU>=0.5, correct class, conf>=THR), then reports

  - per-model recall + per-size-bin recall (who is strong where)
  - the detection-count histogram: how many GT are caught by exactly k models (k=0 => shared blind spot)
  - each model's UNIQUE contribution: GT caught by ONLY that model (its irreplaceable value)
  - pairwise Jaccard of caught-GT sets (high => redundant/correlated, low => diverse)
  - greedy marginal recall: order members by how much NEW recall each adds (the ensemble's real ladder)
  - the shared blind spot broken down by size => what a NEW member would have to target

Runs in the main env:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/diversity_audit.py
"""
from __future__ import annotations

import pickle
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V7 = REPO / ".data" / "tta_ens_cache.pkl"
CN = REPO / ".data" / "convnext_preds.pkl"
YL = REPO / ".data" / "yolo26_preds.pkl"
DN = REPO / ".data" / "dinov3_preds.pkl"
THR = 0.30  # "confident detection" threshold for the caught/missed analysis (mAP-style sweep is separate)


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def size_bin(box):
    s = ((box[2]-box[0]) * (box[3]-box[1])) ** 0.5
    return "small" if s < 32 else ("medium" if s < 96 else "large")


def caught_mask(preds, gts):
    """Greedy match: returns a bool list over gts — True if some conf>=THR same-class det has IoU>=0.5."""
    dets = sorted([p for p in preds if p[2] >= THR], key=lambda x: -x[2])
    used = [False] * len(gts)
    for box, cls, _ in dets:
        bj, bi = -1, 0.5
        for j, g in enumerate(gts):
            if used[j] or g[1] != cls:
                continue
            v = iou(box, g[0])
            if v >= bi:
                bi, bj = v, j
        if bj >= 0:
            used[bj] = True
    return used


def main():
    v7 = pickle.load(open(V7, "rb"))
    caches = {"v7(RF-DETR)": [r["preds"]["v7_896"] for r in v7]}
    for name, path in [("convnext", CN), ("yolo26", YL), ("dinov3", DN),
                       ("vfnet", REPO / ".data" / "vfnet_preds.pkl")]:
        if path.exists():
            caches[name] = pickle.load(open(path, "rb"))
    v16p = REPO / ".data" / "v16_preds.pkl"
    if v16p.exists():
        caches["v16(1120)"] = pickle.load(open(v16p, "rb"))["v16_1120"]
    names = list(caches)
    gts_all = [r["gts"] for r in v7]
    n_gt = sum(len(g) for g in gts_all)
    print(f"[audit] {len(v7)} imgs | {n_gt} GT objects | members: {names} | conf>={THR}\n")

    # detection matrix: per GT object -> set of models that caught it; track size bin
    rows = []  # (size_bin, frozenset(models_that_caught))
    per_model_recall = {m: {"small": [0, 0], "medium": [0, 0], "large": [0, 0], "all": [0, 0]} for m in names}
    masks = {m: [caught_mask(caches[m][i], gts_all[i]) for i in range(len(v7))] for m in names}
    for i, gts in enumerate(gts_all):
        for j, g in enumerate(gts):
            sb = size_bin(g[0])
            who = frozenset(m for m in names if masks[m][i][j])
            rows.append((sb, who))
            for m in names:
                hit = m in who
                per_model_recall[m][sb][1] += 1
                per_model_recall[m]["all"][1] += 1
                if hit:
                    per_model_recall[m][sb][0] += 1
                    per_model_recall[m]["all"][0] += 1

    # 1) per-model recall by size
    print("=" * 78)
    print("1) PER-MODEL RECALL (caught GT / total), by size")
    print("=" * 78)
    print(f"{'model':14s} {'all':>8s} {'small':>8s} {'medium':>8s} {'large':>8s}")
    for m in names:
        r = per_model_recall[m]
        def pct(b): return f"{r[b][0]/max(1,r[b][1])*100:5.1f}%"
        print(f"{m:14s} {pct('all'):>8s} {pct('small'):>8s} {pct('medium'):>8s} {pct('large'):>8s}")

    # 2) coverage histogram: caught by exactly k models
    print("\n" + "=" * 78)
    print("2) HOW MANY MODELS CATCH EACH GT  (k=0 => NOBODY catches it = shared blind spot)")
    print("=" * 78)
    hist = {k: 0 for k in range(len(names) + 1)}
    hist_small = {k: 0 for k in range(len(names) + 1)}
    for sb, who in rows:
        hist[len(who)] += 1
        if sb == "small":
            hist_small[len(who)] += 1
    for k in range(len(names) + 1):
        bar = "#" * round(hist[k] / max(1, n_gt) * 50)
        print(f"  caught by {k} models: {hist[k]:4d} ({hist[k]/n_gt*100:4.1f}%)  {bar}")
    print(f"  -> shared blind spot (k=0): {hist[0]} objs, of which {hist_small[0]} are SMALL "
          f"({hist_small[0]/max(1,hist[0])*100:.0f}%)")

    # 3) unique contribution: GT caught by ONLY this model
    print("\n" + "=" * 78)
    print("3) UNIQUE CONTRIBUTION — GT objects caught by ONLY this model (irreplaceable value)")
    print("=" * 78)
    for m in names:
        uniq = [sb for sb, who in rows if who == {m}]
        from collections import Counter
        c = Counter(uniq)
        print(f"  {m:14s} {len(uniq):3d} unique  (small {c['small']}, medium {c['medium']}, large {c['large']})")

    # 4) pairwise Jaccard of caught-GT sets (correlation proxy)
    print("\n" + "=" * 78)
    print("4) PAIRWISE OVERLAP (Jaccard of caught-GT sets) — HIGH=redundant, LOW=diverse")
    print("=" * 78)
    caught_idx = {m: {i for i, (_, who) in enumerate(rows) if m in who} for m in names}
    for a, b in combinations(names, 2):
        inter = len(caught_idx[a] & caught_idx[b])
        union = len(caught_idx[a] | caught_idx[b])
        print(f"  {a:14s} vs {b:14s}: Jaccard {inter/max(1,union):.3f}")

    # 5) greedy marginal recall ladder
    print("\n" + "=" * 78)
    print("5) GREEDY MARGINAL RECALL — each model's NEW caught-GT over the running union")
    print("=" * 78)
    union = set()
    remaining = set(names)
    order = []
    # seed with the single best-recall model
    seed = max(names, key=lambda m: len(caught_idx[m]))
    while remaining:
        best = max(remaining, key=lambda m: len(caught_idx[m] - union))
        gain = len(caught_idx[best] - union)
        union |= caught_idx[best]
        order.append((best, gain))
        remaining.discard(best)
    base = 0
    for m, gain in order:
        base += gain
        print(f"  + {m:14s} adds {gain:4d} new TP -> union recall {base/n_gt*100:5.1f}% "
              f"({base}/{n_gt})")
    print(f"  ORACLE ceiling (union of all members): {len(union)/n_gt*100:.1f}% recall")


if __name__ == "__main__":
    main()
