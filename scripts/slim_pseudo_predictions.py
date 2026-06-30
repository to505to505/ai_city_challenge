"""Slim the recovered ensemble test predictions into a compact pseudo-label file for bundling.

Input : .data/ensemble_v20/predictions_test.json — the full WBF-fused ensemble output over the
        14925-image test split (COCO-results: list of {image_id, file_name, category_id, bbox, score},
        bbox = pixel [x, y, w, h], category_id 0..9 in the canonical CLASS_NAMES order, scores can
        exceed 1.0 because WBF sums weighted member confidences).
Output: weights/pseudo_test_labels.json — keyed by file_name, only detections with score >= --floor
        (default 0.25, a generous margin so the in-container student can re-threshold per run without
        re-bundling). Compact enough to ship inside trainer.zip.

    python scripts/slim_pseudo_predictions.py [--floor 0.25]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / ".data" / "ensemble_v20" / "predictions_test.json"
DST = REPO / "weights" / "pseudo_test_labels.json"
CLASS = ["Car", "Pickup", "SingleTrk", "ComboTrk", "HeavyDuty", "Trailer", "Moto", "Bicycle", "Van", "Person"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.25, help="keep detections with score >= floor")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    args = ap.parse_args()

    preds = json.load(open(args.src))
    print(f"[slim] loaded {len(preds)} detections from {args.src}")

    by_img: dict[str, list] = defaultdict(list)
    cc: Counter = Counter()
    kept = 0
    for p in preds:
        s = float(p["score"])
        if s < args.floor:
            continue
        c = int(p["category_id"])
        x, y, w, h = (round(float(v), 2) for v in p["bbox"])
        by_img[p["file_name"]].append([x, y, w, h, c, round(s, 4)])  # xywh, cat, score
        cc[c] += 1
        kept += 1

    out = {"floor": args.floor, "format": "xywh_cat_score", "classes": CLASS, "images": by_img}
    json.dump(out, open(args.dst, "w"))
    sz = Path(args.dst).stat().st_size / 1e6
    print(f"[slim] kept {kept} dets (>= {args.floor}) over {len(by_img)} images -> {args.dst} ({sz:.1f} MB)")
    print("[slim] per-class kept:")
    for c in range(10):
        print(f"        {CLASS[c]:10s} cat{c}: {cc.get(c, 0):>8d}")


if __name__ == "__main__":
    main()
