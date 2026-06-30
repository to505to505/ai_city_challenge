"""Merge ensemble pseudo-labels for the TEST split into a Roboflow-COCO train tree (self-training).

The student RF-DETR consumes a single Roboflow-COCO root (``train/`` + ``valid/`` + ``test/`` each with
``_annotations.coco.json``). ``dataset.to_coco_format`` already exports every test image into ``test/``
with an EMPTY annotations list (the competition test split carries no GT). This module attaches the
recovered ensemble pseudo-labels to those test images and folds them into ``train/`` so they become
additional TRAIN samples — clean labeled-train stays as the anchor, pseudo-labeled target-domain test
is added for domain adaptation.

Pseudo source: ``weights/pseudo_test_labels.json`` (built by ``slim_pseudo_predictions.py`` from the
WBF-fused ensemble output): ``{"floor": f, "format": "xywh_cat_score", "classes": [...],
"images": {file_name: [[x, y, w, h, cat, score], ...]}}`` — bbox pixel xywh, cat 0..9 canonical order.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def load_pseudo_labels(path: "str | Path", threshold: float) -> Dict[str, List[List[float]]]:
    """Load the slimmed pseudo-label file and keep only boxes with ``score >= threshold``.

    Args:
        path: Path to the slimmed pseudo-label JSON.
        threshold: Minimum fused confidence kept (the slim file's ``floor`` is the hard lower bound).

    Returns:
        Mapping ``file_name -> [[x, y, w, h, cat, score], ...]`` (only images with >=1 surviving box).
    """
    blob = json.load(open(path))
    floor = float(blob.get("floor", 0.0))
    if threshold < floor:
        raise ValueError(
            f"--pseudo-threshold {threshold} is below the slim file's floor {floor}; "
            f"re-run slim_pseudo_predictions.py with a lower --floor to keep those boxes."
        )
    out: Dict[str, List[List[float]]] = {}
    for fn, boxes in blob["images"].items():
        kept = [b for b in boxes if float(b[5]) >= threshold]
        if kept:
            out[fn] = kept
    return out


def _clip_box(x: float, y: float, w: float, h: float, iw: int, ih: int) -> "Tuple[float, float, float, float] | None":
    """Clip an ``xywh`` box to the image and drop it if degenerate (< 2 px a side)."""
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(iw), x + w), min(float(ih), y + h)
    cw, ch = x1 - x0, y1 - y0
    if cw < 2.0 or ch < 2.0:
        return None
    return (round(x0, 2), round(y0, 2), round(cw, 2), round(ch, 2))


def merge_pseudo_test_into_coco(
    coco_dir: "str | Path",
    pseudo_path: "str | Path",
    threshold: float,
) -> Dict[str, int]:
    """Fold pseudo-labeled test images into ``train/_annotations.coco.json`` (in place).

    Reuses the images already exported under ``test/`` (copies each matched image into ``train/`` and
    appends its image entry + pseudo annotations to ``train``). Category ids are 0-indexed and must
    match the export's ``categories`` exactly (the canonical CLASS_NAMES order), so they are validated.

    Args:
        coco_dir: Roboflow-COCO root containing ``train/`` and ``test/`` (post ``to_coco_format``).
        pseudo_path: Slimmed pseudo-label JSON.
        threshold: Minimum fused confidence kept as a pseudo-box.

    Returns:
        Stats dict: ``images_added``, ``boxes_added``, ``unmatched`` (pseudo file_names with no test
        image), ``test_images``, ``boxes_dropped`` (degenerate after clipping).
    """
    coco_dir = Path(coco_dir)
    train_dir, test_dir = coco_dir / "train", coco_dir / "test"
    train_json_p, test_json_p = train_dir / "_annotations.coco.json", test_dir / "_annotations.coco.json"
    train = json.load(open(train_json_p))
    test = json.load(open(test_json_p))

    # Category sanity: pseudo cats 0..9 must line up with the export's categories.
    train_cat_ids = {c["id"] for c in train["categories"]}
    if not train_cat_ids.issubset(set(range(64))) or 0 not in train_cat_ids:
        raise ValueError(f"unexpected train categories {sorted(train_cat_ids)} — pseudo cats are 0-indexed")

    pseudo = load_pseudo_labels(pseudo_path, threshold)
    test_by_name = {im["file_name"]: im for im in test["images"]}

    next_img = max((im["id"] for im in train["images"]), default=-1) + 1
    next_ann = max((a["id"] for a in train["annotations"]), default=-1) + 1
    images_added = boxes_added = unmatched = boxes_dropped = 0

    for fn, boxes in pseudo.items():
        tim = test_by_name.get(fn)
        if tim is None:
            unmatched += 1
            continue
        iw, ih = int(tim["width"]), int(tim["height"])
        clipped = []
        for x, y, w, h, c, _s in boxes:
            cb = _clip_box(float(x), float(y), float(w), float(h), iw, ih)
            if cb is None:
                boxes_dropped += 1
                continue
            clipped.append((cb, int(c)))
        if not clipped:
            continue
        # Reference the image already exported under test/ via a relative symlink instead of copying
        # — the full test split is ~14.9k images (~tens of GB); duplicating it into train/ would blow
        # the container disk. PIL/dataloader follow symlinks transparently. Fall back to a copy only if
        # the filesystem refuses symlinks.
        src_img = test_dir / tim["file_name"]
        dst_img = train_dir / tim["file_name"]
        if src_img.exists() and not dst_img.exists():
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            rel = os.path.relpath(src_img, dst_img.parent)
            try:
                os.symlink(rel, dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)
        img_id = next_img
        next_img += 1
        train["images"].append({"id": img_id, "file_name": tim["file_name"], "width": iw, "height": ih})
        images_added += 1
        for (bx, by, bw, bh), c in clipped:
            train["annotations"].append({
                "id": next_ann, "image_id": img_id, "category_id": c,
                "bbox": [bx, by, bw, bh], "area": round(bw * bh, 2), "iscrowd": 0,
            })
            next_ann += 1
            boxes_added += 1

    json.dump(train, open(train_json_p, "w"))
    return {
        "images_added": images_added, "boxes_added": boxes_added, "unmatched": unmatched,
        "test_images": len(test["images"]), "boxes_dropped": boxes_dropped,
    }
