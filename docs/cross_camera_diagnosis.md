# Cross-City Detection — Diagnosis & Findings

Working notes on **why the RF-DETR Large baseline generalizes poorly to unseen cameras**,
and what actually moves the needle. All numbers are on the local 300-image sample
(`eccv-cross-city` v1.0.0), evaluated single-process (trustworthy COCO eval).

## TL;DR

- **The cross-camera gap is real and large:** in-domain (seen cameras) COCO mAP@50:95 ≈ **0.76**,
  on the 6 held-out (leave-camera-out) cameras ≈ **0.37**.
- **The gap is a SCALE / resolution problem, not appearance.** On unseen cameras the model misses
  **~62 % of small** and **~30 % of medium** objects; large/near objects are detected fine (~1 % miss).
- **Photometric domain-generalization augmentation does NOT help** (brightness/contrast/colour/noise).
  A 12-epoch DG fine-tune (v6) was identical to the baseline (v5) — same mAP, same failure profile.
  We were treating the wrong disease: the shift between cameras is **viewpoint/scale** (fisheye,
  far-perspective, night), not colour.
- **Tiled (SAHI) inference confirms the cause:** slicing doubles small-object recall (33 % → 67 %)
  and lifts medium (51 % → 71 %) — the information *is* in the frame, the model just can't see it
  when 1920×1080 is downscaled to 704. (Naïve tiling costs precision, so headline mAP dips; see below.)
- **→ The lever is resolution/scale:** higher-res + multi-scale training, and/or a properly merged
  (WBF) test-time tiling pass. Not more colour augmentation.

## Setup

- Model: RF-DETR Large, 704×704, 10 classes, COCO-pretrained (`rf-detr-large-2026.pth`).
- **Validation = leave-camera-out** (`scripts/train.py --val-camera-frac 0.2`): 6 of 28 labeled
  cameras held out *entirely* (no frame leakage), as an honest proxy for the hidden target city.
  The native train/val split shares cameras (every native-val camera is also in train), so it only
  measures *same-camera* generalization and overstates results.
- 6 held-out cameras: `5 POINTS WB`, `GRANDVIEW - DELHI INT`, `HWY 20 - OLD HWY WBA`,
  `LOCUST CONNECTOR NB`, `NW ARTERIAL - CHAVENELLE INT`, `US 61 - TWIN VALLEY INT`.
- Local diagnostic split: 110 seen-camera images vs 36 held-out-camera images (146 GT-bearing total;
  the `test` split has GT withheld).

## Experiments

| run | env | epochs | what | cross-camera mAP@50:95 |
|---|---|---|---|---|
| v3 | Lite | 2 | first working pipeline | — (in-domain ~0.66) |
| v5 | Scale 4×T4 | ~8 (canceled) | leave-camera-out baseline | **~0.375** |
| v6 | Lite | 12 | warm-start v5 + **DG photometric augs** + cosine LR | **~0.373** (no change) |

## Finding 1 — the cross-camera gap is SCALE-driven

`scripts/diagnose_checkpoint.py` on the current checkpoint:

| | SEEN (110 img) | HELD-OUT (36 img) |
|---|---|---|
| COCO mAP@50:95 | 0.760 | **0.375** |
| mAP@50 | 0.930 | 0.501 |
| recall / precision | 0.95 / 0.90 | 0.62 / 0.62 |
| **small miss** | 24 % | **62 %** |
| **medium miss** | 0.8 % | **30 %** |
| large miss | 0 % | 0.6 % |
| Person miss | 20 % | **87 %** |

Large/near objects are detected everywhere; the collapse is concentrated on **small and medium**
objects, and on **Person** (small + viewpoint-sensitive). Subtype confusion (Pickup↔Car, Van→Car)
roughly doubles cross-camera, and false positives increase.

Per held-out camera, the gap is **non-uniform** — half the cameras are fine, a few are catastrophic:

```
NW ARTERIAL - CHAVENELLE   recall 0.89   (normal mounted view)
LOCUST CONNECTOR NB        recall 0.79
US 61 - TWIN VALLEY        recall 0.62
HWY 20 - OLD HWY WBA       recall 0.47   (far-perspective highway)
GRANDVIEW - DELHI INT      recall 0.38   (FISHEYE / wide-angle)
5 POINTS WB                recall 0.35
```

## Finding 2 — photometric DG augmentation does nothing

v6 (`--aug-preset dg`: stronger RandomBrightnessContrast / ColorJitter / GaussNoise / GaussianBlur,
warm-started from v5, 12 epochs cosine) vs v5, on held-out cameras:

| | v5 baseline | v6 DG |
|---|---|---|
| mAP@50:95 | 0.375 | 0.373 |
| small miss | 62 % | 64 % |
| medium miss | 30 % | 33 % |
| Person miss | 87 % | 89 % |

Identical within noise (v6 marginally worse). The held-out cameras differ from training in
**geometry** (camera angle, lens distortion, object-scale distribution), which photometric augs
cannot address. Confirmed visually: `visualization/heldout/` — the failures are fisheye
(`GRANDVIEW`), far-perspective highway (`HWY 20`), and night/low-light cameras, where the **small
distant objects are missed** while close vehicles are detected.

## Finding 3 — tiled (SAHI) inference proves it's resolution

`scripts/test_sahi.py` (full frame + 3×2 overlapping tiles, class-aware NMS), held-out cameras:

| | baseline (full) | SAHI (full+tiles) |
|---|---|---|
| **recall small** | 33 % | **67 %** |
| **recall medium** | 51 % | **71 %** |
| recall large | 95 % | 98 % |
| mAP@50:95 | 0.375 | 0.329 |

Slicing **doubles** small-object recall — the objects are recoverable given more effective pixels,
so the bottleneck is resolution (1920×1080 → 704 shrinks a 40 px car to ~15 px). **But** naïve
tile-merging adds false positives (edge-cut / duplicate boxes), so precision and headline mAP drop.
A real test-time win needs WBF merging + edge-box filtering + score calibration; the cleaner fix is
to push resolution into training.

**Update — tuned SAHI tested** (`scripts/test_sahi_wbf.py`: WBF + tile-edge filtering + tile_thr 0.25):
recovered most of the precision (mAP@50:95 0.329 → **0.360**, mAP@50 0.479 → 0.497) and kept the
recall gain (small 33 % → 64 %, medium 51 % → 70 %), but **still did not beat baseline mAP (0.375)**,
and mAR@100 dropped (0.61 → 0.57). Reason: tiled boxes are recovered but **loosely localized** (good
at IoU 0.5, poor at 0.75–0.95), so recall@0.5 rises while IoU-averaged mAP does not. WBF ≈ NMS here
(the edge filter did the work). **Verdict: SAHI is not a clean test-time win — pursue higher-res
training (B) so the model emits well-localized small-object boxes natively.**

## Pitfalls discovered (don't get fooled again)

- **Lightning sanity-check pollutes "best" metrics.** `num_sanity_val_steps` defaults to 2 → a
  2-batch validation runs *before* training and can log a wildly high value (we saw a bogus
  `Best EMA mAP 0.649 (epoch 0)`), which then locks `checkpoint_best_ema.pth` to the epoch-0 init.
  Always trust the first *full* epoch eval, and use `checkpoint_best_regular` for the actually-trained
  weights when this happens.
- **Multi-GPU (DDP) on RF-DETR detection requires `find_unused_parameters=True`** (fixed in
  `rf-detr/.../training/trainer.py`), else it crashes at epoch 0.
- **Artifacts download only from a terminal experiment** (`/api/v1/experiments/<id>/model`); a running
  job returns 404. Cancel/finish first (cancel still harvests `/opt/ml/model`).
- Per-run **COCO export (~80 min) is billed at the GPU rate** — expensive on Scale (4× idle GPUs).
  Prefer Lite for iteration unless wall-clock-bound (Scale 546 cr/h vs Lite 83 cr/h).

## Conclusion & next steps

The cross-city failure is **small-object detection under viewpoint/scale shift**, not appearance.
Ordered by leverage:

1. **(A, test-time, no training)** Tuned SAHI/WBF: tile + Weighted-Box-Fusion merge, filter
   tile-edge boxes, raise tile threshold — aim for net-positive mAP, not just recall.
2. **(B, training)** **Higher input resolution + `multi_scale=True`** (e.g. 704 → 896/1008) and
   **scale-jitter augmentation** (RandomResizedCrop / wider Affine scale) so the model natively
   detects small objects — should lift both recall and mAP without the merge-FP penalty.
3. Person and subtype confusion should improve alongside the small-object fixes; revisit class
   balancing only if they lag.

## Reproduction

```bash
python scripts/download_weights.py                       # base weights
# (bundle a trained checkpoint as weights/v5_best_ema.pth to diagnose it)
python scripts/diagnose_checkpoint.py weights/v5_best_ema.pth   # seen vs held-out breakdown
python scripts/visualize_heldout.py  weights/v5_best_ema.pth    # GT vs pred, missed highlighted
python scripts/test_sahi.py          weights/v5_best_ema.pth    # tiled vs full inference
```
Evidence images: `visualization/heldout/` (cross-camera) and `visualization/failures/` (in-domain).
