# Model Metrics — ECCV 2026 AI City Challenge, Track 6 (Cross-City Object Detection)

**Snapshot date:** 2026-06-01. **Task:** 10-class fine-grained traffic detection (8 vehicle subtypes + Person + Bicycle); train on source city, evaluate on a hidden target city.

This file records, for every experiment run on the Hafnia platform (ours + Dima's), the validation
metrics in detail. Regenerate the raw numbers with `python scripts/collect_metrics.py`.

## Evaluation setup (so the numbers are comparable)

- **Validation split:** leave-camera-out (`--val-camera-frac 0.2 --split-seed 42`): **6 whole cameras held out** — `5 POINTS WB`, `GRANDVIEW - DELHI INT`, `HWY 20 - OLD HWY WBA`, `LOCUST CONNECTOR NB`, `NW ARTERIAL - CHAVENELLE INT`, `US 61 - TWIN VALLEY INT`. Train ≈ 9873 imgs / val ≈ 3066 imgs. None of the val cameras appear in train → an honest cross-city proxy. **All RF-DETR runs + the YOLO run use this identical split.**
- **Metric:** COCO **mAP@[.50:.95]** on the held-out val set. RF-DETR uses torchmetrics; YOLO uses Ultralytics; ConvNeXt uses mmdetection — all COCO-style, so broadly comparable (minor pipeline differences possible).
- **EMA vs regular:** RF-DETR logs **two** numbers per epoch — the EMA model (`val/ema_mAP_50_95`) and the live/regular model (`val/mAP_50_95`). **EMA is what we would deploy.** Both are reported below; the headline column is EMA. (Earlier comparison tables in chat mixed the two — this doc is consistent.)
- **All "best" values are the max over completed epochs.** Per-class AP is the best observed per class (regular model).

## Headline results (best over training, sorted by EMA mAP@50:95)

| Run | Arch / config | Owner | **EMA mAP** | reg mAP | mAP@50 | mAP@75 | mAR | F1 | State | Cost |
|-----|---------------|-------|------------|---------|--------|--------|-----|----|-------|------|
| **v7** | RF-DETR DINOv2, 896 + multi-scale, baseline aug | us | **0.354** | 0.357 | 0.485 | 0.382 | 0.649 | 0.493 | CANCELED | ~872 cr / 10.6 h |
| **v11** | RF-DETR DINOv2, 896 + ms + fisheye/night aug | us | **0.344** | 0.332 | 0.463 | 0.361 | 0.634 | 0.476 | CANCELED (ep5) | ~1050 cr / 12.8 h |
| v5 | RF-DETR DINOv2, 704, baseline (warm-start base) | us | ~0.32† | — | — | — | — | — | prior session | — |
| Dima ConvNeXt | ConvNeXt (mmdetection) | Dima | n/a‡ | 0.312 | 0.465 | 0.352 | — | — | TRAINING (ep4/15) | in progress |
| v6 | RF-DETR DINOv2, 704, DG aug (strong photometric) | us | 0.306 | 0.304 | 0.422 | 0.326 | 0.613 | 0.438 | SUCCEEDED | ~1193 cr / 14.5 h |
| v8 | YOLO26-L, 1280 (from COCO, not warm-started) | us | n/a‡ | 0.254 | ~0.36 | — | — | — | CANCELED (ep17) | ~1050 cr / 12.8 h |
| v9 | RF-DETR **DINOv3**-S, 704, baseline | us | 0.239 | 0.232 | 0.343 | 0.248 | 0.501 | 0.361 | CANCELED | ~943 cr / 11.5 h |
| **v12** | RF-DETR DINOv2, 896 + ms + **CD-FKD** | us | *pending* | *pending* | — | — | — | — | TRAINING | in progress |

† v5: from the prior session; checkpoint saved at `weights/v5_best_ema.pth`; used as the warm-start base for v6/v7/v11/v12. Exact platform metrics not re-verified this snapshot.
‡ "n/a" EMA: YOLO/mmdet log a single mAP (EMA/regular split not exposed the same way); the value shown is their reported COCO mAP.

## Per-class AP (best observed per class, regular model)

| Class | v7 | v11 | v6 | v9 (DINOv3) |
|-------|-----|-----|-----|------|
| Vehicle.Car | **0.700** | 0.687 | 0.665 | 0.588 |
| Vehicle.Pickup Truck | 0.589 | 0.578 | 0.551 | 0.483 |
| Vehicle.Combo Truck | 0.414 | 0.407 | 0.358 | 0.349 |
| Vehicle.Motorcycle | 0.401 | 0.344 | 0.371 | 0.177 |
| Vehicle.Van | 0.381 | 0.381 | 0.216 | 0.186 |
| Vehicle.Trailer | 0.330 | 0.306 | 0.265 | 0.176 |
| Vehicle.Single Truck | 0.283 | 0.307 | 0.249 | 0.190 |
| Vehicle.Heavy Duty Vehicle | 0.264 | 0.205 | 0.256 | 0.239 |
| Vehicle.Bicycle | 0.201 | 0.160 | 0.165 | 0.023 |
| **Person** | **0.105** | 0.122 | 0.074 | 0.027 |

**The recurring weakness is `Person` (~0.10) and `Bicycle` (~0.18) — the smallest objects** — plus the rarer truck subtypes. Car/Pickup are strong everywhere.

## Per-run notes

- **v7 — current best (EMA 0.354 / reg 0.357).** DINOv2 windowed backbone at 896 px with multi-scale, warm-started from v5. Resolution is the lever that worked: it lifted exactly the weak cross-city cameras vs the 704 baseline. Run was CANCELED early (already on plateau ~ep2-3).
- **v11 — fisheye/night augmentation (EMA 0.344).** Same config as v7 + a corrected (frame-filling, no-black-corner) fisheye `OpticalDistortion` + night photometric preset. **Did not beat v7** — the augmentation is at best neutral here (the cameras are wide-angle, not true fisheye; night is photometric, which our v5→v6 comparison already showed doesn't move cross-camera mAP). Stopped at ep5/12.
- **v6 — DG photometric aug (EMA 0.306).** Strong brightness/contrast/colour/noise at 704. ≈ v5 baseline → confirmed **photometric domain-randomization does not help** the cross-city gap.
- **v9 — DINOv3 backbone (EMA 0.239).** DINOv3-S swapped into RF-DETR, warm-started from v5 with the DINOv2 backbone tensors stripped (head reused). **Underperforms** because the head was trained on DINOv2 features and the DINOv3 backbone was never co-trained for detection (no public DINOv3-RF-DETR pretrain) — a mismatched transplant, not a fair backbone comparison. Killed.
- **v8 — YOLO26-L (mAP@50:95 0.254).** Trained from COCO pretrain (not from our data) at 1280 px. Peaked ~ep2 (pre-augmentation), then oscillated ~0.21–0.25; never approached RF-DETR. Killed.
- **v12 — CD-FKD (in progress).** Same config as v7 + single-source-domain-generalization self-distillation (clean teacher / downscaled+corrupted student + global backbone feature-mimic). Targets the small-object / cross-camera failure directly, using only our own data. Awaiting first eval.
- **Dima — ConvNeXt / mmdetection (coco mAP 0.312, ep4/15).** Same leave-camera-out split. Currently below our RF-DETR v7/v11. Two earlier attempts failed on an `img_path` dataset-key error (~160 cr).

## Conclusions (as of this snapshot)

1. **Best model: RF-DETR DINOv2 @ 896 + multi-scale (v7), EMA ≈ 0.35.** Resolution/scale is the only lever that has clearly helped.
2. **Augmentation (DG photometric, fisheye, night) is neutral-to-negative** for cross-city here — consistent with the diagnosis that the failure is *scale / small-object*, not appearance.
3. **DINOv3 is not worth it in this setup** — no co-trained DINOv3-RF-DETR pretrain exists, so the transplant underperforms; YOLO and ConvNeXt also trail RF-DETR.
4. **The mAP is bottlenecked by small/rare classes (Person ~0.10, Bicycle ~0.18).** This is why **v12 (CD-FKD)** is the current bet — it attacks small-object/cross-camera robustness specifically.

## Provenance / caveats

- Numbers pulled from the platform logs via `scripts/collect_metrics.py` (paginated). Costs (`credits_consumed`) and durations are platform-reported.
- v7 and v8 are CANCELED but **present** on the platform (artifacts may be downloadable in terminal state); v7's `/tmp` working checkpoint from the prior session may not be retained — re-export from the platform if needed.
- v5's experiment record was not separately re-verified this snapshot; its checkpoint lives in `weights/v5_best_ema.pth`.
- EMA ≥ regular is the usual case (v11: 0.344 vs 0.332), but not guaranteed (v7: regular 0.357 marginally > EMA 0.354).
