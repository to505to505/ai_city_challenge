# Model Metrics — ECCV 2026 AI City Challenge, Track 6 (Cross-City Object Detection)

**Snapshot date:** 2026-06-02. **Task:** 10-class fine-grained traffic detection (8 vehicle subtypes + Person + Bicycle); train on source city, evaluate on a hidden target city.

This file records, for every experiment run on the Hafnia platform (ours + Dima's), the validation
metrics in detail. Regenerate the raw numbers with `python scripts/collect_metrics.py`.

> **For the synthesis — what we learned and *why* — read [`findings.md`](findings.md).** This file is the raw data; that one is the story.

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
| Dima ConvNeXt | ConvNeXt (mmdetection) | Dima | n/a‡ | 0.312 | 0.465 | 0.352 | — | — | **best.pth pulled — ensemble member** | done |
| v14 | RF-DETR DINOv2, 896 + **letterbox** | us | ~0.309† | — | — | — | — | — | CANCELED / FAILED | — |
| v6 | RF-DETR DINOv2, 704, DG aug (strong photometric) | us | 0.306 | 0.304 | 0.422 | 0.326 | 0.613 | 0.438 | SUCCEEDED | ~1193 cr / 14.5 h |
| v12/v13 | RF-DETR DINOv2, 896 + ms + **CD-FKD** (α=1 / α=100) | us | ~0.30† | — | — | — | — | — | **CANCELED — failed (no-op objective)** | — |
| v8 | YOLO26-L, 1280 (from COCO, not warm-started) | us | n/a‡ | 0.254 | ~0.36 | — | — | — | CANCELED (ep17) | ~1050 cr / 12.8 h |
| v9 | RF-DETR **DINOv3**-S, 704, baseline | us | 0.239 | 0.232 | 0.343 | 0.248 | 0.501 | 0.361 | CANCELED | ~943 cr / 11.5 h |

† Approximate / not re-verified this snapshot. v5: from the prior session (checkpoint `weights/v5_best_ema.pth`; warm-start base for v6/v7/v11/v12). v12/v13/v14: failed/cancelled runs whose exact best mAP isn't cleanly recoverable from the logs — value is from the prior root-cause analysis; the *verdict* (failed, below v7) is what matters, not the third decimal.
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
- **v12 / v13 — CD-FKD (~0.30, FAILED).** Same config as v7 + single-source-DG self-distillation (clean teacher / downscaled+corrupted student + global backbone feature-mimic). **Did not help, and we know why** (full analysis in `findings.md` §2): the DINOv2 backbone is *already* invariant to the corruption — cosine(clean, corrupted) features = **0.9953**, det-loss(corrupted) only **1.06×** clean — so the mimic target is ≈0 (no learning signal). v12 (α=1) was a no-op; v13 (α=100) just diverted ~26% of the backbone gradient away from detection → divergence. A no-op objective, **not** a code bug.
- **v14 — letterbox (~0.309, below v7).** Aspect-preserving resize. Didn't help — aspect handling wasn't the bottleneck, scale is. (One attempt FAILED, retry CANCELED.)
- **Dima — ConvNeXt / mmdetection (coco mAP 0.312).** Cascade R-CNN + ConvNeXt-Tiny, same leave-camera-out split. Below our RF-DETR as a *single* model — **but the most valuable ensemble member** (different architecture → uncorrelated errors → +0.013 mAP, the biggest ensemble win; see `findings.md` §3). `best.pth` pulled and run through `scripts/convnext_infer.py`. (Two earlier attempts failed on an `img_path` dataset-key error, ~160 cr.)

## Ensemble & TTA (offline, local 36-image held-out set)

Measured offline in the main env over the **same 36 held-out images** (leave-camera-out) via `scripts/ensemble_eval.py` — so absolute values differ from the platform val numbers above (v7 here = 0.380 vs EMA 0.354 on platform val); **read the deltas, not the absolutes.** Fusion = model-weighted Weighted Box Fusion. TTA = v7 @896 + horizontal-flip + @1024 fused by WBF. ConvNeXt = Dima's Cascade-R-CNN/ConvNeXt via `scripts/convnext_infer.py` (Docker linux/amd64, mmdet 3.3 prebuilt CPU wheels). YOLO26 = our v8 (YOLO26-L @1280) via `scripts/yolo_infer.py`. All three share our canonical class order (v8 verified by its non-zero solo mAP — a permutation would collapse it to ~0).

| strategy | mAP@.50:.95 | mAP@.50 | mAR100 | small rec | med rec | large rec | Δ vs v7 |
|----------|:-----------:|:-------:|:------:|:---------:|:-------:|:---------:|:-------:|
| v7 (single, @896)              | 0.380 | 0.542 | 0.540 | 28.6% | 57.8% | 92.6% | base |
| v7 + full TTA                  | 0.398 | 0.579 | 0.580 | 26.2% | 60.6% | 95.1% | +0.019 |
| v7 + ConvNeXt (w0.5)           | 0.389 | 0.578 | 0.568 | **40.5%** | 63.9% | 95.7% | +0.009 |
| v7 + TTA + ConvNeXt (w0.5)     | 0.411 | 0.590 | 0.607 | 31.0% | 63.5% | 95.7% | +0.031 |
| YOLO26 (v8) solo               | 0.206 | 0.288 | 0.268 | 14.3% | 42.2% | 79.0% | −0.174 |
| v7 + TTA + YOLO26 (w0.5)       | 0.408 | 0.587 | 0.589 | 26.2% | 59.4% | 95.1% | +0.028 |
| v7 + TTA + ConvNeXt(.7) + YOLO26(.3) | 0.417 | 0.595 | 0.618 | 28.6% | 62.7% | 95.7% | +0.038 |
| DINOv3-RF-DETR (v9) solo       | 0.317 | 0.447 | 0.475 | 19.0% | 47.4% | 90.7% | −0.063 |
| v7 + TTA + DINOv3 (w0.5)       | 0.402 | 0.575 | 0.588 | 26.2% | 60.2% | 94.4% | +0.022 |
| **v7 + TTA + ConvNeXt(.7) + YOLO26(.3) + DINOv3(.5)** | **0.419** | 0.590 | **0.623** | 26.2% | 61.0% | 95.7% | **+0.040** |

ConvNeXt-weight sweep for the TTA combo (broad plateau, **not** a knife-edge): w0.3 → 0.406, **w0.5 → 0.411**, w0.7 → 0.409, w1.0 → 0.406. Adding YOLO26 (3 models): cn.5/yl.5 → 0.416, cn.5/yl.3 → 0.416, **cn.7/yl.3 → 0.417**. Adding DINOv3 on top (4 models): dn.3 → 0.418, dn.5 → 0.419 — a +0.002 hair, within noise.

- **Best to date: 0.419** (4-model WBF), but the *honest* best is effectively the **3-model 0.417** — DINOv3 adds only +0.002, a statistical tie. Diminishing-returns ladder: TTA **+0.019**, ConvNeXt **+0.013**, YOLO26 **+0.006**, DINOv3 **+0.002**.
- **ConvNeXt is the big additive win** (unlike v7+v6, two RF-DETRs, which *hurt*): a different architecture (Cascade R-CNN + FPN) that lifts the cross-city weakness — **small-object recall 28.6%→40.5%**, medium 57.8%→63.9%.
- **YOLO26 is a marginal third** (+0.006): sparse (537 boxes) and weak on small objects, so it refines medium/large + overall recall (mAR100 0.607→0.618) — a *different* axis than ConvNeXt.
- **DINOv3 demonstrates diversity > strength.** It is the **strongest standalone** of the three add-ons (solo 0.317 vs YOLO 0.206) yet the **weakest ensemble partner** (as the 2nd model: 0.402 vs YOLO's 0.408 vs ConvNeXt's 0.411). Reason: v9 shares v7's exact RF-DETR detection head, so its boxes are **correlated** with v7's and WBF extracts little novel signal. Its different DINOv3 backbone decorrelates it just enough to not *hurt* (cf. v6, same backbone → hurt), but not enough to help. **Verdict: not worth the 4th-model inference cost.**
- **Strongest submission candidate: the 3-model WBF (ConvNeXt 0.7, YOLO26 0.3) = 0.417.** Caveat: weights tuned on 36 images, but every sweep point is a plateau, so the choice is safe.

## Conclusions (as of this snapshot)

1. **Best single model: RF-DETR DINOv2 @ 896 + multi-scale (v7), EMA ≈ 0.35.** Resolution/scale is the only training lever that has clearly helped.
2. **Best overall: v7 + TTA + ConvNeXt + YOLO26 WBF ensemble (≈ 0.42 on the 36-img held-out, +0.038 over v7).** ConvNeXt is the big additive win (small-object recall); YOLO26 is a marginal third (+0.006, different axis). Architecture diversity in WBF is the one ensemble pattern that helped instead of hurting (cf. v7+v6, two RF-DETRs, which *hurt*).
3. **Augmentation (DG photometric, fisheye, night) is neutral-to-negative** for cross-city here — consistent with the diagnosis that the failure is *scale / small-object*, not appearance.
4. **DINOv3 is not worth it — as a single model OR in the ensemble.** As a single model the transplant underperforms (no co-trained DINOv3-RF-DETR pretrain). In the ensemble it adds only +0.002 *despite* being the strongest of the three add-ons standalone (solo 0.317) — because it shares v7's RF-DETR head, its errors are correlated and WBF gains little. The clean lesson: **ensemble value = architectural diversity, not member strength** (weaker-but-diverse YOLO/ConvNeXt each add more than stronger-but-correlated DINOv3).
5. **The mAP is bottlenecked by small/rare classes (Person ~0.10, Bicycle ~0.18).** Resolution and the ConvNeXt ensemble are the two things that have moved this; pure-appearance techniques (incl. CD-FKD v12) have not.

## Provenance / caveats

- Numbers pulled from the platform logs via `scripts/collect_metrics.py` (paginated). Costs (`credits_consumed`) and durations are platform-reported.
- v7 and v8 are CANCELED but **present** on the platform (artifacts may be downloadable in terminal state); v7's `/tmp` working checkpoint from the prior session may not be retained — re-export from the platform if needed.
- v5's experiment record was not separately re-verified this snapshot; its checkpoint lives in `weights/v5_best_ema.pth`.
- EMA ≥ regular is the usual case (v11: 0.344 vs 0.332), but not guaranteed (v7: regular 0.357 marginally > EMA 0.354).
