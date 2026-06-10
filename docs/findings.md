# Findings & Lessons — ECCV 2026 AI City Challenge, Track 6 (Cross-City Detection)

*Last updated: 2026-06-02.* Companion to [`model_metrics.md`](model_metrics.md) (the raw per-run data).
This file is the **synthesis**: what we tried, what worked, what didn't, and — most importantly — **why**.
Read this first; drop into `model_metrics.md` for the numbers.

**Task.** 10-class fine-grained traffic detection (8 vehicle subtypes + Person + Bicycle). Train on a
source city, score on a hidden target city. Metric: **COCO mAP@[.50:.95]**. We proxy the cross-city
gap with a **leave-camera-out** split (6 whole cameras held out, none seen in training).

---

## TL;DR

1. **Best single model:** `v16` — RF-DETR @ **1120 px + multi-scale** (Dima), EMA **0.389**. The resolution ladder 896→1024→1120 = 0.354→0.376→0.389; 1120 is the practical ceiling (diminishing returns + VRAM).
2. **Best submission candidate:** **`v16(native+flip) + ConvNeXt(0.7) + YOLO26(0.3) + VFNet(0.3–0.5)`** WBF ≈ **0.436** local (**+0.056** over single v7; v7 itself is subsumed and dropped). See §3 round 2.
3. **The cross-city gap is SCALE / small-object, not appearance.** This single diagnosis explains every result below.
4. **Every appearance-side technique failed** — DINOv3 backbone, CD-FKD self-distillation, photometric / fisheye / night augmentation — and we now understand *why each one* failed, not just *that* it did.
5. **Ensembling laws confirmed empirically:** value = **diversity along the failure axis** (resolution counts; same-res same-head doesn't), **decorrelated recall beats solo mAP** (VFNet 0.235-solo adds more than DINOv3 0.317-solo), and **every member needs ≥2 views** in the WBF pool.
6. **Remaining levers:** final all-cameras retrain @1120 for submission; the Person blind spot is a class-imbalance training problem, out of ensemble reach.

---

## 1. The central diagnosis: it's scale, not appearance

Everything we observed points to the same root cause: the model fails on the target city primarily
because objects there appear at **scales/geometries it underweights** (small, far, dense), **not**
because the target city *looks* different (lighting, colour, weather, lens).

**Evidence chain:**

- **Resolution is the only training lever that moved the metric.** 704 → 896 px lifted EMA from ~0.32 (v5) to **0.354** (v7), and it lifted *exactly* the weakest cross-city cameras. Nothing else did.
- **The per-class floor is the smallest objects.** `Person` (~0.10) and `Bicycle` (~0.18) are the worst classes across *every* run — the two physically smallest categories. Car/Pickup (large) are strong everywhere (~0.6–0.7).
- **The ensemble win came from small-object recall.** Adding ConvNeXt lifted small-object recall **28.6% → 40.5%** — the biggest single movement we produced — and that's where its mAP gain came from.
- **Every photometric/appearance intervention was neutral-to-negative** (see §2). If the gap were appearance, domain-randomization would have helped. It didn't.

**Implication:** keep spending on resolution / multi-scale / small-object recall (incl. diverse-model
ensembling). Stop spending on appearance robustness — it's solving a problem we don't have.

---

## 2. Training-side: what we tried, the verdict, and WHY

| Lever | Run | Verdict | **Why** |
|-------|-----|:-------:|---------|
| **Resolution 704→896 + multi-scale** | v7 | ✅ **WORKED** | The gap is scale. Bigger input + scale jitter = the only thing that lifted the weak cameras. **This is our base model.** |
| Baseline @704 | v5 | baseline | Warm-start base for everything after. |
| DG photometric aug (strong brightness/contrast/colour/noise) | v6 | ❌ neutral | ≈ v5. The gap isn't appearance, so domain-randomizing appearance does nothing. |
| Fisheye + night aug | v11 | ❌ neutral (0.344 < v7) | Cameras are wide-angle, not true fisheye; night is photometric → same null result as v6. (Also: our first fisheye was *wrong* — black corners; real cameras fill the frame. Fixed to frame-filling `OpticalDistortion`, still didn't help.) |
| Letterbox (aspect-preserving resize) | v14 | ❌ ~0.309 < v7 | Aspect handling wasn't the bottleneck; scale is. |
| **DINOv3** backbone swap | v9 | ❌ 0.239 | **Mismatched transplant.** The LWDETR head was trained on DINOv2 features; no public *co-trained* DINOv3-RF-DETR pretrain exists, so swapping the backbone and reusing the head gives the head features it was never trained for. Not a fair "DINOv3 vs DINOv2" test. |
| **CD-FKD** self-distillation (clean teacher / corrupted student / feature-mimic) | v12, v13 | ❌ failed | **The backbone is already robust to the corruption we distilled against.** Measured cosine(clean, corrupted) features = **0.9953**; det-loss(corrupted) was only **1.06×** det-loss(clean). So the feature-mimic target is ~0 → no learning signal. Calibrating α=100 just diverted ~26% of the backbone gradient *away from detection* → divergence. **Not a code bug** — a no-op objective. (My earlier "train/eval mismatch" hypothesis was *refuted* by the 1.06× measurement.) |
| YOLO26-L from COCO @1280 | v8 | ❌ 0.254 (single) | Trained from COCO (not warm-started from our data); peaked ~ep2, never approached RF-DETR. *But it earns a place in the ensemble — see §3.* |
| ConvNeXt / Cascade R-CNN (mmdet) | Dima | ⚠️ 0.312 (single) | Below RF-DETR as a single model, *but the most valuable ensemble member — see §3.* |

**One-line takeaway:** the only training idea that helped was **make the input bigger**. Every "clever"
appearance/architecture trick failed for a *specific, understood* reason — all reducible to "the gap is scale."

---

## 3. Inference-side: TTA + ensembling (this is where the remaining wins are)

Measured offline over the **same 36-image local held-out set** with `scripts/ensemble_eval.py`
(model-weighted Weighted Box Fusion). Absolute mAP differs from the platform val numbers (v7 here =
0.380); **read the deltas**.

### The diminishing-returns ladder
```
v7 single                                   0.380
  + TTA (multi-scale 896+1024 + h-flip)   +0.019  ->  0.398
  + ConvNeXt   (Cascade R-CNN + FPN)      +0.013  ->  0.411   diverse arch, fixes SMALL objects
  + YOLO26     (anchor-free CNN)          +0.006  ->  0.417   diverse arch, refines MED/LARGE
  + DINOv3     (RF-DETR variant)          +0.002  ->  0.419   correlated -> negligible (skip)
```
**Submission candidate: the 3-model `v7 + TTA + ConvNeXt(0.7) + YOLO26(0.3)` = 0.417.**

### The law we confirmed: diversity > strength
The cleanest result of the whole investigation. As the **second** model added to `v7 + TTA`:

| add-on | solo mAP (strength) | ensemble mAP (value) |
|--------|:---:|:---:|
| ConvNeXt | 0.31 | **0.411** |
| YOLO26 | 0.21 | 0.408 |
| DINOv3 | **0.32** ← strongest | **0.402** ← weakest |

DINOv3 is the **strongest add-on standalone yet the weakest ensemble partner**, because it shares
v7's exact RF-DETR detection head → its errors are **correlated** with v7 → WBF extracts little novel
signal. A different *backbone* (DINOv3 vs DINOv2) decorrelates it just enough to not actively *hurt*
(contrast `v6`, which shared v7's backbone *and* head → it **hurt** the ensemble), but not enough to
help. **Weaker-but-different beats stronger-but-similar.**

### Round 2 (2026-06-05): v16 anchor + VFNet — new best 0.436, and two refined laws

Two additions re-opened the ceiling we thought we'd hit:

```
old best  v7tta + ConvNeXt(.7) + YOLO26(.3)                  0.417
NEW BEST  v16(native+flip) + ConvNeXt(.7) + YOLO26(.3) + VFNet(.3-.5)   0.436  (+0.056 over v7 single)
          (plateau 0.433–0.436 across compositions; v7 fully subsumed — dropped)
```

- **v16 = Dima's RF-DETR @1120** (EMA 0.389, best single). Swapping it in as anchor: +0.014.
  **Law refined:** *diversity must lie along the failure axis.* Same-family RF-DETRs at **different
  resolutions** (896 vs 1120) decorrelate where same-res (v6) hurt and same-head (DINOv3) added ~0 —
  because resolution IS the scale axis, and scale is the cross-city failure.
- **Law 2 — pair every member's views.** v16 with ONE view *dropped* the pool 0.417→0.396 (its unique
  detections carry no agreement weight under WBF's conf normalization and dilute everyone else);
  native+flip together → 0.431. Any new member must enter with ≥2 views (TTA).
- **VFNet R-50 (trained for this purpose, 12 ep) is the mirror-image of DINOv3 and validates the
  program:** worst solo mAP of the pool (0.235 local — loose boxes) but **best recall of all six
  models** (overall 73.1%, **small 45.2%**, medium 65.5%) and **19 unique TP — the most irreplaceable
  member** (ConvNeXt 4, v16 4, v7 1, YOLO 0, DINOv3 2). The dense anchor-free head over-fires on small
  objects; WBF supplies the precision it lacks. **Ensemble value = decorrelated *recall*, not solo mAP.**
- **Diversity audit after round 2:** shared blind spot shrank 24.5% → **19.0%** of GT; oracle union
  recall 75.5% → ~80%. Greedy ladder: VFNet (331) → v16 (+24) → ConvNeXt (+7) → others ≈ 0.
  The remaining 86 blind objects are still dominated by **Person** — unreachable by ensembling.

### Caveat (honesty)
Ensemble weights are tuned on **36 images** — small. But every weight sweep is a **broad plateau**
(round 1: ConvNeXt 0.3–1.0 all 0.406–0.411; round 2: all compositions 0.433–0.436), not a knife-edge,
so the choices are robust rather than overfit to one lucky value. The hidden target city may still
shift the optimum; low weights on the add-ons limit the downside.

---

## 4. Competition rules (verified) — what is and isn't allowed

- **Metric:** COCO-style **mAP** (IoU .50:.95, averaged per-class and per-city). *Not* F1.
- **External datasets for training: BANNED.** (This is why CD-FKD / self-distillation on our own data was attractive — it adds no external data.)
- **Pretrained models: ALLOWED.** (COCO/ImageNet/SSL backbones are fine — that's how every model here starts.)
- **Ensembles: ALLOWED.** (Hence the entire §3 line of attack is legal for the final submission.)

---

## 5. Methodology & infrastructure lessons (so nobody re-learns these the hard way)

- **Hafnia Training-aaS is network-isolated.** Weights must be **bundled into `trainer.zip`** (`COPY weights ./weights`) — the container can't download at runtime. Populate `weights/` locally first (`scripts/download_weights.py`).
- **One experiment at a time, per user.** Cancelling via the API doesn't take — you must click **Stop** in the web UI.
- **The experiments list API is paginated.** Always follow the `next` cursor. (We mis-reported "v7/v8 were deleted" once because page 1 didn't contain them — they were just CANCELED and off-page.)
- **Download a trained checkpoint** from `GET /api/v1/experiments/<id>/model` (a gzipped TAR; RF-DETR bundles `checkpoint_best_ema.pth` + `_regular.pth`, YOLO bundles `best.pt`). Works once the run is in a terminal state. See `scripts/dinov3_infer.py` / the inline fetch we used for v8/v9.
- **mmcv does not build on macOS-arm64** (no prebuilt wheels; source compile fails: `Error compiling objects for extension`). **Solution:** run mmdet inference in a **`linux/amd64` Docker container** using **prebuilt CPU wheels** (`mmcv==2.1.0 -f .../cpu/torch2.1.0/index.html`, `mmdet==3.3.0`) — no compile. **Gotcha:** the slim `python:3.10` image lacks `libGL.so.1`; `apt-get install -y libgl1 libglib2.0-0` (or use `opencv-python-headless`). See `scripts/_docker_convnext.sh`.
- **Verify class order before ensembling** — a silent label permutation corrupts fusion and *lowers* mAP invisibly. Cheap check: a model's **standalone mAP**. If the order is right it scores ~its known value; if permuted, ~0 (boxes match spatially but the class never matches). We verified v8 (0.206) and v9 (0.317) this way — both non-zero → orders correct, no remap needed.
- **Model-weighted WBF** (`conf = Σ wᵢ·confᵢ / Σ w_models`) is what lets a weaker model add *agreed* true-positives without dragging the leader down: a box seen by few models is auto-down-weighted, and a per-model weight < 1 down-weights a weaker model further. This is *why* the diverse ensemble helps where a naive merge (or two correlated RF-DETRs) hurts.
- **EMA vs regular metrics:** RF-DETR logs both `val/ema_mAP_50_95` (what we'd deploy) and `val/mAP_50_95`. EMA is the headline, but not always higher (v7: regular 0.357 > EMA 0.354). Keep them straight.
- **Local-held-out harness pattern:** cache every model's raw predictions over a fixed image list **once**, then sweep fusion strategies **instantly** offline (`scripts/tta_ensemble.py` builds the v7 cache; `ensemble_eval.py` consumes it + the ConvNeXt/YOLO/DINOv3 caches). Iterate on fusion without re-running any model.
- **macOS OpenMP clash:** torch + torchmetrics each link a `libomp` → `OMP Error #15`. Run eval with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` (single-threaded is fine for 36 images and removes the parallel-correctness concern).
- **Secrets:** the Hafnia API key lives only in `~/.hafnia` (keychain). Never commit it; we secret-scan files before every commit.

---

## 6. What's left / open levers

- **Resolution 1024 training** — the *one* untested lever with a real prior. Resolution is the only thing that has ever moved this metric, and we've only gone to 896. **This is the recommended next training run** if we spend more credits.
- **The ensemble is maxed** — no fourth diverse architecture remains (§3). Don't spend more on ensembling unless a genuinely new detector family appears.
- **Per-class bottleneck remains `Person` / `Bicycle`** (the smallest objects). Anything that specifically lifts tiny-object recall (higher res, SAHI-style tiling, a small-object-specialist member) is the highest-leverage direction. (SAHI tiling was mAP-neutral here but boosted small recall +24pp — worth revisiting at the 1024 scale.)

---

## Reproduce

```bash
# 1. Pull the base + trained checkpoints into weights/ (gitignored)
python scripts/download_weights.py                      # RF-DETR pretrain
#    trained v7/v8/v9 come from GET /api/v1/experiments/<id>/model (see §5)

# 2. Build the v7 prediction cache over the fixed held-out image list
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=rf-detr/src python scripts/tta_ensemble.py

# 3. Per-model predictions (each writes .data/<model>_preds.pkl, index-aligned)
bash scripts/_docker_convnext.sh                         # ConvNeXt (Docker linux/amd64)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/yolo_infer.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=rf-detr/src python scripts/dinov3_infer.py

# 4. Sweep fusion strategies offline (instant) — prints the table in §3
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/ensemble_eval.py
```
