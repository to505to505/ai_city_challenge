# Findings & Lessons — ECCV 2026 AI City Challenge, Track 6 (Cross-City Detection)

*Last updated: 2026-06-02.* Companion to [`model_metrics.md`](model_metrics.md) (the raw per-run data).
This file is the **synthesis**: what we tried, what worked, what didn't, and — most importantly — **why**.
Read this first; drop into `model_metrics.md` for the numbers.

**Task.** 10-class fine-grained traffic detection (8 vehicle subtypes + Person + Bicycle). Train on a
source city, score on a hidden target city. Metric: **COCO mAP@[.50:.95]**. We proxy the cross-city
gap with a **leave-camera-out** split (6 whole cameras held out, none seen in training).

---

## TL;DR

1. **Best single model:** `v7` — RF-DETR (DINOv2 windowed backbone) @ **896 px + multi-scale**, EMA ≈ **0.354**.
2. **Best submission candidate:** a **3-model Weighted-Box-Fusion ensemble** — `v7 + TTA + ConvNeXt + YOLO26` ≈ **0.417** on the local held-out set (**+0.038** over v7 alone).
3. **The cross-city gap is SCALE / small-object, not appearance.** This single diagnosis explains every result below.
4. **Every appearance-side technique failed** — DINOv3 backbone, CD-FKD self-distillation, photometric / fisheye / night augmentation — and we now understand *why each one* failed, not just *that* it did.
5. **Ensembling law we confirmed empirically:** *value = architectural **diversity**, not member **strength***. A weaker-but-different model adds more than a stronger-but-similar one.
6. **The ensemble is effectively maxed.** The only untested training lever with a plausible payoff is **higher resolution (1024)**.

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

### Why the ensemble is effectively complete
We have exactly **three distinct detector families**, and all three are now in the ensemble:
- **RF-DETR** (DETR transformer) → `v7`
- **Cascade R-CNN + ConvNeXt** (two-stage CNN) → ConvNeXt
- **YOLO26** (anchor-free CNN) → `v8`

Every *other* checkpoint we own (`v5/v6/v9/v11/v12/v13/v14`) is an **RF-DETR variant** — i.e. a clone of
v7's head. By the diversity law they will all behave like DINOv3: correlated, contributing ≈0 or
hurting. **There is no fourth diverse architecture left to add**, so the 3-model ensemble is the
ceiling of this approach.

### Caveat (honesty)
Ensemble weights are tuned on **36 images** — small. But every weight sweep is a **broad plateau**
(ConvNeXt 0.3–1.0 all give 0.406–0.411; YOLO configs all 0.416–0.417), not a knife-edge, so the
choices are robust rather than overfit to one lucky value. The hidden target city may still shift the
optimum; low weights on the add-ons limit the downside.

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
