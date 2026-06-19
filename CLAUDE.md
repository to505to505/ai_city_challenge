# Hafnia / ECCV cross-city — operational notes

Project root: `/home/dsa/hafnia`. RF-DETR Large trainer for the **ECCV 2026 AI City Challenge Track #6
(eccv-cross-city)** object-detection competition. This file is for picking up the work and querying
the Hafnia platform API from a fresh session.

## Hafnia API access

The CLI key configured locally (`Config().api_key`, prefix `spSpsAeKlEq…`) only sees 2 stale runs.
**The key that sees all the real experiments** (the one tied to the actual training runs) must be
passed explicitly via env:

```
export HAFNIA_API_KEY='ApiKey DYexiq-novJrWzkJr122RFI1TNqEs0_JO9pA1kfXMcs:ub6k6bKVgDq1tce2v3P8To2moSKPpCHkKNcT4FyLT2BfxhIHN6cKKmEojmorS-rzzs1Ia0lfidQ_18fIhf5ceWpTNgLMnoznBNu8QuAEKQ6MW3iKaGrHmnQ1Z1-AmUlKRy2skqJhh7Tu2tbsR2C6Hq0olgSWKmuVGpAg_I9i_T4'
```

> ⚠️ This key has been pasted in chat repeatedly and SHOULD be rotated/revoked. Until then it is the
> working read key. Account: `d.sakharov@student.maastrichtuniversity.nl`.

- Base URL: `https://api.hafnia.milestonesys.com`
- Auth header: `{"Authorization": <ApiKey ...>}`
- Use `from hafnia import http` → `http.fetch(BASE + path, headers=HDR)` (returns parsed JSON).
- State is **not** settable via PATCH (returns 200 but no-op). Cancel runs from the dashboard.

### Endpoints

- `GET /api/v1/experiments?limit=100` → `{data:[...], next, count}`. Paginate via `next`.
  Each item: `id, name, state, created_at, training_started_at, training_finished_at,
  training_duration_seconds, command, error, dataset, trainer`.
- `GET /api/v1/experiments/{id}` → full object incl. `error` (e.g. `['AlgorithmError: , exit code: 1']`).
- `GET /api/v1/experiments/{id}/logs?order=desc&limit=1000[&before=<created_at>]` → list of
  `{created_at, raw_message}`. Paginate backward with `before=<last created_at>`. Logs are heavily
  line-wrapped during the docker BUILD phase; the real Python traceback is short and near the end.
  ⚠️ For FAILED runs the platform keeps only a ~100-line tail — per-epoch `val/...` metric lines are
  gone, but the `Best EMA mAP improved to X (epoch N)` announcement usually survives (how v20's
  0.4224 was confirmed).
- `GET /api/v1/experiments/{id}/model` → streams `model.tar.gz` = the live-mirrored "Trained model"
  (`/opt/ml/model`). **Works even for TRAINING_FAILED runs** — used to recover v20's epoch-0
  `checkpoint_best_ema.pth` (EMA mAP 0.4224, ~137 MB, now at `artifacts/v20/`; verified epoch=0,
  pos-embed 88×88 ⇒ R1408). No other subresource endpoint exists (`/checkpoints`, `/artifacts`,
  `/outputs`, `/credentials` … all 404); files in the experiment's `checkpoints[]` list (tfevents,
  hparams.yaml) are downloadable only via the dashboard ("Download Experiment Outputs").

### Metric log format (RF-DETR)

Regex the `raw_message`:
- `val/mAP_50_95'  ent_type='metric' value=X` (also `val/ema_mAP_50_95'`, `val/mAP_50'`, `val/AP/<class>'`)
- mmdet runs (VFNet/ConvNeXt): `coco/bbox_mAP: X`
- pycocotools summary: `(AP) @[ IoU=0.50:0.95 ... ] = X`
- Steps: `step=(\d+)`. Note GPU system metrics are OFF (`Skip logging GPU metrics`) — no VRAM in logs.

Ready-made scanners live in `tools/hafnia_leaderboard.py` (ranks best mAP per experiment). Run with
`HAFNIA_API_KEY='ApiKey ...' python tools/hafnia_leaderboard.py`.

## Trainer packaging

- Build: `hafnia trainer create-zip . --output ./trainer.zip` (respects `.hafniaignore`). ~260 MB.
- Root `Dockerfile` = RF-DETR trainer (base `pytorch/pytorch:2.5.1-cuda12.1`, WORKDIR `/opt/recipe`,
  COPYs `src configs scripts rf-detr weights`). The separate `trainer-convnext/` is a DIFFERENT
  trainer, excluded via `.hafniaignore` (was accidentally bloating the zip before).
- Launch is done via the **web dashboard** (upload zip → run). CLI alt:
  `hafnia experiment create --name <n> --dataset eccv-cross-city --trainer-path . --environment Lite --cmd "<train cmd>"`.
- After ANY Dockerfile/requirements change you MUST re-upload the zip (platform rebuilds the image);
  reusing a previously-uploaded trainer package will NOT pick up image-level changes.

### Two pinned fixes in the trainer (do not regress)

1. `requirements.txt`: `transformers==5.9.0` + `torchvision<0.21`. Newer transformers imports
   `torch.float8_e8m0fnu` at module load (a torch-2.7 dtype) → `AttributeError` on the 2.5.1 base
   image, killing `from rfdetr import ...`. This killed v16/v17.
2. `Dockerfile` ENV `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Multi-scale + the
   val→train phase change fragments the CUDA caching allocator; a large contiguous request then
   fails even with GiBs free-but-reserved. This killed v17–v20 (all at the ~66 min epoch-0/val boundary).
3. `requirements.txt`: `albumentations==2.0.8` + `stringzilla==3.12.6`. albumentations→albucore→
   stringzilla; latest stringzilla (4.6.x) ships source-only for this platform → pip compiles → the
   base image has no gcc → `gcc: No such file or directory` → BUILD_FAILED (killed v21). 3.12.6 has
   a manylinux wheel. `Dockerfile` also now installs `build-essential` as insurance against future
   wheel-drift of any other dep.

All four fixes validated end-to-end via `hafnia runc build-local trainer.zip` + `from rfdetr import
RFDETRLarge` smoke (build exit 0, imports clean, env var present). Rebuild + re-upload after edits.

## Hardware / memory reality (measured, not guessed)

- Platform GPU is a single **T4 with 14.74 GiB usable** (NOT 16). Lite = 1×T4, Scale = 4×T4.
- Measured locally on an RTX 3060 with a faithful train step (RFDETRLarge + criterion + AdamW + bf16,
  `group_detr=13`): the **network's** activation memory is modest and **independent of #boxes**
  (3900 fixed queries = 300 × group_detr 13). E.g. bs1: R1248≈2.7 GB, R1536≈4.7 GB reserved.
- #boxes/image affects ONLY the Hungarian matcher cost matrix `[bs·3900 × all_boxes]` (built ~5×:
  main + aux decoder layers + enc), NOT the network and NOT the loss. Scales steeply with bs.
- eccv sample (300 imgs): boxes/image train mean≈12, p95≈25, **max=40**; FULL dataset has denser
  frames. test split has 0 labels.
- **bs1 is the safe choice** at high res: matcher matrix stays small (1 image), v16_fixed ran 18h at
  R1248 bs1 with zero OOM. bs2/bs4 add intermittent multi-GB spikes on dense frames → OOM.
- `RFDETRSmall` vs `RFDETRLarge`: same DINOv2-S backbone, differ by ONE decoder layer (3 vs 4).
  Switching L→S saves ~170 MB (~1%) — does NOT let you raise resolution. Backbone is the memory lever.
- `--multi-scale` IS real per-step dynamic resolution (GPU interpolation each step in
  `module_model.py:on_train_batch_start`), peak scale = base+128 (step 32, offsets −3..+4). It is NOT
  just a +128 bump. `skip_random_resize` (no CLI flag) only gates the CPU-side path.

## ⚠️ The "epoch-0 mAP 0.42/0.475" numbers were FAKE (sanity-check artifact, now fixed)

PTL's pre-fit sanity check runs validation hooks on 2 batches; `COCOEvalCallback` had no
`trainer.sanity_checking` guard, so v20/v22's "0.4224 (epoch 0)" and v23's "0.4754 (epoch 0)" were
mAP **over 2 images** computed BEFORE any training step (telltales: per-class table shows only 2
classes; Precision=1.0/Recall=0.5; appears seconds after the PTL model summary; identical value
across runs). Worse, `BestModelCallback` recorded the fake value as best → honest epochs (~0.39)
never beat it → **real best checkpoints were never published**. Fixed with sanity guards in
`coco_eval.py` (batch+epoch hooks) and `best_model.py:on_validation_end` — do not regress.
Corollary: v20/v22's OOM was on the FIRST training step (right after sanity), not at an epoch
boundary; the ~66 min before it was the COCO export of ~28k images, not training.

## Run history / leaderboard (HONEST best val mAP@[.50:.95], leave-camera-out 0.2 seed 42)

- **REAL BEST: v23 fine-tune @ R1280+ms bs1 → EMA 0.3945 (epoch 1)**; e0 0.3925, e2 0.3860 (declines
  after e1 → short fine-tunes only). Checkpoint LOST (best-tracker was stuck on fake 0.4754, then
  run cancelled; CANCELED runs serve no /model artifact). Reproducible: rerun ~2-3 epochs @1280.
- v5 weights evaluated at R1280, no training (v24 eval-only): 0.358 — the resolution bump alone.
- v16_fixed (R1120 ms bs1): 0.3811 EMA (genuine, from metric lines), ran 18h no OOM.
- prior: hires1024 ≈ 0.376; squash v7 ≈ 0.357. Letterbox (v14/v15): dead end. Person/rare classes
  remain the bottleneck (class-balanced sampling / copy-paste, not input geometry).
- Memory ceiling on the T4 (14.74 GiB): R1280+ms (peak 1408) bs1 trains fine; R1408+ms (peak 1536)
  bs1 OOMs on the first training step. bs>1 OOMs harder (matcher matrix × batch).
- `vfnet_r50_fpn_eccv_v17b`: separate mmdet VFNet trainer, SUCCEEDED (user's parallel experiment).
- v24 (`rfdetr_eval_predict_v24`, SUCCEEDED): eval-only + test inference with v5 @1280 → val 0.358,
  `predictions_test.json` (1.097M dets / 14925 imgs) in Experiment outputs; submission packed at
  `submission_v24.zip`. NOTE: a refreshed fine-tune best (~0.394) would beat this submission.

## Eval-only & test-prediction modes (added 2026-06-12, tested end-to-end)

- `--epochs 0` → `trainer.validate()` instead of fit (no optimizer; logs val/mAP as usual) and
  persists the evaluated weights to `/opt/ml/model/checkpoint_best_ema.pth` so the run completes
  SUCCEEDED with a downloadable model.
- `--predict-test [--predict-threshold 0.05]` → inference over the TEST split, written in the
  **official Hafnia submission format** (matches `milestone-hafnia/trainer-object-detection`):
  builds a HafniaDataset of predicted `Bbox` primitives (`ground_truth=False`, normalized coords)
  via `run_inference_on_dataset` and saves with `dataset.write_annotations(logger._path_artifacts())`
  → `annotations.parquet` + `dataset_info.json` keyed by `remote_path`, in **`/opt/ml/output/data`**.
  ⚠️ The platform scorer reads THAT. The earlier COCO-results `predictions_test.json` in
  `/opt/ml/model` was the WRONG format AND wrong location (scorer errored `unable to find column
  remote_path`). v24/v25 COCO submissions were invalid; re-run with the fixed trainer.
- The submission run loads the best checkpoint (bundled `weights/v25_best_total.pth` = v25's 0.394
  model) and predicts from it; `--epochs 0` makes it pure inference (~1.3 h over 14925 test imgs).

## Current recommended launch (relaunch after re-uploading the fixed zip)

```
python scripts/train.py --epochs 15 --devices 1 --resolution 1408 --batch-size 1 --grad-accum-steps 8 --multi-scale --aug-preset baseline --init-weights weights/v5_best_ema.pth --lr-scheduler cosine --lr 5e-5 --lr-encoder 7.5e-5 --val-camera-frac 0.2 --run-name rfdetr_large_hires1408_ms_bs1_v21
```

- Base 1408 → per-step scales [1312…1536], peak 1536. bs1 effective batch = 1×8 = 8.
- Warm-starts from `weights/v5_best_ema.pth` (shipped in zip).
- If OOM persists (worst-case need ~14.1 GB, ceiling 14.74 — tight): drop to `--resolution 1344`
  (peak 1472, ~1 GB more headroom; validation runs at base res so mAP barely changes).
- ALWAYS give the train command on a SINGLE line (multi-line backslash commands get pasted literally
  and fail with `unrecognized arguments: \`).

## Split modes (`--split-mode`, added 2026-06-15)

- `camera_out` (default): leave-camera-out — `--val-camera-frac` of CAMERAS held out ENTIRELY for
  val (DG-honest proxy for the hidden city; val cameras absent from train; SMALLER train).
- `stratified`: per-camera split — every camera stays in train, `--val-camera-frac` of each camera's
  FRAMES sampled into val. BIGGEST train + all views seen (use for the FINAL submission model). val
  mAP is same-camera (optimistic) — checkpoint-selection only, NOT a cross-city signal. Verified on
  the sample: train 134 vs native 117 vs camera_out 118; val-only cameras = 0.
- `native`: dataset's native split (also shares cameras train↔val, so all views are in train).
- ⚠️ Trained from COCO-pretrain (no `--init-weights`) the model adapts SLOWLY: 1 epoch ≈ 0.075 on the
  sample vs ~0.39 from the v5 warm-start. From-scratch needs ~15-30 epochs (R1280 ms ≈ 4 h/epoch on
  the T4 → 2.5-5 days). The watcher mirrors best checkpoints live, so a long run is cancellable.

## train.py CLI flags

`--epochs --batch-size --grad-accum-steps --lr --lr-encoder --num-workers --devices --resolution
--encoder {dinov2_windowed_small,dinov3_small,dinov3_base,dinov3_large} --aug-preset {baseline,dg,fisheye_night}
--init-weights --lr-scheduler {step,cosine} --multi-scale --expanded-scales --letterbox
--val-camera-frac --split-seed --split-mode {camera_out,stratified,native}
--predict-test [--predict-threshold] --cd-fkd [+ --cd-fkd-alpha/-min-scale/-noise-std] --run-name`.
`--resolution` must be divisible by 32. No `--do-random-resize-via-padding` flag exists.
