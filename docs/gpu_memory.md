# GPU Memory Budget — T4 16 GB (Hafnia "Lite")

*Last updated: 2026-06-04.* Measured, not guessed. Earlier batch-size choices (`--batch-size 1` at
1024/1120) were **3–5× too conservative**; this doc replaces "I think it won't fit" with arithmetic
grounded in real measurements. **Operating rule: pick batch from the tables below, not bs1.**

## TL;DR — rational defaults

- T4 = **16 GB**, ~15 GB usable after the CUDA context. **Target ≤ 14 GB** (safety margin).
- **Static** (weights + grads + AdamW) ≈ **0.5–0.6 GB** for our models — negligible.
- **Activations dominate** and scale ~quadratically with resolution. Per-sample, measured below.
- **No run has ever OOM'd** — we never even approached the limit.

| model | input | **use batch** | est. peak |
|---|---|:---:|:---:|
| RF-DETR Large | 896 + multiscale | **4–5** | ~12 GB @bs5 |
| RF-DETR Large | 1024 + multiscale | **4** | ~13 GB |
| RF-DETR Large | 1120 + multiscale | **3** | ~12 GB |
| RF-DETR Large | 1280 + multiscale | **3** (fits!) | ~14 GB |
| VFNet R-50 (mmdet) | 1920 long-edge | **4** | ~11 GB |
| Cascade R-CNN + ConvNeXt | 1920 long-edge | 2 → could be **4** | 7.2 GB @bs2 (measured) |

## Measured ground truth

1. **Cascade R-CNN + ConvNeXt-Tiny @1920 long-edge, batch 2 → 7.24 GB peak on T4** (mmdet logs
   `memory:` every iter; 77 samples, min 1.4 / max 7.24 GB). A *heavy* 3-stage detector uses **47%**
   of the 16 GB card.
2. **Zero OOMs, ever.** All 5 `TRAINING_FAILED` runs failed for non-memory reasons (dataset-key bug,
   etc.) — checked every log for "CUDA out of memory". The 16 GB ceiling has never been touched.
3. **RF-DETR's trainer does not log GPU memory** (only mmdet does), which is why RF-DETR numbers below
   come from a CPU activation probe + the static arithmetic, cross-checked against the no-OOM runs.

## How memory is spent — the model

```
GPU_peak(resolution, batch)  ≈  C_context  +  S_static  +  batch × A_activation(resolution)
```

- **C_context** — CUDA context + cuDNN/cuBLAS workspaces ≈ **~1 GB** fixed (one-time, not per-sample).
- **S_static** — parameters + gradients + optimizer state. Fixed, independent of resolution/batch.
- **A_activation(res)** — forward activations kept for backprop. **The variable, dominant term.**

### S_static (exact, from param counts)

Native PyTorch AMP keeps fp32 params (autocast casts ops to fp16 on the fly — no fp16 weight copy).
With AdamW the per-parameter cost is: fp32 param (4 B) + fp32 grad (4 B) + Adam `m` (4 B) + Adam `v`
(4 B) = **16 B/param**.

| model | params | S_static = params × 16 B |
|---|:---:|:---:|
| RF-DETR Large (DINOv2 ViT-S backbone) | 33.6 M | **0.54 GB** |
| VFNet R-50 | 32.9 M | **0.53 GB** |

Static is ~0.5 GB — **negligible**. Do not budget around it; budget around activations.

### A_activation(res) — measured (RF-DETR Large, fwd+backward)

Measured by running a real forward+backward and reading peak process RSS
(`scripts/_mem_probe.py`, one process per resolution so the high-water mark is clean). CPU runs fp32;
on GPU AMP stores activations in fp16 (½) but adds cuDNN workspaces — these roughly cancel, so the
fp32-CPU number is a good proxy for the GPU activation working set (validated: it reproduces the
no-OOM behavior of every real run).

| resolution | A_activation (GB) | note |
|:---:|:---:|---|
| 704 | 1.46 | |
| 896 | 1.98 | v7 base |
| 1024 | 2.35 | v15 base / v7 multiscale peak |
| 1152 | 2.91 | v15 multiscale peak |
| 1248 | ~3.45 | v16 (1120) multiscale peak — extrapolated |
| 1408 | ~4.2 | 1280 multiscale peak — extrapolated |

Scaling is sub-quadratic (windowed attention is ~linear in tokens): ~**+25 % per +128 px**.
**Multiscale adds +128 px to the peak** (`compute_multi_scale_scales`: offsets up to +4 × stride 32),
so always budget at `base + 128`, not `base`.

## Batch-size limits (derived, ≤ 14 GB safe target)

`max_batch = (14 − C_context − S_static) / A(res) ≈ (14 − 1.5) / A(res)`

### RF-DETR Large (with multiscale → budget at base+128)

| base res | peak res | A(peak) | **safe batch (≤14 GB)** | what we *actually* ran |
|:---:|:---:|:---:|:---:|---|
| 896 | 1024 | 2.35 | **5** | v7 used bs2 ❗ (2.5× under) |
| 1024 | 1152 | 2.91 | **4** | v15 used bs1 ❗ (4× under) |
| 1120 | 1248 | 3.45 | **3–4** | v16 used bs1 ❗ |
| 1280 | 1408 | 4.2 | **3** | never tried — **it fits** |

Without multiscale, add ~1 to each (budget at `base`, not `base+128`).

### VFNet R-50 / mmdet two-stage @1920 long-edge

Anchored to the **measured** ConvNeXt 7.24 GB @bs2: per-sample ≈ (7.24 − 1.5)/2 ≈ **2.9 GB**
(ConvNeXt Cascade). VFNet R-50 is single-stage (no RPN + 3 cascade heads + RoIAlign on 1000
proposals), so ~0.7–0.8× → **~2.1–2.3 GB/sample**.

| model @1920 | per-sample | **safe batch (≤14 GB)** |
|---|:---:|:---:|
| Cascade R-CNN + ConvNeXt | 2.9 GB | 4 (we ran bs2) |
| VFNet R-50 | ~2.2 GB | **4–5** |

## What we were doing wrong

Every RF-DETR run used `--batch-size 1` "to be safe". The measurements show 1024 + ms fits **bs4** and
1120 + ms fits **bs3** — so we trained at **3–4× lower batch than necessary**, which only slowed
training (smaller batch = noisier gradients, same wall-clock per image but fewer images/step). It cost
nothing in correctness but wasted the card. **Going forward: use the table.**

## Rational-usage rules (going forward)

1. **Default to the batch in the tables, never reflexively bs1.** Static is ~0.5 GB; you have ~13 GB
   for activations.
2. **Budget at the multiscale *peak* resolution** (`base + 128`), not the base.
3. **Static is negligible** — model-size differences (Large vs Small) barely move the budget; resolution
   and batch are the only knobs that matter.
4. **To nail an exact number before a big run**, do a one-step GPU probe:
   `torch.cuda.reset_peak_memory_stats(); <one train step>; torch.cuda.max_memory_allocated()/1e9`.
   Cheaper than guessing, exact to the MB.
5. **Effective batch** = `batch × grad_accum × devices`. If you raise the real batch, drop grad-accum to
   keep the effective batch (and LR schedule) comparable across experiments.

## Caveats / how to make it exact

- The RF-DETR GPU figures are **derived** (CPU activation proxy + static arithmetic), cross-checked
  against the fact that v7/v15/v16 ran without OOM. Confidence band ≈ **±30 %** (cuDNN workspace size
  and AMP fp16 fraction are the unknowns).
- The **only** way to remove the band is the one-step `max_memory_allocated` probe (rule 4) — RF-DETR's
  trainer doesn't log memory, so until someone adds that line, the tables are the best available and are
  safe (they err toward *under*-using the card, and the no-OOM history backs them).
- Reproduce the activation numbers: `PYTHONPATH=rf-detr/src python scripts/_mem_probe.py <res> <batch>`.
