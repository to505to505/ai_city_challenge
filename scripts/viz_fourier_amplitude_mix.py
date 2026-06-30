"""Visualize the FACT Fourier amplitude-mix (phase-preserving style transfer) on real eccv data.

Renders four figures into ``.data/viz/fourier_demo/`` using the *real* trainer code
(``rfdetr.datasets.xcity_augs._fourier_amplitude_mix`` / ``FourierAmplitudeMix``):

1. ``01_style_grid.png``   — CONTENT (phase) + STYLE DONOR (amplitude) -> result across a lambda sweep.
2. ``02_boxsafe.png``      — GT boxes + edge maps before/after: structure (=> box locations) unchanged.
3. ``03_spectra.png``      — Fourier log-amplitude spectra: the donor's low-freq center swapped in.
4. ``04_training_regime.png`` — what the trainer ACTUALLY applies (beta=0.05, lambda~U(0,0.3)).

Run:
    PYTHONPATH=rf-detr/src python scripts/viz_fourier_amplitude_mix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rf-detr" / "src"))

from rfdetr.datasets.xcity_augs import (  # noqa: E402
    FourierAmplitudeMix,
    _fourier_amplitude_mix,
    set_reference_image_dir,
)

DATA = REPO / ".data" / "datasets" / "eccv-cross-city"
OUT = REPO / ".data" / "viz" / "fourier_demo"
MAX_SIDE = 640  # work/display resolution (full-res FFT is identical in character, just slower)


def load_rgb(path: Path, max_side: int = MAX_SIDE) -> np.ndarray:
    """Load *path* as uint8 RGB, longest side resized to <= ``max_side`` (aspect kept)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = max_side / max(w, h)
        if s < 1.0:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        return np.asarray(im)


def match_size(ref: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Resize *ref* to the H x W of *like* (area interpolation), as the real transform does."""
    return cv2.resize(ref, (like.shape[1], like.shape[0]), interpolation=cv2.INTER_AREA)


def scan_frames() -> List[Dict[str, Any]]:
    """Scan the labeled frames; return per-frame color stats + boxes for picking donors/content."""
    df = pd.read_parquet(DATA / "annotations.parquet")
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        fp = DATA / r["file_path"]
        if not fp.exists():
            continue
        with Image.open(fp) as im:
            im = im.convert("RGB")
            im.thumbnail((128, 128))
            t = np.asarray(im).astype(np.float32)
        luma = float(0.299 * t[..., 0].mean() + 0.587 * t[..., 1].mean() + 0.114 * t[..., 2].mean())
        warm = float(t[..., 0].mean() - t[..., 2].mean())  # R - B  (>0 warm, <0 cool)
        vi = r.get("video_info") or {}
        cap = (vi.get("captured_at") or "") if isinstance(vi, dict) else ""
        hour = int(cap[11:13]) if len(cap) >= 13 and cap[11:13].isdigit() else -1
        cam = r.get("camera_info") or {}
        rows.append(
            {
                "path": fp,
                "luma": luma,
                "warm": warm,
                "hour": hour,
                "camera": cam.get("name", "?") if isinstance(cam, dict) else "?",
                "n_boxes": len(r["bboxes"]) if r["bboxes"] is not None else 0,
                "bboxes": r["bboxes"],
            }
        )
    return rows


def pick(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Pick a well-lit, object-rich CONTENT frame and three maximally-distinct STYLE donors."""
    # CONTENT: daytime, mid-bright, the richest structure (so box/edge preservation is visible).
    lit = [r for r in rows if 95 <= r["luma"] <= 175 and r["n_boxes"] >= 4]
    content = max(lit or rows, key=lambda r: r["n_boxes"])
    others = [r for r in rows if r["path"] != content["path"]]
    night = min(others, key=lambda r: r["luma"])            # darkest -> night look
    warm = max(others, key=lambda r: r["warm"])             # warmest -> sodium / sunset cast
    cool = max(others, key=lambda r: r["luma"] - r["warm"])  # bright & blue -> overcast daylight
    # de-dup donors if extremes collide
    donors: List[Dict[str, Any]] = []
    for d, tag in [(night, "night"), (warm, "warm"), (cool, "cool/bright")]:
        if all(d["path"] != x["path"] for x in donors):
            d = dict(d, tag=tag)
            donors.append(d)
    return content, donors


def draw_boxes(img: np.ndarray, bboxes: Any, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw normalized (top_left_x/y, width, height) GT boxes on a copy of *img*."""
    out = img.copy()
    h, w = out.shape[:2]
    if bboxes is None:
        return out
    for b in bboxes:
        x0 = int(round(b["top_left_x"] * w))
        y0 = int(round(b["top_left_y"] * h))
        x1 = int(round((b["top_left_x"] + b["width"]) * w))
        y1 = int(round((b["top_left_y"] + b["height"]) * h))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def log_spectrum(img: np.ndarray) -> np.ndarray:
    """Centered log-amplitude spectrum of the luma channel, normalized to [0, 1] for display."""
    g = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    lg = np.log1p(mag)
    return (lg - lg.min()) / (lg.max() - lg.min() + 1e-8)


def fig_style_grid(content: np.ndarray, donors: List[Dict[str, Any]], donor_imgs: List[np.ndarray]) -> None:
    """Figure 1: CONTENT + DONOR -> result across a lambda sweep (beta=0.05)."""
    lams = [0.25, 0.5, 0.75, 1.0]
    ncols = 2 + len(lams)
    fig, axes = plt.subplots(len(donors), ncols, figsize=(3.0 * ncols, 3.0 * len(donors)))
    axes = np.atleast_2d(axes)
    headers = ["CONTENT\n(phase — fixed)", "STYLE DONOR\n(amplitude source)"] + [f"λ = {l:g}" for l in lams]
    for i, (d, dimg) in enumerate(zip(donors, donor_imgs)):
        ref = match_size(dimg, content)
        cells = [content, dimg] + [_fourier_amplitude_mix(content, ref, beta=0.05, lam=l) for l in lams]
        for j, cell in enumerate(cells):
            ax = axes[i, j]
            ax.imshow(cell)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(headers[j], fontsize=11, fontweight="bold")
            if j == 1:
                ax.set_ylabel(f"donor: {d['tag']}\n({d['camera']})", fontsize=9)
    fig.suptitle(
        "FACT Fourier Amplitude Mix — phase (content/structure) kept, low-freq amplitude (style) blended in",
        fontsize=14, fontweight="bold",
    )
    fig.text(0.5, 0.005, "Same scene, same objects in the same pixels — only colour/lighting/texture shift. "
             "Trainer uses λ~U(0, 0.3) (≈ the second column).", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT / "01_style_grid.png", dpi=110)
    plt.close(fig)


def fig_boxsafe(content: np.ndarray, bboxes: Any, donor_img: np.ndarray) -> None:
    """Figure 2: GT boxes + Canny edges before/after prove the structure (box locations) is intact."""
    ref = match_size(donor_img, content)
    styled = _fourier_amplitude_mix(content, ref, beta=0.05, lam=0.9)  # strong mix to stress-test
    g_c = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY)
    g_s = cv2.cvtColor(styled, cv2.COLOR_RGB2GRAY)
    e_c, e_s = cv2.Canny(g_c, 80, 160), cv2.Canny(g_s, 80, 160)
    diff = np.abs(content.astype(int) - styled.astype(int)).mean(axis=2)
    edge_agree = float((e_c == e_s).mean()) * 100.0

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ax[0, 0].imshow(draw_boxes(content, bboxes, (0, 255, 0))); ax[0, 0].set_title("CONTENT + GT boxes")
    ax[0, 1].imshow(draw_boxes(styled, bboxes, (0, 255, 0)))
    ax[0, 1].set_title("RESTYLED (λ=0.9) + SAME GT boxes\n→ boxes still wrap the same objects")
    im = ax[0, 2].imshow(diff, cmap="magma"); ax[0, 2].set_title("|pixel difference|  (low-freq tint only)")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)
    ax[1, 0].imshow(e_c, cmap="gray"); ax[1, 0].set_title("Canny edges — CONTENT")
    ax[1, 1].imshow(e_s, cmap="gray"); ax[1, 1].set_title(f"Canny edges — RESTYLED\n{edge_agree:.1f}% pixels identical")
    ax[1, 2].imshow((e_c != e_s), cmap="hot"); ax[1, 2].set_title("edge disagreement (≈ none)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Box-safe by construction: phase preserved ⇒ edges/structure unmoved ⇒ GT boxes stay valid",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "02_boxsafe.png", dpi=110)
    plt.close(fig)


def fig_spectra(content: np.ndarray, donor_img: np.ndarray) -> None:
    """Figure 3: log-amplitude spectra; the low-freq window adopts the donor's energy."""
    ref = match_size(donor_img, content)
    styled = _fourier_amplitude_mix(content, ref, beta=0.05, lam=1.0)
    h, w = content.shape[:2]
    bh, bw = max(1, round(h * 0.05)), max(1, round(w * 0.05))
    panels = [(content, "CONTENT"), (ref, "DONOR"), (styled, "RESULT (λ=1)")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for j, (img, name) in enumerate(panels):
        ax[0, j].imshow(img); ax[0, j].set_title(name); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
        sp = log_spectrum(img)
        ax[1, j].imshow(sp, cmap="viridis")
        ax[1, j].add_patch(mpatches.Rectangle((w // 2 - bw, h // 2 - bh), 2 * bw, 2 * bh,
                                              fill=False, edgecolor="red", lw=2))
        ax[1, j].set_title(f"{name} — log|FFT| (red = β=0.05 window)")
        ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
    fig.suptitle("Inside the red low-frequency window, RESULT's amplitude is the DONOR's; "
                 "phase (everywhere) stays CONTENT's", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "03_spectra.png", dpi=110)
    plt.close(fig)


def fig_training_regime(content: np.ndarray) -> None:
    """Figure 4: the actual augmentation the model sees (FourierAmplitudeMix, beta=0.05, lambda_max=0.3)."""
    set_reference_image_dir(str(DATA / "data"))
    import rfdetr.datasets.xcity_augs as m
    m._REF_POOL = None  # rebuild pool from the real dataset
    t = FourierAmplitudeMix(p=1.0, beta=0.05, lambda_max=0.3)
    np.random.seed(0)
    n = 5
    fig, ax = plt.subplots(1, n + 1, figsize=(3.0 * (n + 1), 3.2))
    ax[0].imshow(content); ax[0].set_title("ORIGINAL", fontweight="bold")
    for k in range(n):
        ax[k + 1].imshow(t(image=content)["image"]); ax[k + 1].set_title(f"aug #{k + 1}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("What the trainer actually applies each step: FourierAmplitudeMix(p=…, β=0.05, λ~U(0,0.3)) "
                 "— subtle, realistic cross-city restyle (boxes untouched)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "04_training_regime.png", dpi=110)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[scan] reading frames from {DATA}/annotations.parquet …")
    rows = scan_frames()
    content_meta, donor_metas = pick(rows)
    print(f"[pick] CONTENT: {content_meta['path'].name}")
    print(f"        luma={content_meta['luma']:.0f}  boxes={content_meta['n_boxes']}  cam={content_meta['camera']}")
    for d in donor_metas:
        print(f"[pick] DONOR ({d['tag']:>11}): {d['path'].name}  luma={d['luma']:.0f}  warm={d['warm']:+.0f}  cam={d['camera']}")

    content = load_rgb(content_meta["path"])
    donor_imgs = [load_rgb(d["path"]) for d in donor_metas]

    fig_style_grid(content, donor_metas, donor_imgs)
    fig_boxsafe(content, content_meta["bboxes"], donor_imgs[0])
    fig_spectra(content, donor_imgs[0])
    fig_training_regime(content)

    manifest = {
        "content": content_meta["path"].name,
        "donors": [{"tag": d["tag"], "file": d["path"].name, "camera": d["camera"]} for d in donor_metas],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote 4 figures + manifest.json to {OUT}")


if __name__ == "__main__":
    main()
