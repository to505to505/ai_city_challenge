"""Fetch RF-DETR pretrain weights into ./weights so a fresh clone can build.

Why this exists
---------------
The repo's `.gitignore` ignores `weights/` (we don't commit 135 MB binaries), so a
fresh clone has no pretrain checkpoint. The Hafnia Training-aaS container is
network-isolated and CANNOT download weights at runtime, so the Dockerfile bundles
`weights/` into the trainer.zip (`COPY weights ./weights`). This script populates
`weights/` locally BEFORE you run `hafnia experiment create`.

Source of truth for URL + MD5: rf-detr/src/rfdetr/assets/model_weights.py
(class ModelWeights). The default model matches RFDETRLargeConfig.pretrain_weights
in rf-detr/src/rfdetr/config.py ("rf-detr-large-2026.pth").

Usage
-----
    python scripts/download_weights.py                 # default: rf-detr-large-2026.pth
    python scripts/download_weights.py --model rf-detr-large.pth
    python scripts/download_weights.py --force         # re-download even if present

Stdlib only — no third-party deps, safe to run on a bare clone.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"

# (filename -> (url, md5)). Mirror of rf-detr ModelWeights registry; keep in sync if
# the upstream registry changes. Only the entries we actually use are listed.
WEIGHTS = {
    "rf-detr-large-2026.pth": (
        "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
        "5cb72153541cbcb9aa6efa26222acc75",
    ),
    "rf-detr-large.pth": (
        "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
        "992c8e862aa733a7bb2777e45d49f1a0",
    ),
}

DEFAULT_MODEL = "rf-detr-large-2026.pth"  # == RFDETRLargeConfig.pretrain_weights


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _disp(path: Path) -> str:
    """Path relative to the repo root for display, or absolute if it lives elsewhere."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}GB"


def download(url: str, dst_tmp: Path) -> None:
    """Stream `url` to `dst_tmp` with a simple progress line."""
    req = urllib.request.Request(url, headers={"User-Agent": "rf-detr-weights-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted https host)
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        chunk = 1 << 20
        with dst_tmp.open("wb") as out:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                got += len(buf)
                if total:
                    pct = got / total * 100
                    print(f"\r  downloading… {human(got)}/{human(total)} ({pct:4.1f}%)", end="", flush=True)
                else:
                    print(f"\r  downloading… {human(got)}", end="", flush=True)
    print()


def fetch(model: str, force: bool) -> int:
    if model not in WEIGHTS:
        print(f"[error] unknown model {model!r}. Known: {', '.join(WEIGHTS)}", file=sys.stderr)
        return 2
    url, expected_md5 = WEIGHTS[model]
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / model

    if dst.exists() and not force:
        print(f"[check] {_disp(dst)} exists — verifying MD5…")
        actual = md5_of(dst)
        if actual == expected_md5:
            print(f"[skip] already present and MD5 matches ({expected_md5}). Nothing to do.")
            return 0
        print(f"[warn] MD5 mismatch (got {actual}, expected {expected_md5}). Re-downloading.")

    print(f"[fetch] {model}\n        from {url}")
    # Download to a temp file in the same dir, verify, then atomically move into place,
    # so an interrupted run never leaves a corrupt weights file that later "looks" present.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{model}.", suffix=".part", dir=str(WEIGHTS_DIR))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        download(url, tmp)
        actual = md5_of(tmp)
        if actual != expected_md5:
            print(f"[error] MD5 verification FAILED: got {actual}, expected {expected_md5}", file=sys.stderr)
            return 1
        os.replace(tmp, dst)  # atomic on the same filesystem
        print(f"[ok] saved {_disp(dst)} ({human(dst.stat().st_size)}), MD5 verified.")
        # local provenance note (weights/ is gitignored, so this is a local-only record)
        try:
            (WEIGHTS_DIR / "PROVENANCE.md").write_text(
                f"# weights provenance\n\n- `{model}`\n  - url: {url}\n  - md5: {expected_md5}\n"
                "  - fetched by: scripts/download_weights.py\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return 0
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"[error] download failed: {exc}", file=sys.stderr)
        print("        (Run this on a machine WITH internet — the Hafnia cloud is network-isolated.)", file=sys.stderr)
        return 1
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(WEIGHTS), help="which checkpoint to fetch")
    ap.add_argument("--force", action="store_true", help="re-download even if a valid file is already present")
    args = ap.parse_args()
    sys.exit(fetch(args.model, args.force))


if __name__ == "__main__":
    main()
