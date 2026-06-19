"""Hafnia experiment leaderboard — best validation mAP per experiment, ranked.

Enumerates every experiment the API key can see, scans each one's logs for COCO mAP, and prints a
ranked table. Works for both frameworks: mmdet logs `coco/bbox_mAP:` lines, RF-DETR (and mmdet) emit
the universal pycocotools `Average Precision (AP) @[ IoU=0.50:0.95 ... ] =` summary — we match both
and take the max per experiment.

Run on the machine whose key HAS access to the experiments, or pass a key explicitly:
    python tools/hafnia_leaderboard.py
    HAFNIA_API_KEY='ApiKey ....' python tools/hafnia_leaderboard.py
"""

from __future__ import annotations

import os
import re
from urllib.parse import quote

from hafnia_cli.config import Config
from hafnia import http

BASE = "https://api.hafnia.milestonesys.com"
KEY = os.environ.get("HAFNIA_API_KEY") or Config().api_key
HDR = {"Authorization": KEY}

# mAP@[.50:.95] — mmdet (coco/bbox_mAP:), pycocotools summary, and RF-DETR (val/[ema_]mAP_50_95)
RE_9550 = re.compile(r"coco/bbox_mAP:\s*([0-9.]+)")
RE_9550_COCO = re.compile(r"\(AP\)\s*@\[\s*IoU=0\.50:0\.95.*?\]\s*=\s*([0-9.]+)")
RE_9550_RFDETR = re.compile(r"val/(?:ema_)?mAP_50_95'\s+ent_type='metric'\s+value=([0-9.]+)")
# Crashed runs keep only a ~100-line log tail on the platform; the per-metric lines are gone but
# RF-DETR's "Best EMA mAP improved to X (epoch N)" announcement usually survives (caught v20).
RE_9550_BEST_EMA = re.compile(r"Best EMA mAP improved to\s*([0-9.]+)")
PATS_9550 = (RE_9550, RE_9550_COCO, RE_9550_RFDETR, RE_9550_BEST_EMA)
# mAP@.50 — note the closing quote so val/mAP_50' does NOT match val/mAP_50_95'
RE_50 = re.compile(r"coco/bbox_mAP_50:\s*([0-9.]+)")
RE_50_COCO = re.compile(r"\(AP\)\s*@\[\s*IoU=0\.50\s*\|.*?\]\s*=\s*([0-9.]+)")
RE_50_RFDETR = re.compile(r"val/(?:ema_)?mAP_50'\s+ent_type='metric'\s+value=([0-9.]+)")
PATS_50 = (RE_50, RE_50_COCO, RE_50_RFDETR)


def fetch(path: str):
    return http.fetch(BASE + path, headers=HDR)


def list_experiments() -> list[dict]:
    out, url = [], "/api/v1/experiments?limit=100"
    while url:
        r = fetch(url)
        data = r.get("data", []) if isinstance(r, dict) else r
        out += data or []
        nxt = r.get("next") if isinstance(r, dict) else None
        url = (nxt.replace(BASE, "") if nxt and nxt.startswith("http") else nxt) if nxt else None
    return out


def _max_match(pats, msg: str) -> float | None:
    vals = [float(m.group(1)) for p in pats if (m := p.search(msg))]
    return max(vals) if vals else None


def best_map(eid: str, max_pages: int = 70) -> tuple[float | None, float | None, int]:
    """Return (best mAP@.50:.95, best mAP@.50, number of mAP readings seen).

    mAP@.50:.95 and mAP@.50 are tracked independently (RF-DETR logs them on separate lines), and
    the max includes RF-DETR's EMA series — so this is the BEST the run achieved on its own val split.
    """
    best, best50, n_pts, cursor, pages = -1.0, -1.0, 0, None, 0
    while pages < max_pages:
        url = f"/api/v1/experiments/{eid}/logs?order=desc&limit=1000"
        if cursor:
            url += f"&before={quote(cursor)}"
        try:
            logs = fetch(url)
        except Exception:
            break
        if not logs:
            break
        cursor = logs[-1]["created_at"]
        pages += 1
        for l in logs:
            msg = l.get("raw_message", "")
            v = _max_match(PATS_9550, msg)
            if v is not None:
                n_pts += 1
                best = max(best, v)
            v50 = _max_match(PATS_50, msg)
            if v50 is not None:
                best50 = max(best50, v50)
    return (best if best >= 0 else None), (best50 if best50 >= 0 else None), n_pts


def main() -> None:
    exps = list_experiments()
    print(f"# {len(exps)} experiments visible to this key\n", flush=True)

    rows = []
    for e in exps:
        eid, name, state = e.get("id"), e.get("name"), e.get("state")
        mp, m50, n = best_map(eid)
        rows.append((mp, m50, n, name, state, eid))
        tag = f"{mp:.4f}" if mp is not None else "  -  "
        print(f"  scanned {name:<40} {state:<18} best={tag}", flush=True)

    scored = sorted([r for r in rows if r[0] is not None], key=lambda r: -r[0])
    empty = [r for r in rows if r[0] is None]

    print("\n=== LEADERBOARD — best validation mAP@[.50:.95] ===")
    print(f"{'#':>2}  {'mAP@.50:.95':>11}  {'mAP@.50':>8}  {'pts':>3}  {'state':<18} name")
    for i, (mp, m50, n, name, state, _) in enumerate(scored, 1):
        m50s = f"{m50:.3f}" if m50 is not None else "  -  "
        print(f"{i:>2}  {mp:>11.4f}  {m50s:>8}  {n:>3}  {state:<18} {name}")

    if empty:
        print("\n=== no mAP found (failed / in-progress / different log format) ===")
        for _, _, _, name, state, eid in empty:
            print(f"   {state:<18} {name}  [{eid}]")


if __name__ == "__main__":
    main()
