"""Collect best val metrics for every experiment on the platform (ours + Dima's) and print a
structured snapshot used to build docs/model_metrics.md.

RF-DETR runs log torchmetrics names (val/ema_mAP_50_95, val/mAP_50_95, val/mAP_50, val/mAP_75,
val/mAR, val/AP/<class>); our YOLO trainer forwards val/mAP_50_95; Dima's mmdet logs coco/bbox_mAP*.
Reads creds from ~/.hafnia (never prints the key).

    python scripts/collect_metrics.py
"""
from __future__ import annotations

import re
from urllib.parse import quote

from hafnia import http
from hafnia_cli.config import Config

cfg = Config()
BASE = cfg.config.platform_url
HDR = {"Authorization": cfg.api_key}

RF_RE = r"name='(val/ema_mAP_50_95|val/mAP_50_95|val/mAP_50|val/mAP_75|val/mAR|val/F1)'.*?value=([0-9.]+)"
PC_RE = r"name='val/AP/([^']+)'.*?value=([0-9.]+)"
MM_RE = r"(coco/bbox_mAP|coco/bbox_mAP_50|coco/bbox_mAP_75):\s*([0-9.]+)"


def collect(eid: str, regex: str, stop: str, max_pages: int = 18) -> dict:
    vals: dict = {}
    cursor = None
    for _ in range(max_pages):
        url = f"{BASE}/api/v1/experiments/{eid}/logs?order=desc&limit=1000"
        if cursor:
            url += f"&before={quote(cursor)}"
        logs = http.fetch(url, headers=HDR)
        if not logs:
            break
        cursor = logs[-1]["created_at"]
        for line in logs:
            for nm, v in re.findall(regex, line.get("raw_message", "")):
                vals.setdefault(nm, []).append(float(v))
        if cursor < stop:
            break
    return vals


def list_all_experiments() -> list:
    """Fetch ALL experiments, following the API's pagination cursor (one page is NOT the full set)."""
    url = f"{BASE}/api/v1/experiments?limit=100&order=desc"
    items: list = []
    for _ in range(20):
        r = http.fetch(url, headers=HDR)
        items += r.get("data", [])
        nxt = r.get("next")
        if not nxt:
            break
        url = nxt if str(nxt).startswith("http") else BASE + nxt
    return items


def main() -> None:
    by_name = {e.get("name"): e for e in list_all_experiments()}

    # (label, name, kind) — kind selects the metric vocabulary
    models = [
        ("v6 RF-DETR DINOv2 704 DG-aug", "rfdetr_large_dg_finetune_lite_v6", "rf"),
        ("v7 RF-DETR DINOv2 896+ms", "rfdetr_large_hires896_ms_lite_v7", "rf"),
        ("v9 RF-DETR DINOv3 704", "rfdetr_dinov3s_704_lite_v9", "rf"),
        ("v11 RF-DETR DINOv2 896+ms+fisheye/night", "rfdetr_dinov2_896_fisheyenight_lite_v11", "rf"),
        ("v12 RF-DETR DINOv2 896+ms+CD-FKD", "rfdetr_dinov2_896_cdfkd_lite_v12", "rf"),
        ("v8 YOLO26-L 1280", "yolo26l_hires1280_lite_v8", "yolo"),
        ("Dima ConvNeXt (mmdet)", "convnext_v2_v2", "mm"),
    ]
    for label, name, kind in models:
        e = by_name.get(name)
        if not e:
            print(f"\n### {label}\n  (experiment not found — may be deleted)")
            continue
        cr = e.get("credits_consumed") or 0
        dur = round((e.get("training_duration_seconds") or 0) / 3600, 1)
        print(f"\n### {label}")
        print(f"  state={e.get('state')}  ~{cr}cr  {dur}h  id={e.get('id')[:8]}")
        if kind == "rf":
            v = collect(e["id"], RF_RE, "2026-05-27T00:00")
            mx = lambda k: (max(v[k]) if v.get(k) else None)
            print(f"  EMA mAP@50:95={mx('val/ema_mAP_50_95')}  reg={mx('val/mAP_50_95')}  "
                  f"mAP50={mx('val/mAP_50')}  mAP75={mx('val/mAP_75')}  mAR={mx('val/mAR')}  F1={mx('val/F1')}")
            pc = collect(e["id"], PC_RE, "2026-05-27T00:00")
            if pc:
                print("  per-class AP (best, regular): " +
                      ", ".join(f"{c}={max(vs):.3f}" for c, vs in sorted(pc.items(), key=lambda kv: -max(kv[1]))))
        elif kind == "yolo":
            v = collect(e["id"], RF_RE, "2026-05-30T00:00")
            mx = lambda k: (max(v[k]) if v.get(k) else None)
            print(f"  mAP@50:95={mx('val/mAP_50_95')}  mAP50={mx('val/mAP_50')}")
        else:  # mmdet
            v = collect(e["id"], MM_RE, "2026-06-01T00:00")
            mx = lambda k: (max(v[k]) if v.get(k) else None)
            print(f"  coco/bbox_mAP={mx('coco/bbox_mAP')}  mAP50={mx('coco/bbox_mAP_50')}  mAP75={mx('coco/bbox_mAP_75')}")


if __name__ == "__main__":
    main()
