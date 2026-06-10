"""Poll a Hafnia experiment until a meaningful event, then exit (one notification).

Backgrounded watcher for a launched training run. Exits — printing the reason — on the FIRST of:
  - terminal state (FAILED / SUCCEEDED / CANCELED)
  - an error signature in the logs (OOM / CUDA / Traceback / arg error)  -> fast early-failure catch
  - the first validation mAP appearing                                    -> the datapoint we want
  - a hard time cap (~3.5h)

Intermediate heartbeats go to stdout (readable via the task output file); the process only EXITS on an
event, so the single completion notification lands at the moment that matters.

    python scripts/_poll_experiment.py <experiment_id>
"""
import re
import sys
import time

from hafnia import http
from hafnia_cli.config import Config

EID = sys.argv[1] if len(sys.argv) > 1 else "8108ee90-0e23-422e-bcfd-0d4de494ca49"
MIN_STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # only fire on an eval at step >= this
cfg = Config()
BASE = cfg.config.platform_url
HDR = {"Authorization": cfg.api_key}

TERMINAL = {"TRAINING_SUCCEEDED", "TRAINING_FAILED", "FAILED", "CANCELED", "CANCELLED", "ERROR", "STOPPED"}
# Genuinely fatal signatures only (a benign albumentations version-check warning under network
# isolation prints "[Errno 99] Cannot assign requested address" — must NOT trip this).
ERR = re.compile(r"out of memory|CUDA out of memory|CUDA error|Traceback \(most recent|RuntimeError|"
                 r"AssertionError|Segmentation fault|Killed|OutOfMemoryError|terminate called|"
                 r"Fatal|core dumped|is not in the.*registry|KeyError|ModuleNotFound", re.I)
# Lines to ignore even if they look error-ish — known-benign (network-isolation warnings + the
# EXPECTED 80->10 class-head size mismatch when load_from a COCO checkpoint reinits vfnet_cls).
BENIGN = re.compile(r"check_version|albumentations|urlopen error|Errno 99|Cannot assign requested|"
                    r"version info|UserWarning|fetch_version|size mismatch|shape mismatch|"
                    r"will not be loaded|copying a param|unexpected key|missing keys?|"
                    r"are not used|be loaded", re.I)
EVAL = re.compile(r"ema_mAP_50_95['\"]?[:=]\s*([0-9.]+)|mAP_50_95['\"]?[:=]\s*([0-9.]+)|"
                  r"Average Precision.*?=\s*([0-9.]+)", re.I)

INTERVAL = 90
MAX_ITERS = 140  # ~3.5h


def state():
    try:
        return http.fetch(f"{BASE}/api/v1/experiments/{EID}", headers=HDR).get("state", "?")
    except Exception:
        return "?"


def recent_logs(n=250):
    try:
        r = http.fetch(f"{BASE}/api/v1/experiments/{EID}/logs?order=desc&limit={n}", headers=HDR)
        return r if isinstance(r, list) else r.get("data", [])
    except Exception:
        return []


def main():
    print(f"[poll] watching {EID} (every {INTERVAL}s, cap ~{MAX_ITERS * INTERVAL // 3600}h)", flush=True)
    seen_training = False
    for i in range(MAX_ITERS):
        st = state()
        logs = recent_logs()
        msgs = "\n".join(l.get("raw_message", "") for l in logs)
        # training started (build done) — heartbeat only, don't exit
        if not seen_training and ("Epoch" in msgs or "epoch" in msgs or st == "TRAINING"):
            seen_training = True
            print(f"[poll] iter {i}: build done, training started (state={st})", flush=True)
        # 1) eval mAP at step >= MIN_STEP?  Match RF-DETR (val/ema_mAP_50_95) OR mmdet (val/bbox_mAP).
        hit_eval = None
        for l in logs:
            mm = re.search(r"step=(\d+).*name='(val/(?:ema_mAP_50_95|bbox_mAP))'.*value=([0-9.]+)",
                           l.get("raw_message", ""))
            if mm and int(mm.group(1)) >= MIN_STEP:
                hit_eval = (int(mm.group(1)), mm.group(2), mm.group(3))
        if hit_eval:
            print(f"EVAL step={hit_eval[0]} {hit_eval[1]}={hit_eval[2]} "
                  f"(refs: v7 RF-DETR 0.354 EMA, ConvNeXt 0.312, Dima letterbox 0.30)", flush=True)
            return
        # 2) terminal state?
        if st in TERMINAL:
            tail = [l.get("raw_message", "") for l in logs[:8]]
            print(f"TERMINAL state={st}\n  " + "\n  ".join(tail), flush=True)
            return
        # 3) error signature (only after build; skip known-benign warnings)?
        if seen_training:
            hit = [l.get("raw_message", "") for l in logs
                   if ERR.search(l.get("raw_message", "")) and not BENIGN.search(l.get("raw_message", ""))][:5]
            if hit:
                print(f"ERROR_IN_LOGS state={st}\n  " + "\n  ".join(hit), flush=True)
                return
        print(f"[poll] iter {i}: state={st}, no eval yet", flush=True)
        time.sleep(INTERVAL)
    print(f"TIMEOUT after ~{MAX_ITERS * INTERVAL // 3600}h, last state={state()}", flush=True)


if __name__ == "__main__":
    main()
