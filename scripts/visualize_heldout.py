"""Render the worst HELD-OUT-camera cases (cross-camera failures) for the current
checkpoint: GT vs predictions, with MISSED objects highlighted (small misses extra-bold)
so the small/medium-object scale failure is visible.

    python scripts/visualize_heldout.py [weights/v5_best_ema.pth] [--topk 9]
"""
from __future__ import annotations
import sys, warnings, argparse
from pathlib import Path
warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rf-detr" / "src"))
import transformers  # noqa: E402
from transformers.utils import backbone_utils as _bu  # noqa: E402
for _n in ("BackboneConfigMixin", "BackboneMixin", "BackboneType"):
    if not hasattr(transformers, _n) and hasattr(_bu, _n):
        setattr(transformers, _n, getattr(_bu, _n))
import polars as pl  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from hafnia.dataset.hafnia_dataset import HafniaDataset  # noqa: E402
from rfdetr import RFDETRLarge  # noqa: E402

SHORT = ["Car", "Pickup", "SingleTrk", "ComboTrk", "HeavyDuty", "Trailer", "Moto", "Bicycle", "Van", "Person"]
HELDOUT_CAMS = {"5 POINTS WB", "GRANDVIEW - DELHI INT", "HWY 20 - OLD HWY WBA",
                "LOCUST CONNECTOR NB", "NW ARTERIAL - CHAVENELLE INT", "US 61 - TWIN VALLEY INT"}
OUT = REPO_ROOT / "visualization" / "heldout"
C_GT, C_TP, C_MIS, C_FP, C_MISS = (0, 200, 0), (40, 120, 255), (255, 150, 0), (235, 30, 30), (255, 0, 220)


def load_font(sz):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if Path(p).exists():
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1)
    inter = iw*ih
    return 0. if inter <= 0 else inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9)


def gt_boxes(row):
    W, H = row["width"], row["height"]
    return [((b["top_left_x"]*W, b["top_left_y"]*H, (b["top_left_x"]+b["width"])*W, (b["top_left_y"]+b["height"])*H), int(b["class_idx"]))
            for b in (row["bboxes"] or []) if b.get("task_name") == "object_detection"]


def small(box):
    return ((box[2]-box[0])*(box[3]-box[1]))**0.5 < 96  # small+medium (the failing regime)


def rect(d, box, color, w=3, dash=False):
    x1, y1, x2, y2 = box
    if not dash:
        d.rectangle([x1, y1, x2, y2], outline=color, width=w); return
    for (xa, ya, xb, yb) in [(x1, y1, x2, y1), (x1, y2, x2, y2), (x1, y1, x1, y2), (x2, y1, x2, y2)]:
        horiz = ya == yb; p = xa if horiz else ya; end = xb if horiz else yb
        while p < end:
            q = min(p+12, end)
            d.line([p, ya, q, yb] if horiz else [xa, p, xb, q], fill=color, width=w)
            p += 20


def label(d, box, txt, color, font):
    x1, y1 = box[0], box[1]
    tb = d.textbbox((0, 0), txt, font=font); tw, th = tb[2]-tb[0], tb[3]-tb[1]
    ty = max(0, y1-th-4)
    d.rectangle([x1, ty, x1+tw+6, ty+th+4], fill=color)
    d.text((x1+3, ty+1), txt, fill=(255, 255, 255), font=font)


def banner(img, text, font):
    d0 = ImageDraw.Draw(img); tb = d0.textbbox((0, 0), text, font=font); h = tb[3]-tb[1]+12
    out = Image.new("RGB", (img.width, img.height+h), (20, 20, 20))
    ImageDraw.Draw(out).text((8, 4), text, fill=(255, 255, 255), font=font)
    out.paste(img, (0, h)); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=str(REPO_ROOT/"weights"/"v5_best_ema.pth"))
    ap.add_argument("--topk", type=int, default=9)
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.ckpt}")
    model = RFDETRLarge(num_classes=10, resolution=704, pretrain_weights=args.ckpt); model.optimize_for_inference()
    ds = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
    df = ds.samples.with_columns(pl.col("camera_info").struct.field("name").alias("cam"))
    held = df.filter(pl.col("split").is_in(["train", "validation"]) & pl.col("cam").is_in(list(HELDOUT_CAMS)))
    print(f"[data] {len(held)} held-out-camera images")
    fbig, flbl = load_font(30), load_font(18)
    results = []
    for i in range(len(held)):
        row = held.row(i, named=True)
        det = model.predict(Image.open(row["file_path"]).convert("RGB"), threshold=args.thr)
        preds = sorted([(tuple(float(v) for v in b), int(c), float(s)) for b, c, s in zip(det.xyxy, det.class_id, det.confidence)], key=lambda x: -x[2])
        gts = gt_boxes(row)
        matched = [False]*len(gts); tp, mis, fp = [], [], []
        for box, cls, conf in preds:
            bi, bj = 0., -1
            for j, g in enumerate(gts):
                if matched[j]: continue
                v = iou(box, g[0])
                if v > bi: bi, bj = v, j
            if bi >= 0.5 and bj >= 0:
                matched[bj] = True
                (tp if gts[bj][1] == cls else mis).append((box, cls, conf, gts[bj][1] if gts[bj][1] != cls else None))
            else:
                fp.append((box, cls, conf, None))
        missed = [g for j, g in enumerate(gts) if not matched[j]]
        small_missed = sum(1 for g in missed if small(g[0]))
        score = len(missed) + small_missed + len(fp)*0.5 + len(mis)
        results.append((score, row, gts, tp, mis, fp, missed, small_missed))
    results.sort(key=lambda r: -r[0])
    panels = []
    for rank, (score, row, gts, tp, mis, fp, missed, sm) in enumerate(results[:args.topk]):
        base = Image.open(row["file_path"]).convert("RGB")
        g = base.copy(); dg = ImageDraw.Draw(g)
        for box, c in gts:
            rect(dg, box, C_GT, 2); label(dg, box, SHORT[c], C_GT, flbl)
        g = banner(g, f"GT — {row['cam']} ({len(gts)} objs)", fbig)
        p = base.copy(); dp = ImageDraw.Draw(p)
        for box, c in missed:
            w = 4 if small(box) else 2
            rect(dp, box, C_MISS, w, dash=True); label(dp, box, f"MISS:{SHORT[c]}", C_MISS, flbl)
        for box, c, cf, _ in tp: rect(dp, box, C_TP, 2)
        for box, c, cf, gtc in mis:
            rect(dp, box, C_MIS, 3); label(dp, box, f"{SHORT[c]}?", C_MIS, flbl)
        for box, c, cf, _ in fp:
            rect(dp, box, C_FP, 2); label(dp, box, f"FP:{SHORT[c]}", C_FP, flbl)
        p = banner(p, f"PRED  missed={len(missed)} (small/med={sm})  FP={len(fp)} misclass={len(mis)}", fbig)
        gap = 10
        combo = Image.new("RGB", (g.width+p.width+gap, max(g.height, p.height)), (20, 20, 20))
        combo.paste(g, (0, 0)); combo.paste(p, (g.width+gap, 0))
        s = 1600/combo.width; combo = combo.resize((int(combo.width*s), int(combo.height*s)))
        fn = OUT / f"heldout_{rank+1:02d}_{row['cam'].replace(' ', '').replace('-', '')[:14]}.jpg"
        combo.save(fn, quality=88)
        panels.append(p.resize((int(p.width*0.33), int(p.height*0.33))))
        print(f"  {fn.name}  missed={len(missed)} small/med-missed={sm} fp={len(fp)}")
    if panels:
        cols = 3; rows = (len(panels)+cols-1)//cols
        cw, ch = max(x.width for x in panels), max(x.height for x in panels)
        grid = Image.new("RGB", (cols*cw, rows*ch), (10, 10, 10))
        for idx, x in enumerate(panels): grid.paste(x, ((idx % cols)*cw, (idx//cols)*ch))
        grid.save(OUT/"montage_heldout.jpg", quality=85)
        print("  montage_heldout.jpg")
    print("[done]", OUT)


if __name__ == "__main__":
    main()
