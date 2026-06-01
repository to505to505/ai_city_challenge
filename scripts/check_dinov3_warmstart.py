"""Decisive smoke test for the DINOv3-RF-DETR WARM-START path.

The DINOv2 RF-DETR checkpoint (rf-detr-large-2026.pth) carries a DINOv2 backbone + a COCO/O365
head. We want to fine-tune a DINOv3-backbone RF-DETR FROM that checkpoint, i.e.:
    DINOv3 pretrained backbone (bundled)  +  RF-DETR pretrained projector/transformer/head.

Risks this script checks empirically (no platform credits spent):
  1. CRASH: a key present in BOTH the checkpoint and the DINOv3 model with a DIFFERENT shape makes
     load_state_dict raise RuntimeError (strict=False only tolerates missing/unexpected, not shape).
  2. CLOBBER: a same-name/same-shape backbone key would silently overwrite a bundled DINOv3 tensor
     with a DINOv2 value (mild corruption).
  3. PRESERVE: after warm-start, the DINOv3 backbone must still equal the bundled DINOv3 weights
     (proving we did NOT end up with a random backbone — the whole point of the backbone.py fix).

    python scripts/check_dinov3_warmstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rf-detr" / "src"))

import torch  # noqa: E402

from rfdetr import RFDETRLarge  # noqa: E402

from train import prepare_warmstart_for_encoder  # noqa: E402  (scripts/ is on sys.path[0])

CKPT = REPO / "weights" / "rf-detr-large-2026.pth"
RES = 640  # div by both 32 (dinov2 windowed) and 16 (dinov3) — fast CPU build
BB = "backbone.0.encoder.encoder."  # prefix of the actual ViT backbone tensors


def get_sd(m) -> dict:
    """Underlying LWDETR nn.Module state_dict (RFDETR -> .model ctx -> .model nn.Module)."""
    return m.model.model.state_dict()


def main() -> None:
    print(f"[1/4] building DINOv3 RF-DETR (NO warm-start) @res={RES} to read its key/shape map ...")
    m0 = RFDETRLarge(encoder="dinov3_small", num_classes=10, resolution=RES, pretrain_weights=None)
    sd_model = get_sd(m0)
    bb_keys = {k: v.shape for k, v in sd_model.items() if k.startswith(BB)}
    print(f"      model has {len(sd_model)} tensors total, {len(bb_keys)} in the DINOv3 backbone")
    # snapshot a couple of backbone tensors to verify preservation after warm-start
    probe_keys = [k for k in bb_keys][:3] + [k for k in bb_keys if k.endswith("weight")][-2:]
    probe0 = {k: sd_model[k].clone() for k in probe_keys}

    print(f"[2/4] loading checkpoint {CKPT.name} to compare keys/shapes ...")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ck_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    ck_shapes = {k: tuple(v.shape) for k, v in ck_sd.items() if hasattr(v, "shape")}

    # cross-check overlap
    crash, clobber, nonbb_match = [], [], 0
    for k, shp in ck_shapes.items():
        if k in sd_model:
            same = tuple(sd_model[k].shape) == shp
            if not same:
                crash.append((k, shp, tuple(sd_model[k].shape)))
            elif k.startswith(BB):
                clobber.append(k)
            elif same:
                nonbb_match += 1
    ck_bb = [k for k in ck_shapes if k.startswith(BB)]
    print(f"      checkpoint: {len(ck_shapes)} tensors, {len(ck_bb)} backbone")
    print(f"      -> SAME-NAME/DIFF-SHAPE (would CRASH load): {len(crash)}")
    for k, a, b in crash[:8]:
        print(f"           {k}: ckpt{a} vs model{b}")
    print(f"      -> backbone keys that would CLOBBER (same name+shape): {len(clobber)}")
    for k in clobber[:8]:
        print(f"           {k}")
    print(f"      -> non-backbone tensors that MATCH and will load (head/neck/transformer): {nonbb_match}")

    print("[3/4] stripping DINOv2 backbone keys (train.py path) + building warm-started model ...")
    safe_ckpt = prepare_warmstart_for_encoder(CKPT, "dinov3_small", REPO / ".data")
    if safe_ckpt == CKPT:
        print("      !!! strip helper returned the ORIGINAL path — backbone keys NOT removed. STOP.")
        return
    try:
        m1 = RFDETRLarge(encoder="dinov3_small", num_classes=10, resolution=RES, pretrain_weights=str(safe_ckpt))
        print("      build OK — load_state_dict did not raise on the stripped checkpoint")
    except RuntimeError as e:
        print(f"      !!! RuntimeError during warm-start load: {str(e)[:400]}")
        return

    print("[4/4] verifying backbone PRESERVED, head WARM-STARTED, and a forward pass runs ...")
    sd1 = get_sd(m1)
    n_same = sum(int(torch.equal(sd1[k], probe0[k])) for k in probe_keys)
    print(f"      {n_same}/{len(probe_keys)} probed DINOv3 backbone tensors identical to bundled init (expect all)")
    # confirm a non-backbone (transformer) tensor actually CHANGED (loaded from ckpt, not random)
    nb = [k for k in sd_model if k.startswith("transformer.") and k.endswith("weight")
          and k in ck_shapes and ck_shapes[k] == tuple(sd_model[k].shape)][:1]
    head_loaded = None
    if nb:
        head_loaded = not torch.equal(sd1[nb[0]], sd_model[nb[0]])
        print(f"      transformer tensor {nb[0]} changed vs random-init: {head_loaded} (expect True)")
    with torch.no_grad():
        out = m1.model.model(torch.randn(1, 3, RES, RES))
    pl, pb = tuple(out["pred_logits"].shape), tuple(out["pred_boxes"].shape)
    # train mode emits num_queries(300) * group_detr(13) = 3900 queries; eval collapses to 300.
    print(f"      forward OK: pred_logits={pl} pred_boxes={pb} (train-mode 3900 = 300*group_detr 13)")
    shapes_ok = pl[0] == 1 and pl[2] == 11 and pb == (pl[0], pl[1], 4) and pl[1] in (300, 3900)
    ok = (n_same == len(probe_keys)) and (head_loaded is not False) and shapes_ok
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} — DINOv3 backbone preserved, "
          f"{nonbb_match} head/neck tensors warm-started, forward shapes correct (11 classes).")


if __name__ == "__main__":
    main()
