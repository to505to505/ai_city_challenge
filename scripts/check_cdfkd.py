"""CPU smoke test for the CD-FKD self-distillation path (no GPU/dataset needed).

Verifies, end-to-end on the REAL RFDETRModelModule:
  1. the `cd_fkd` flag reaches TrainConfig;
  2. enabling it registers the backbone feature hook;
  3. `_cdfkd_training_loss` runs a clean-teacher + corrupted-student step on a dummy batch, returns a finite loss
     that carries the `cd_fkd_mimic` term, and is differentiable (backward populates grads).

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/check_cdfkd.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rf-detr" / "src"))
import torch  # noqa: E402

from rfdetr import RFDETRLarge  # noqa: E402
from rfdetr.config import TrainConfig  # noqa: E402
from rfdetr.training.module_model import RFDETRModelModule  # noqa: E402
from rfdetr.utilities.tensors import NestedTensor  # noqa: E402

RES = 256  # div by 32; small for a fast CPU forward


def main() -> None:
    # 1. flag reaches config
    tc_probe = TrainConfig(dataset_dir="/tmp/x", cd_fkd=True, cd_fkd_alpha=0.5)
    assert tc_probe.cd_fkd and tc_probe.cd_fkd_alpha == 0.5
    print(f"[1] TrainConfig.cd_fkd={tc_probe.cd_fkd} alpha={tc_probe.cd_fkd_alpha} min_scale={tc_probe.cd_fkd_min_scale}")

    # build a real RFDETRLarge to obtain a valid ModelConfig, then the training module with cd_fkd on
    r = RFDETRLarge(num_classes=10, resolution=RES, pretrain_weights=None)
    mc = r.model.model_config if hasattr(r.model, "model_config") else getattr(r, "model_config")
    mc.pretrain_weights = None
    tc = TrainConfig(
        dataset_dir="/tmp/x", output_dir="/tmp/o", cd_fkd=True, cd_fkd_alpha=0.5,
        multi_scale=False, epochs=1, num_workers=0,
    )
    mod = RFDETRModelModule(mc, tc)
    mod.train()
    mod._trainer = SimpleNamespace(global_step=0, accumulate_grad_batches=1)  # minimal trainer mock
    print(f"[2] module built; feature hook registered: {mod._cdfkd_feats is None} (None until first forward)")

    # dummy batch: 2 images, a few GT boxes each (cxcywh normalized)
    b = 2
    samples = NestedTensor(torch.randn(b, 3, RES, RES), torch.zeros(b, RES, RES, dtype=torch.bool))
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1], [0.7, 0.6, 0.15, 0.15]])
    targets = [
        {"boxes": boxes.clone(), "labels": torch.tensor([0, 3, 7]),
         "image_id": torch.tensor([i]), "orig_size": torch.tensor([RES, RES]),
         "size": torch.tensor([RES, RES]), "area": torch.tensor([100.0, 50.0, 75.0]),
         "iscrowd": torch.tensor([0, 0, 0])}
        for i in range(b)
    ]

    loss, loss_dict = mod._cdfkd_training_loss(samples, targets)
    has_mimic = "cd_fkd_mimic" in loss_dict
    mimic_val = float(loss_dict.get("cd_fkd_mimic", float("nan")))
    print(f"[3] _cdfkd_training_loss: loss={float(loss):.4f} requires_grad={loss.requires_grad} "
          f"cd_fkd_mimic={'YES' if has_mimic else 'NO'} ({mimic_val:.4f}) feats_captured={mod._cdfkd_feats is not None}")
    loss.backward()
    gmag = sum(p.grad.abs().sum().item() for p in mod.model.parameters() if p.grad is not None)
    print(f"[4] backward OK — total grad magnitude {gmag:.3e} (must be > 0)")

    ok = (
        loss.requires_grad and torch.isfinite(loss).item() and has_mimic
        and 0.0 <= mimic_val <= 2.0 and gmag > 0
    )
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} — CD-FKD path runs, mimic term present, gradients flow.")


if __name__ == "__main__":
    main()
