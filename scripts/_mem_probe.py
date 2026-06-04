"""Measure RF-DETR-Large activation memory by running a real forward+backward and reading peak RSS.

Per-resolution in its OWN process (ru_maxrss is a monotonic high-water mark, so isolation gives a clean
per-res number). CPU runs fp32, so GPU-with-AMP activation memory ~= half of the measured activation
delta. Static (weights+grad+optimizer) is computed separately and exactly.

    PYTHONPATH=rf-detr/src python scripts/_mem_probe.py <resolution> [batch]
"""
import gc
import platform
import resource
import sys

import torch

res = int(sys.argv[1])
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
UNIT = 1 if platform.system() == "Darwin" else 1024  # ru_maxrss: bytes on macOS, KB on Linux

from rfdetr import RFDETRLarge

torch.set_grad_enabled(True)
m = RFDETRLarge(num_classes=10, resolution=res)
net = m.model.model  # underlying LWDETR nn.Module
net.train()

n_params = sum(p.numel() for p in net.parameters())
gc.collect()
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

x = torch.randn(bs, 3, res, res)
out = net(x)

tensors = []
def collect(o):
    if torch.is_tensor(o):
        if o.is_floating_point() and o.requires_grad:
            tensors.append(o)
    elif isinstance(o, dict):
        [collect(v) for v in o.values()]
    elif isinstance(o, (list, tuple)):
        [collect(v) for v in o]
collect(out)
loss = sum(t.float().sum() for t in tensors)
loss.backward()

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"RESULT res={res} bs={bs} params_M={n_params/1e6:.1f} "
      f"base_MB={base*UNIT/1e6:.0f} peak_MB={peak*UNIT/1e6:.0f} "
      f"activation_MB={(peak-base)*UNIT/1e6:.0f} (fp32 CPU; GPU-AMP activations ~= half)")
