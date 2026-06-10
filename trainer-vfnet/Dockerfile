# Cascade R-CNN + ConvNeXt-Tiny (mmdet 3.x) trainer for eccv-cross-city — Hafnia Training-aaS.
#
# IMPORTANT (Hafnia): the platform bootstrap cd's into /opt/recipe before running the user command,
# so WORKDIR must be /opt/recipe. The training runtime is network-isolated — all pip installs happen
# HERE at build time, and the pretrained checkpoint is baked in via `COPY weights`.
#
# Version matrix (do not bump blindly — these pin each other):
#   torch 2.1.2 / cu121  <- base image
#   mmcv 2.1.0           <- prebuilt wheel exists for torch2.1/cu121 (no compiler needed at build)
#   mmdet 3.3.0          <- requires mmcv >=2.0.0rc4,<2.2.0  (2.1.0 fits)
#   mmpretrain 1.2.0     <- provides mmpretrain.ConvNeXt backbone
#   mmengine 0.10.4
# mmcv >=2.2 / torch >=2.2 would break mmdet 3.3.0's mmcv ceiling, so we stay on torch 2.1.

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRAINER_PACKAGE_DIR=/opt/recipe

WORKDIR $TRAINER_PACKAGE_DIR

# System deps for opencv (cv2) / mmcv / pycocotools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. Hafnia SDK + non-mm runtime deps (NumPy pinned <2 — torch 2.1 / mmcv 2.1 predate NumPy 2).
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r ./requirements.txt

# 2. OpenMMLab stack. mmcv is pinned to its EXACT prebuilt wheel index so an index miss errors out
#    instead of silently source-building (this -runtime image has no nvcc and would fail). mim then
#    installs the rest (mmcv already satisfied).
RUN pip install -U openmim && \
    pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html && \
    mim install "mmengine==0.10.4" "mmdet==3.3.0" "mmpretrain==1.2.0"

# 2b. CRITICAL: openmim/mmdet installs drag in NumPy 2.x, but torch 2.1.2 is built against NumPy 1.x,
#     so at runtime `tensor.numpy()` dies with "RuntimeError: Numpy is not available". Force NumPy
#     back to the 1.x line as the LAST dependency step so nothing can re-upgrade it.
RUN pip install "numpy<2" && \
    python -c "import numpy, torch; assert numpy.__version__[0]=='1', numpy.__version__; print('numpy', numpy.__version__, '| tensor.numpy() ok:', torch.zeros(1).numpy())"

# 3. Project sources (relative to /opt/recipe).
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
# Bundled slimmed BDD100K checkpoint (~292 MB) — network isolation means no runtime download.
COPY weights ./weights

ENV PYTHONPATH=${TRAINER_PACKAGE_DIR}/src

# Hafnia overrides CMD with the experiment's --cmd; this default is for local docker runs.
CMD ["python", "scripts/train.py", "--epochs", "1"]
