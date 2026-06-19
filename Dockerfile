# RF-DETR Large trainer for eccv-cross-city — runs on Hafnia Training-aaS or locally via runc.
#
# IMPORTANT: the Hafnia platform's bootstrap entrypoint cd's into /opt/recipe before running the
# user command, so our code must live there (not /workspace). See trainer-object-detection for ref.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Multi-scale training varies the input size every step, and the val->train phase change
    # allocates differently-sized blocks; the default caching allocator fragments and then fails a
    # large contiguous request even when GiBs are free-but-reserved (killed every hires ms run at
    # the epoch-0/val boundary). expandable_segments lets the allocator grow segments instead.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRAINER_PACKAGE_DIR=/opt/recipe

WORKDIR $TRAINER_PACKAGE_DIR

# System deps used by torchvision / albumentations / pycocotools.
# build-essential: insurance against pip wheel-drift — when a transitive dep releases without a
# prebuilt wheel for this platform (stringzilla 4.6.x did), pip falls back to building from source
# and dies without a compiler (BUILD_FAILED). Costs ~200 MB image, saves the run.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. Hafnia SDK + everything RF-DETR needs at runtime.
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r ./requirements.txt

# 2. Vendored RF-DETR source — installed editable so we can patch if needed.
COPY rf-detr ./rf-detr
RUN pip install -e "./rf-detr[train]"

# 3. Project sources (relative to TRAINER_PACKAGE_DIR).
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
# Bundled pretrain weights — Hafnia cloud has no outbound network, so RF-DETR cannot fetch them.
COPY weights ./weights

ENV PYTHONPATH=${TRAINER_PACKAGE_DIR}/src:${TRAINER_PACKAGE_DIR}/rf-detr/src

# Hafnia overrides CMD with the `--cmd` value from `experiment create`,
# so the default below is just for local docker runs.
CMD ["python", "scripts/train.py", "--epochs", "5"]
