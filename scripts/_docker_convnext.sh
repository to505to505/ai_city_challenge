#!/bin/bash
# Runs inside a linux/amd64 python:3.10 container: install the OpenMMLab stack from PREBUILT CPU
# wheels (no source compile — the whole reason for using linux), then run Dima's ConvNeXt inference.
set -e
echo "[docker] installing system libs for opencv (libGL/glib) ..."
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null 2>&1
echo "[docker] installing torch 2.1 (cpu) ..."
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
pip install -q "numpy<2" "setuptools<81" pillow pycocotools mmengine
echo "[docker] installing mmcv 2.1 from prebuilt cpu/torch2.1 wheel ..."
pip install -q mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
echo "[docker] installing mmdet + mmpretrain ..."
pip install -q mmdet==3.3.0 mmpretrain==1.2.0
python -c "import mmcv,mmdet,mmpretrain,torch; print('[docker] versions torch',torch.__version__,'mmcv',mmcv.__version__,'mmdet',mmdet.__version__)"
echo "[docker] running ConvNeXt inference ..."
python scripts/convnext_infer.py
echo "[docker] DONE"
