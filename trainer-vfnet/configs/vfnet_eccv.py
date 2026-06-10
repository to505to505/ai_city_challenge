# mmdet 3.x config — VFNet (VarifocalNet) R-50 FPN for eccv-cross-city.
#
# A dense, anchor-FREE detector (FCOS+ATSS core, IoU-aware Varifocal Loss, star-shaped deformable
# box refinement). Chosen as an ENSEMBLE member: its localization (point-distance regression) and its
# IoU-aware confidence calibration decorrelate from RF-DETR (query head) and from Cascade/ConvNeXt
# (two-stage anchor head), which is what WBF rewards.
#
# Structure mirrors cascade_convnext_eccv.py EXACTLY except the `model` block, so the same
# scripts/train.py overrides apply (data_root/ann_file, work_dir, load_from, max_epochs, batch_size,
# optim_wrapper.optimizer.lr, paramwise_cfg.custom_keys.backbone.lr_mult, param_scheduler[0]/[1]).
#
# load_from = a COCO-pretrained vfnet_r50_fpn_ms-2x checkpoint (44.8 box AP), slimmed + bundled into
# the image (runtime is network-isolated). Its 80-class `bbox_head.vfnet_cls` is skipped on load
# (size mismatch 80->10) and trained from scratch; everything else (backbone/neck/head towers/star-dcn)
# is reused.

default_scope = "mmdet"

# ---- eccv-cross-city v1.0.0 classes, in dataset.info order (index == class_idx in annotations) ----
classes = (
    "Vehicle.Car",
    "Vehicle.Pickup Truck",
    "Vehicle.Single Truck",
    "Vehicle.Combo Truck",
    "Vehicle.Heavy Duty Vehicle",
    "Vehicle.Trailer",
    "Vehicle.Motorcycle",
    "Vehicle.Bicycle",
    "Vehicle.Van",
    "Person",
)
num_classes = len(classes)

# ---- training knobs (train.py may override base_lr / backbone_lr_mult / max_epochs / batch_size) ----
base_lr = 1e-4           # neck + head learning rate ("normal")
backbone_lr_mult = 0.1   # backbone (ResNet-50) learns 10x slower -> effective 1e-5
max_epochs = 12
batch_size = 2           # per GPU; VFNet R-50 is lighter than Cascade, fits a single T4 16 GB @1920 long edge

data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

# ============================ MODEL (VFNet R-50 FPN) ============================
model = dict(
    type="VFNet",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=None,  # full detector loaded via `load_from`; no torchvision download (network-isolated)
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",  # P6, P7 from P5 (VFNet uses strides 8..128)
        num_outs=5,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type="VFNetHead",
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,  # backbone-tower DCN off (matches the r50_fpn ms-2x checkpoint).
        use_atss=True,           # the star-shaped DCN in the HEAD is built-in either way.
        use_vfl=True,
        loss_cls=dict(
            type="VarifocalLoss",
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.5),
        loss_bbox_refine=dict(type="GIoULoss", loss_weight=2.0),
    ),
    train_cfg=dict(
        assigner=dict(type="ATSSAssigner", topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,  # low -> keep recall for WBF ensembling
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

# ============================ DATA (identical to the ConvNeXt trainer) ============================
dataset_type = "CocoDataset"
data_root = ".data/coco/eccv-cross-city/"   # train.py rewrites this + every split's ann/prefix
backend_args = None

train_scales = [(1920, 896), (1920, 960), (1920, 1024), (1920, 1080)]
test_scale = (1920, 1080)

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True, backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomChoiceResize", scales=train_scales, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.85, 1.15),
        saturation_range=(0.85, 1.15),
        hue_delta=18,
    ),
    dict(
        type="RandomAffine",
        max_rotate_degree=5.0,
        max_translate_ratio=0.05,
        scaling_ratio_range=(0.9, 1.1),
        max_shear_degree=0.0,
        border=(0, 0),
        border_val=(114, 114, 114),
    ),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=test_scale, keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img="train/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(img="valid/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "valid/_annotations.coco.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# ============================ SCHEDULE / OPTIM (same shape as ConvNeXt) ============================
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# fp32 (NOT AmpOptimWrapper): VFNet's head (VarifocalLoss / star-deformable indexed writes) is not
# AMP-safe in mmdet 3.3 — fp16 autocast triggers "Index put requires source and destination dtypes
# match (Half vs Float)" at the first step. The mmdet VFNet model-zoo configs all run fp32. fp32 ~2x
# the activation memory of AMP, so pair with batch_size 2 @1920 (~10 GB, see docs/gpu_memory.md).
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={"backbone": dict(lr_mult=backbone_lr_mult)},
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),  # VFNet benefits from grad clipping
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        by_epoch=True,
        begin=0,
        end=max_epochs,
        milestones=[max_epochs - 8, max_epochs - 2],
        gamma=0.1,
    ),
]

auto_scale_lr = dict(enable=False, base_batch_size=16)

# ============================ RUNTIME ============================
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook", draw=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"

load_from = None  # injected by train.py -> the bundled slimmed VFNet COCO checkpoint
resume = False
