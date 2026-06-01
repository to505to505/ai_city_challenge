# mmdet 3.x config — Cascade R-CNN + ConvNeXt-Tiny (FPN) for eccv-cross-city.
#
# Translated from the BDD100K 2.x config embedded in the original checkpoint
# (cascade_rcnn_convnext-s_fpn_fp16_3x_det_bdd100k.pth) and adapted for:
#   * 10 eccv-cross-city classes (a different SET from BDD's 10, not a different count)
#   * resolution-preserving multiscale input: cap long edge 1920, jitter short edge, pad /32
#   * backbone trains slower than neck+head (backbone lr_mult)
#   * basic augmentations: h-flip + photometric (brightness/contrast + colorjitter) + mild affine
#   * fp16 (AMP) and load_from the bundled BDD-pretrained detector
#
# scripts/train.py OVERRIDES a few fields at runtime (it alone knows the exported COCO dir
# and the Hafnia checkpoint dir): data_root + ann_file/data_prefix of every split, work_dir,
# load_from, train_cfg.max_epochs, the dataloader batch_size, base lr and backbone_lr_mult.
# Everything here is a sensible standalone default so the file also runs via mmdet's tools/train.py.

# ConvNeXt backbone lives in mmpretrain — must be imported so the registry knows 'mmpretrain.ConvNeXt'.
custom_imports = dict(imports=["mmpretrain.models"], allow_failed_imports=False)

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
base_lr = 1e-4           # neck + rpn + roi-head learning rate ("normal")
backbone_lr_mult = 0.1   # backbone learns 10x slower  ->  effective 1e-5 ("slow")
max_epochs = 24
batch_size = 2           # per GPU; tuned for a single T4 16 GB at ~1920 long edge (see README)

# ImageNet normalization, exactly as the original BDD config. The /32 padding that the FPN
# requires is applied HERE, at batch collation (not in the pipeline) — see README "how we pad".
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

# ============================ MODEL ============================
model = dict(
    type="CascadeRCNN",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="mmpretrain.ConvNeXt",
        arch="tiny",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        # No init_cfg / pretrained download: the FULL detector (backbone+neck+heads) is loaded
        # from the bundled checkpoint via `load_from` (the runtime is network-isolated).
    ),
    neck=dict(type="FPN", in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
    ),
    roi_head=dict(
        type="CascadeRoIHead",
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ],
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=[
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
        ],
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)

# ============================ DATA ============================
dataset_type = "CocoDataset"
# Placeholder — train.py rewrites data_root + every split's ann_file/data_prefix to the actual
# COCO export it materializes from the Hafnia dataset.
data_root = ".data/coco/eccv-cross-city/"
backend_args = None

# Multiscale: cap the LONG edge at 1920, jitter the SHORT edge -> scale augmentation that never
# upsizes past 1920 and keeps native aspect ratio (keep_ratio=True). See README for worked examples.
train_scales = [(1920, 896), (1920, 960), (1920, 1024), (1920, 1080)]
test_scale = (1920, 1080)

# Basic augmentations (same intent as the previous RF-DETR trainer), all mmdet-NATIVE so there is no
# third-party version coupling. (The earlier Albu approach crashed at runtime: mmdet 3.3's Albu wrapper
# forwards non-image keys like img_path to albumentations, which >=1.4 rejects with
# "Key img_path is not in available keys".)
#   * PhotoMetricDistortion -> mild brightness/contrast/saturation/hue jitter (operates on img only).
#   * RandomAffine -> mild geometric (scale/translate/rotate); boxes transformed correctly by mmdet.
train_pipeline = [
    # to_float32=True is REQUIRED: PhotoMetricDistortion asserts a float32 image (and the original
    # BDD config loaded float32 too). RandomChoiceResize / RandomAffine / cv2 all handle float32.
    dict(type="LoadImageFromFile", to_float32=True, backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomChoiceResize", scales=train_scales, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    # Mild photometric jitter, ~+/-15% like the old brightness/contrast/colorjitter
    # (brightness_delta 32/255 ~= 0.125; contrast/saturation +/-15%; hue_delta 18 ~= the old hue 0.05).
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.85, 1.15),
        saturation_range=(0.85, 1.15),
        hue_delta=18,
    ),
    # Mild affine (scale 0.9-1.1, translate +/-5%, rotate +/-5deg). border=0 keeps canvas size;
    # boxes outside the frame are clipped/filtered by mmdet.
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
    # LoadAnnotations BEFORE Resize so gt_bboxes are rescaled with the image (idiomatic; keeps GT
    # correct for DetVisualizationHook). CocoMetric reads GT from the ann_file regardless.
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
    # Group same-orientation images into a batch so landscape+portrait don't waste padding.
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img="train/"),
        # Some frames legitimately have 0 objects — keep them (don't filter empties).
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

# ============================ SCHEDULE / OPTIM ============================
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# fp16 mixed precision via AMP. Backbone gets lr_mult (slow); neck+heads use base_lr (normal).
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={"backbone": dict(lr_mult=backbone_lr_mult)},
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_grad=None,
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

# AdamW is the optimizer; auto_scale_lr lets mmdet scale lr if effective batch differs from base.
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
    # draw=False: don't write per-val-image PNGs into work_dir (== /opt/ml/checkpoints in cloud),
    # which the platform would then collect as artifacts.
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

# Injected by train.py: absolute path to the bundled slimmed BDD checkpoint.
load_from = None
resume = False
