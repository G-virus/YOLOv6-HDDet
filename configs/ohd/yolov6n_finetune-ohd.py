# YOLOv6s model
model = dict(
    type="YOLOv6n",
    pretrained="weights/yolov6n.pt",#weights/yolov6n.pt
    # /media/qd1205/8t/train_tb/HRSC_200e_n_MDD_6,6_1-coshb_0.08aw/weights/last_ckpt.pt
    depth_multiple=0.33,
    width_multiple=0.25,
    backbone=dict(
        type="EfficientRep",
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,
        cspsppf=True,
    ),
    neck=dict(
        type="RepBiFPANNeck",
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
    ),
    head=dict(
        type="EffiDeHead",
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=3,
        anchors_init=[[10, 13, 19, 19, 33, 23], [30, 61, 59, 59, 59, 119], [116, 90, 185, 185, 373, 326]],
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type="giou",
        use_dfl=False,  # set to True if you want to further train with distillation
        reg_max=0,  # set to 16 if you want to further train with distillation
        distill_weight={
            "class": 1.0,
            "dfl": 1.0,
            "angle": 1.0
        },
        # NOTE for angle regression
        # angle_fitting_methods='regression',
        # angle_max=1,
        # NOTE for angle fullcsl
        angle_fitting_methods="csl",
        angle_max=360,
        # NOTE for angle dfl
        # angle_fitting_methods='dfl',
        # angle_max=180,
        # NOTE for angle MGAR
        # angle_fitting_methods='MGAR',
        # angle_max=5,
        # NOTE for angle MDD
        # angle_fitting_methods='MDD',
        # angle_max=(6,6),
    ),
)

loss = dict(
    # loss_mode="hbb+angle",
    loss_mode="hbb+cosiou+angle",
    # loss_mode="ohd+cosiou+angle",
    # loss_mode="ohd+cosiou",
    # loss_mode="ohd+angle+cosiou",
    # NOTE for angle regression
    # loss_weight={"class": 1.0, "iou": 2.0, "dfl": 0.5, "angle": 0.05},
    # NOTE for angle fullcsl
    loss_weight={"class": 1.0, "iou": 2.5, "dfl": 0.5, "angle": 0.5, 'cwd': 0.2},
    # NOTE for angle dfl
    # loss_weight={"class": 1.0, "iou": 2.5, "dfl": 0.5, "angle": 0.25, 'cwd': 10},
    # NOTE for angle MGAR
    # loss_weight={"class": 1.0, "iou": 2.5, "dfl": 0.5, "angle": 0.08, "MGAR_cls": 0.05, "MGAR_reg": 0.05, 'cwd': 0.2, },
    # NOTE for angle MDD
    # loss_weight={"class": 1.0, "iou": 2.5, "dfl": 0.5, "angle": 0.09, 'cwd': 0.2},
)


solver = dict(
    optim="AdamW",
    lr_scheduler="Cosine",
    lr0=0.00025,
    lrf=0.05,
    momentum=0.843,
    weight_decay=0.05,
    warmup_epochs=1.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.05,
)

data_aug = dict(
    # degrees=0.373,
    # translate=0.245,
    # scale=0.898,
    # shear=0.602,
    hsv=0.0,
    hsv_h=0.0138,
    hsv_s=0.664,
    hsv_v=0.464,
    flipud=0.5,
    fliplr=0.5,
    # rotate=0.0,
    # rect_classes=[9, 11],
    rect_classes=None,
    # NOTE mosaic 数值需要确定一下
    mosaic=0.0,
    mixup_mosaic=0.0,
    mixup=0.3,

    degrees=360.,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.000
)


eval_params = dict(
    conf_thres=0.03,
    verbose=True,
    do_coco_metric=False,
    do_pr_metric=True,
    plot_curve=False,
    plot_confusion_matrix=True,
    # NOTE VOC12 VOC07 COCO
    # NOTE DOTA use COCO
    # NOTE HRSC use VOC07 or VOC12
    # NOTE DIOR use VOC07
    ap_method="COCO",
)
