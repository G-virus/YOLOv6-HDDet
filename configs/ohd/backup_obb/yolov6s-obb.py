# YOLOv6s model
model = dict(
    type="YOLOv6s",
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.50,
    backbone=dict(
        type="EfficientRep",
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,
        cspsppf=True,
    ),
    neck=dict(type="RepBiFPANNeck", num_repeats=[12, 12, 12, 12], out_channels=[256, 128, 128, 256, 256, 512],),
    head=dict(
        type="EffiDeHead",
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=3,
        # REVIEW 这个anchors init 是为了什么?
        anchors_init=[[10, 13, 19, 19, 33, 23], [30, 61, 59, 59, 59, 119], [116, 90, 185, 185, 373, 326]],
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        # NOTE 可以选择, 包括 giou/ciou/siou
        iou_type="giou",
        use_dfl=False,  # set to True if you want to further train with distillation
        reg_max=0,  # set to 16 if you want to further train with distillation
        distill_weight={"class": 1.0, "dfl": 1.0, },
        angle_max=1,
        angle_fitting_methods='regression',
    ),
)

solver = dict(
    optim="SGD",
    lr_scheduler="Cosine",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    # NOTE translate 可能对obb有bug
    # translate=0.1,
    translate=0.0,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    # NOTE fliplr 有影响
    fliplr=0.0,
    # NOTE mosaic 对obb有bug
    mosaic=0.0,
    # NOTE 可以增加 for ohd
    mixup=0.0,
)
