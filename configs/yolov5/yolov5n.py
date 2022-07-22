dep_mul = 0.33
wid_mul = 0.25

anchors = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]

img_size = (640, 640)
dataset_type = "YOLODetectionDataset"
img_path = "/d/dataset/helmet/VOC2028/images"
batch_size = 8
normal_batch_size = 64
accumulate = max(round(normal_batch_size / batch_size), 1)

# model settings
model = dict(
    type="YOLOV5",
    backbone=dict(type="CSPDarknet", dep_mul=dep_mul, wid_mul=wid_mul),
    neck=dict(
        type="YOLOPAFPN",
        in_channels=[int(256 * wid_mul), int(512 * wid_mul), int(1024 * wid_mul)],
        num_csp_blocks=round(3 * dep_mul),
    ),
    head=dict(
        type="YOLOV5Head",
        num_classes=80,
        anchors=anchors,
        strides=[8, 16, 32],
        in_channels=[int(256 * wid_mul), int(512 * wid_mul), int(1024 * wid_mul)],
    ),
)

train_pipeline = [
    dict(type="Mosaic", img_scale=img_size, pad_value=114.0),
    dict(
        type="RandomPerspective",
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=10,
        perspective=0.0,
        area_thr=0.2,
    ),
    dict(type="RandomHSV", hgain=0.5, sgain=0.5, vgain=0.5),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="Resize", img_scale=img_size, keep_ratio=True),
    dict(
        type="Pad",
        size=img_size,
        center_pad=True,
        pad_value=114,
    ),
    dict(type="FilterAnnotations", wh_thr=2, ar_thr=20),
    # dict(type="FormatBundle", bbox_type="cxcywh", coord_norm=True),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        img_path=img_path,
        prefix="train:",
        pipeline=[
            dict(type="LoadImageFromFile", to_float32=False),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
    ),
    pipeline=train_pipeline,
)

scale_w = batch_size * accumulate / normal_batch_size  # scale weight_decay
optimizer = dict(
    type="SGD",
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4 * scale_w,
    nesterov=True,
)
