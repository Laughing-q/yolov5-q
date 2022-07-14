from yolov5.ndata import build_datasets
from lqcv import Config
import cv2
import numpy as np
import pycocotools.mask as maskUtils
from yolov5.data.data_utils import polygon2mask_downsample


def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(np.bool)
    return bitmap_mask


cfg = Config.fromfile("/home/laughing/codes/yolov5-q/configs/yolov5/yolov5n.py")

dataset = build_datasets(cfg.train_dataset)
np.random.seed(0)
color = (255, 255, 0)

# cv2.namedWindow('p', cv2.WINDOW_NORMAL)
for d in dataset:
    # print(d["img_file"])
    bboxes = d["gt_bboxes"].bboxes
    h, w = d["img"].shape[:2]

    # print(d["gt_labels"])
    # print(bboxes)
    img = d["img"]
    for i, b in enumerate(bboxes):
        x1, y1, x2, y2 = b
        # print(x1, y1, x2, y2)
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    cv2.imshow("p", img)
    if cv2.waitKey(0) == ord("q"):
        break
