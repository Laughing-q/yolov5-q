from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolov5.core import Yolov5Evaluator
from yolov5.utils.plots import plot_one_box
from yolov5.utils.general import coco80_to_coco91_class
import pycocotools.mask as mask_util
import json
import cv2
import os
import numpy as np
from pathlib import Path


# rle = mask_util.encode(np.array(img[:, :, None], order="F", dtype="uint8"))[0]
# rle["counts"] = rle["counts"].decode("utf-8")


if __name__ == "__main__":

    # evaluator = Yolov5Evaluator(
    #     data="./data/coco_local.yaml",
    #     conf_thres=0.01,
    #     iou_thres=0.6,
    #     exist_ok=False,
    #     half=True,
    #     mask=True,
    # )
    #
    # evaluator.run(
    #     weights="./runs/coco_s_new3/weights/best.pt", batch_size=8, imgsz=640, save_json=True
    #     # weights="./weights/yolov5s.pt", batch_size=16, imgsz=640, save_json=True
    # )
    #
    anno = COCO('/d/dataset/COCO/annotations/instances_val2017.json')  # init annotations api
    pred = anno.loadRes('./predictions.json')  # init predictions api
    # eval = COCOeval(anno, pred, 'bbox')
    eval = COCOeval(anno, pred)
    # eval.params.imgIds = [int(Path(x).stem) for x in os.listdir('/d/dataset/COCO/images/val2017')]  # image IDs to evaluate
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    # new_predictions = []
    # with open('./runs/val/exp/predictions.json', 'r') as f:
    #     preditions = json.load(f)
    # for p in preditions:
    #     category_id = int(p["category_id"])
    #     p["category_id"] = coco80_to_coco91_class()[category_id]
    #     new_preditions.append(p)
    #
    # with open("predictions.json", "w") as f:
    #     json.dump(new_predictions, f)
    # img_root = '/d/dataset/COCO/images/val2017'
    # print(preditions[0])
    # for p in preditions:
    #     box = p['bbox']
    #     mask = mask_util.decode(p['segmentation'])
    #     x1, y1, w, h = box
    #     img_id = str(p['image_id']).zfill(12)
    #     img_name = f'{img_id}.jpg'
    #     img = cv2.imread(os.path.join(img_root, img_name))
    #     plot_one_box([x1, y1, x1 + w, y1 + h], img)
    #     print(mask.shape, img.shape)
    #     img[mask.astype(bool)] = img[mask.astype(bool)] * 0.5 + np.array([0, 0, 255]) * 0.5
    #     cv2.imshow('p', img)
    #     # cv2.imshow('m', mask * 255)
    #     print(box, p['score'], p['category_id'], img_name)
    #     if cv2.waitKey(0) == ord('q'):
    #         break
    #
