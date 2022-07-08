from ..builder import PIPELINES
from typing import Optional
import random
import cv2
from lqcv.image import imrescale
from lqcv.utils import colorstr
import logging
import numpy as np
import os
from copy import deepcopy


@PIPELINES.register()
class Mosaic:
    """Mosaic augmentation.
    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
    """

    def __init__(
        self,
        img_scale=(640, 640),
        pad_value=114.0,
        center_ratio_range=(0.5, 1.5),
        neg_dir: Optional[str] = None,
        prob=1.0,
    ) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, (
            "The probability should be in range [0,1]. " f"got {prob}."
        )
        self.img_scale = img_scale
        self.pad_value = pad_value
        self.center_ratio_range = center_ratio_range
        self.prob = prob
        if neg_dir is not None:
            self.img_neg_files = self._get_neg_files(neg_dir)

    def __call__(self, results):
        if random.uniform(0, 1) < self.prob:
            return results
        return self._mosaic_transform(results)

    def get_indexes(self, dataset):
        return [random.randint(0, len(dataset)) for _ in range(3)]

    def _mosaic_transform(self, results):
        mosaic_results = []
        assert (
            results.get("target_shape", None) is None
        ), "rect and mosaic is exclusive."
        assert (
            len(results.get("mix_results", [])) > 0
        ), "There are no other images for mosaic augment."
        mosaic_img = np.full(
            (self.img_scale[0] * 2, self.img_scale[1] * 2, results["ori_shape"][2]),
            114,
            dtype=np.uint8,
        )  # base image with 4 tiles

        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        num_neg = random.randint(0, 2) if len(self.img_neg_files) else 0

        mix_results = results["mix_results"]
        random_indexes = list(range(4))
        random.shuffle(random_indexes)
        for i, index in enumerate(random_indexes):
            # get the images
            if index == 0:
                img = imrescale(
                    results["img"],
                    new_wh=(self.img_scale[1], self.img_scale[0]),
                    keep_ratio=True,
                    return_scale=True,
                )
                results_patch = deepcopy(results)
            elif num_neg > 0:
                neg_index = random.choice(range(len(self.img_neg_files)))
                img = cv2.imread(self.img_neg_files[neg_index])
                img = imrescale(
                    img, new_wh=(self.img_scale[1], self.img_scale[0]), keep_ratio=True,
                )
                results_patch = None
                num_neg -= 1
            else:
                img = imrescale(
                    mix_results[index - 1]["img"],
                    new_wh=(self.img_scale[1], self.img_scale[0]),
                    keep_ratio=True,
                )
                results_patch = deepcopy(mix_results[index - 1])

            h, w = img.shape[:2]
            # put image to the big mosaic_img
            if i == 0:  # top left
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = (
                    max(center_x - w, 0),
                    max(center_y - h, 0),
                    center_x,
                    center_y,
                )
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = (w - (x2a - x1a), h - (y2a - y1a), w, h)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    center_x,
                    max(center_y - h, 0),
                    min(center_x + w, self.img_scale[1] * 2),
                    center_y,
                )
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(center_x - w, 0),
                    center_y,
                    center_x,
                    min(self.img_scale[0] * 2, center_y + h),
                )
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    center_x,
                    center_y,
                    min(center_x + w, self.img_scale[1] * 2),
                    min(self.img_scale[0] * 2, center_y + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # mosaic_img[ymin:ymax, xmin:xmax]
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if results_patch is not None:
                results_patch = self._update_results(results_patch, w, h, padw, padh)
                mosaic_results.append(results_patch)

    def _update_results(self, results, img_w, img_h, padw, padh):
        """Update labels"""
        bboxes = results["gt_bboxes"].convert(format="xyxy")
        if results["norm"]:
            bboxes.denormalize(img_w, img_h)
            bboxes.add_offset(offset=(padw, padh, padw, padh))
        else:
            pass

    def _get_neg_files(self, neg_dir):
        """Get negative pictures."""
        img_neg_files = []
        if os.path.isdir(neg_dir):
            img_neg_files = [os.path.join(neg_dir, i) for i in os.listdir(neg_dir)]
            logging.info(
                colorstr("Negative dir: ")
                + f"'{neg_dir}', using {len(img_neg_files)} pictures from the dir as negative samples during training"
            )
        else:
            logging.info(colorstr("Negative dir: ") + f"'{neg_dir}' is not a directory")

        return img_neg_files


@PIPELINES.register()
class FilterAnnotations:
    def __init__(self) -> None:
        pass
