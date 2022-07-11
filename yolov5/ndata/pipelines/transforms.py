from ..builder import PIPELINES
from yolov5.utils.segment import segment2box
from yolov5.core import Bboxes
from typing import Optional
import random
import cv2
import math
from lqcv.image import imrescale
from lqcv.utils import colorstr, to_3tuple
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
        center_ratio_range=(0.5, 1.5),
        pad_value=114,
        neg_dir: Optional[str] = None,
        prob=1.0,
    ) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, (
            "The probability should be in range [0,1]. " f"got {prob}."
        )
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_value = pad_value
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
        assert results.get("rect_shape", None) is None, "rect and mosaic is exclusive."
        assert (
            len(results.get("mix_results", [])) > 0
        ), "There are no other images for mosaic augment."
        mosaic_img = np.full(
            (self.img_scale[0] * 2, self.img_scale[1] * 2, results["ori_shape"][2]),
            self.pad_value,
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
                    img,
                    new_wh=(self.img_scale[1], self.img_scale[0]),
                    keep_ratio=True,
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
            # rescaled image
            results_patch["img"] = img
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
                results_patch = self._update_results(results_patch, padw, padh)
                mosaic_results.append(results_patch)
        final_results = self._cat_results(mosaic_results)
        final_results["img"] = mosaic_img
        return final_results

    def _update_results(self, results, padw, padh):
        """Update labels"""
        bboxes = results["gt_bboxes"].convert(format="xyxy")
        segments = results.get("gt_segments", None)
        keypoints = results.get("gt_keypoints", None)

        nh, nw = results["img"].shape[:2]
        if results["norm"]:
            scale_x, scale_y = nw, nh
            results["norm"] = False
        else:
            h, w = results["ori_shape"][:2]
            scale_x, scale_y = nw / w, nh / h

        bboxes.mul(scale=(scale_x, scale_y, scale_x, scale_y))
        if segments is not None:
            # (num_samples, 500, 2)
            segments[..., 0] *= scale_x
            segments[..., 1] *= scale_y
        if keypoints is not None:
            # (num_samples, nl, 2)
            keypoints[..., 0] *= scale_x
            keypoints[..., 1] *= scale_y

        bboxes.add(offset=(padw, padh, padw, padh))
        if segments is not None:
            # (num_samples, 500, 2)
            segments[..., 0] += padw
            segments[..., 1] += padh
            results["gt_segments"] = segments
        if keypoints is not None:
            # (num_samples, nl, 2)
            keypoints[..., 0] += padw
            keypoints[..., 1] += padh
            results["gt_keypoints"] = keypoints
        results["gt_bboxes"] = bboxes
        return results

    def _cat_results(self, mosaic_results):
        if len(mosaic_results) == 0:
            return dict()
        final_results = dict()
        final_results["ori_shape"] = (
            self.img_scale[0] * 2,
            self.img_scale[1] * 2,
            results["ori_shape"][2],
        )
        final_results["img_file"] = "mosaic"
        gt_labels = []
        gt_bboxes = []
        gt_segments = []
        gt_keypoints = []
        for results in mosaic_results:
            gt_labels.append(results["gt_labels"])
            gt_bboxes.append(results["gt_bboxes"])
            gt_segments.append(results.get("gt_segments"), None)
            gt_keypoints.append(results.get("gt_keypoints"), None)

        final_results["gt_labels"] = np.concatenate(gt_labels, 0)
        final_results["gt_bboxes"] = Bboxes.cat(gt_bboxes, 0)
        if all([gt_segment is not None for gt_segment in gt_segments]):
            final_results["gt_segments"] = np.concatenate(gt_segments, 0)
        if all([gt_keypoints is not None for gt_keypoints in gt_keypoints]):
            final_results["gt_keypoints"] = np.concatenate(gt_keypoints, 0)
        return final_results

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
class RandomPerspective:
    """
    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
    """

    def __init__(
        self,
        img_scale=(640, 640),
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=10,
        perspective=0.0,
        area_thr=0.2,
    ) -> None:
        self.img_scale = img_scale
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        # mosaic border
        self.area_thr = area_thr

    def _get_affine_matrix(self, img):
        # Center
        C = np.eye(3)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(
            -self.perspective, self.perspective
        )  # x perspective (about y)
        P[2, 1] = random.uniform(
            -self.perspective, self.perspective
        )  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # x shear (deg)
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate)
            * self.img_scale[1]
        )  # x translation (pixels)
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate)
            * self.img_scale[0]
        )  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        return M, s

    def apply_bboxes(self, bboxes, M):
        """apply affine to bboxes only.

        Args:
            bboxes(ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M(ndarray): affine matrix.
        Returns:
            new_bboxes(ndarray): bboxes after affine, [num_bboxes, 4].
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3))
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(
            n, 8
        )  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_bboxes = (
            np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        )

        # clip
        new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, self.img_scale[0])
        new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, self.img_scale[1])
        return new_bboxes

    def apply_segments(self, segments, M):
        """apply affine to segments and generate new bboxes from segments.

        Args:
            segments(ndarray): list of segments, [num_samples, 500, 2].
            M(ndarray): affine matrix.
        Returns:
            new_segments(ndarray): list of segments after affine, [num_samples, 500, 2].
            new_bboxes(ndarray): bboxes after affine, [N, 4].
        """
        n = len(segments)
        if n == 0:
            return [], segments
        new_bboxes = np.zeros((n, 4))
        new_segments = []
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T  # transform
            xy = (
                xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]
            )  # perspective rescale or affine

            # clip
            xy[:, 0] = xy[:, 0].clip(0, self.img_scale[1])
            xy[:, 1] = xy[:, 1].clip(0, self.img_scale[0])
            new_bboxes[i] = segment2box(xy, self.img_scale[1], self.img_scale[0])
            new_segments.append(xy)
        new_segments = (
            np.stack(new_segments, axis=0)
            if len(new_segments) > 1
            else new_segments[0][None, :]
        )
        return new_bboxes, new_segments

    def apply_keypoints(self, keypoints, M):
        """apply affine to keypoints.

        Args:
            keypoints(ndarray): keypoints, (num_samples, num_points, 2), [x, y, x, y...].
            M(ndarray): affine matrix.
        Return:
            new_keypoints(ndarray): keypoints after affine, (num_samples, num_points, 2),
                [x, y, x, y...].
        """
        n = len(keypoints)
        if n == 0:
            return keypoints
        new_keypoints = []
        for keypoint in keypoints:
            xy = np.ones((len(keypoint), 3))
            xy[:, :2] = keypoint
            xy = xy @ M.T  # transform
            xy = (
                xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]
            )  # perspective rescale or affine

            # clip
            # NOTE: abandon the coords beyond width or height
            xy = np.where(xy < 0, -1, xy)
            xy[:, 0] = np.where(xy[:, 0] > self.img_scale[1], -1, xy[:, 0])
            xy[:, 1] = np.where(xy[:, 1] > self.img_scale[0], -1, xy[:, 1])
            xy[:, 0] = np.where(xy[:, 1] == -1, -1, xy[:, 0])
            xy[:, 1] = np.where(xy[:, 0] == -1, -1, xy[:, 1])
            # xy[:, 0] = xy[:, 0].clip(0, width)
            # xy[:, 1] = xy[:, 1].clip(0, height)
            new_keypoints.append(xy)
        new_keypoints = (
            np.stack(new_keypoints, axis=0)
            if len(new_keypoints) > 1
            else new_keypoints[0][None, :]
        )
        return new_keypoints

    def __call__(self, results):
        """
        Affine images and targets.

        Args:
            img(ndarray): image.
            results(Dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        assert results.get("rect_shape", None) is None, "rect and affine is exclusive."

        img = results["img"]
        # M is affine matrix
        # scale for func:`box_candidates`
        M, scale = self._get_affine_matrix(img)

        labels = results["gt_labels"]
        bboxes = results["gt_bboxes"].convert(format="xyxy")
        segments = results.get("gt_segments", None)
        keypoints = results.get("gt_keypoints", None)

        new_bboxes = self.apply_bboxes(bboxes.bboxes)
        # update bboxes if there are segments.
        if segments is not None:
            new_bboxes, new_segments = self.apply_segments(segments)

        new_bboxes = Bboxes(new_bboxes, format="xyxy")
        i = self._box_candidates(
            ori_bboxes=bboxes.mul(scale),
            affined_bboxes=new_bboxes,
            labels=labels,
        )

        if segments is not None:
            new_segments = new_segments[i]
            results["gt_segments"] = new_segments

        if keypoints is not None:
            new_keypoints = self.apply_keypoints(keypoints)
            new_keypoints = new_keypoints[i]
            results["gt_keypoints"] = new_keypoints

        labels = labels[i]
        results["gt_labels"] = labels
        bboxes = new_bboxes[i]
        results["gt_bboxes"] = bboxes

        # affine image
        if (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(
                    img,
                    M,
                    dsize=(self.img_scale[1], self.img_scale[0]),
                    borderValue=(114, 114, 114),
                )
            else:  # affine
                img = cv2.warpAffine(
                    img,
                    M[:2],
                    dsize=(self.img_scale[1], self.img_scale[0]),
                    borderValue=(114, 114, 114),
                )
        results["img"] = img
        return results

    def _filter_area_threshold(self, ori_bboxes, affined_bboxes, labels, eps=1e-16):
        """filter the results after affine, if affined_bboxes / ori_bboxes

        Args:
            ori_bboxes (Bboxes): bboxes before affine.
            affined_bboxes (Bboxes): bboxes after affine.
            labels (ndarray): labels.
        """
        area_thr = (
            np.array(self.area_thr)[labels.astype(np.int)]
            if isinstance(self.area_thr, list)
            else self.area_thr
        )
        if isinstance(self.area_thr, list) and len(self.area_thr) == 1:
            area_thr = self.area_thr[0]
        return affined_bboxes.areas() / (ori_bboxes.areas() + eps) > area_thr


@PIPELINES.register()
class RandomHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, results):
        img = results["img"]
        if self.hgain or self.sgain or self.vgain:
            r = (
                np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            )  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            )
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        results["img"] = img
        return results


@PIPELINES.register()
class RandomFlip:
    def __init__(self, prob=None, direction="horizontal") -> None:
        assert direction in [
            "horizontal",
            "vertical",
        ], f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= prob <= 1.0

        self.prob = prob
        self.direction = direction

    def __call__(self, results):
        img = results["img"]
        bboxes = results["gt_bboxes"].convert(format="xyxy").bboxes
        segments = results.get("gt_segments", None)
        keypoints = results.get("gt_keypoints", None)
        # Flip up-down
        if self.direction == "vertical" and random.random() < self.prob:
            img = np.flipud(img)
            if len(bboxes):
                bboxes[:, 1] = 1 - bboxes[:, 1]
            if segments is not None:
                segments[..., 1] = 1 - segments[..., 1]
                results["segments"] = segments
            if keypoints is not None:
                keypoints[..., 1] = 1 - keypoints[..., 1]
                results["keypoints"] = keypoints
        if self.direction == "horizontal" and random.random() < self.prob:
            img = np.fliplr(img)
            if len(bboxes):
                bboxes[:, 0] = 1 - bboxes[:, 0]
            if segments is not None:
                results["segments"] = segments
                segments[..., 0] = 1 - segments[..., 0]
            if keypoints is not None:
                results["keypoints"] = keypoints
                keypoints[..., 0] = 1 - keypoints[..., 0]
        results["gt_bboxes"] = Bboxes(bboxes, format="xyxy")
        results["img"] = img
        return results


@PIPELINES.register()
class Resize:
    """Resize image

    Args:
        img_scale (tuple): target image height and width.
        keep_ratio (bool): Whether to keep the aspect ratio.
        auto (bool): auto get the rect image size.
    """

    def __init__(self, img_scale=(640, 640), keep_ratio=True) -> None:
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        if h == self.img_scale[0] and w == self.img_scale[1]:
            return results

        img = imrescale(
            new_wh=(self.img_scale[1], self.img_scale[0]), keep_ratio=self.keep_ratio
        )
        results["img"] = img
        results = self._update_results(results)
        return results

    def _update_results(self, results):
        """Update labels"""
        bboxes = results["gt_bboxes"].convert(format="xyxy")
        segments = results.get("gt_segments", None)
        keypoints = results.get("gt_keypoints", None)

        nh, nw = results["img"].shape[:2]
        if results["norm"]:
            scale_x, scale_y = nw, nh
            results["norm"] = False
        else:
            h, w = results["ori_shape"][:2]
            scale_x, scale_y = nw / w, nh / h

        bboxes.mul(scale=(scale_x, scale_y, scale_x, scale_y))
        if segments is not None:
            # (num_samples, 500, 2)
            segments[..., 0] *= scale_x
            segments[..., 1] *= scale_y
            results["gt_segments"] = segments
        if keypoints is not None:
            # (num_samples, nl, 2)
            keypoints[..., 0] *= scale_x
            keypoints[..., 1] *= scale_y
            results["gt_keypoints"] = keypoints

        results["gt_bboxes"] = bboxes
        return results


@PIPELINES.register()
class Pad:
    """Pad image

    Args:
        size (tuple): target image height and width.
        pad_value (int): padding value.
        center_pad (bool): center padding if True, otherwise padding
            in right bottom.
        auto (bool): auto get the rect image size.
        stride (int): stride.
    """

    def __init__(
        self,
        size=(640, 640),
        pad_value=114,
        center_pad=True,
        auto=False,
        stride=32,
    ) -> None:
        self.size = size
        self.center_pad = center_pad
        self.auto = auto
        self.stride = stride
        self.pad_value = (
            to_3tuple(pad_value) if isinstance(pad_value, int) else pad_value
        )

    def __call__(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        if h == self.size[0] and w == self.size[1]:
            return results

        if results.get("rect_shape", None) is not None:
            nh, nw = results["rect_shape"][:2]
            dw, dh = nw - w, nh - h  # wh padding
        else:
            dw, dh = self.size[1] - w, self.size[0] - h  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        if self.center_pad:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        top, bottom = int(round(dh - 0.1)) if self.center_pad else 0, int(
            round(dh + 0.1)
        )
        left, right = int(round(dw - 0.1)) if self.center_pad else 0, int(
            round(dw + 0.1)
        )
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_value
        )  # add border
        results = self._update_results(results, padw=left, padh=top)
        results["img"] = img
        return results

    def _update_results(self, results, padw, padh):
        bboxes = results["gt_bboxes"].convert(format="xyxy")
        segments = results.get("gt_segments", None)
        keypoints = results.get("gt_keypoints", None)

        bboxes.add(offset=(padw, padh, padw, padh))
        if segments is not None:
            # (num_samples, 500, 2)
            segments[..., 0] += padw
            segments[..., 1] += padh
            results["gt_segments"] = segments
        if keypoints is not None:
            # (num_samples, nl, 2)
            keypoints[..., 0] += padw
            keypoints[..., 1] += padh
            results["gt_keypoints"] = keypoints
        results["gt_bboxes"] = bboxes
        return results


@PIPELINES.register()
class CopyPaste:
    def __init__(self) -> None:
        pass
