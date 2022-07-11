from ..builder import PIPELINES
from yolov5.utils.segment import resample_segments
from yolov5.core import Bboxes
import cv2
import numpy as np


@PIPELINES.register()
class LoadImageFromFile:
    def __init__(self, to_float32=False) -> None:
        self.to_float32 = to_float32

    def __call__(self, results):
        img_file = results["img_file"]
        img = cv2.imread(img_file)
        if self.to_float32:
            img = img.astype(np.float32)
        results["img"] = img
        return results


@PIPELINES.register()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        with_keypoint (bool): Whether to parse and load the keypoint
            annotation. Default: False.
        denorm (bool): Whether to convert bbox, segmentation and keypoint
            from relative value to absolute value.
            Default: False.
    """

    def __init__(
        self,
        with_label=True,
        with_bbox=True,
        with_seg=False,
        with_keypoint=False,
        denorm=False,
        bbox_type="cxcywh",
    ) -> None:
        assert bbox_type in [
            "cxcywh",
            "xyxy",
        ], f"Support `bbox_type` 'cxcywh' or 'xyxy', but got {bbox_type}"
        self.with_label = with_label
        self.with_bbox = with_bbox
        self.with_seg = with_seg
        self.with_keypoint = with_keypoint
        self.denorm = denorm
        self.bbox_type = bbox_type

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        bbox_type = results.get("bbox_type", None)
        assert bbox_type in [
            "cxcywh",
            "xyxy",
            "ltwh",
        ], f"Support `bbox_type` 'cxcywh', 'xyxy' and 'ltwh', but got {bbox_type}"
        if self.denorm:
            results["norm"] = False
        if self.with_bbox:
            results = self._load_bboxes(results)
        if self.with_label:
            results = self._load_labels(results)
        if self.with_seg:
            results = self._load_segments(results)
        if self.with_keypoint:
            results = self._load_keypoints(results)
        return results

    def _load_bboxes(self, results):
        """
        Args:
            results (dict)
        Returns:
            result (dict)
        """
        # (N, 4)
        bboxes = results["label"]["gt_bboxes"].copy()
        bboxes = Bboxes(bboxes, format=results["bbox_type"])
        results.pop("bbox_type")
        num_bboxes = len(bboxes)
        if self.denorm and num_bboxes > 0:
            h, w = results["ori_shape"][:2]
            bboxes.mul([w, h, w, h])
        results["gt_bboxes"] = bboxes
        return results

    def _load_labels(self, results):
        """
        Args:
            results (dict)
        Returns:
            results (dict)
        """
        # (N, )
        results["gt_labels"] = results["label"]["gt_classes"].copy()
        return results

    def _load_segments(self, results):
        """
        Args:
            results (dict)
        Returns:
            results (dict)
        """
        # list[np.array(n, 2)] * num_samples, n is the number of points for each instance,
        # and `num_samples` is the number of instances.
        segments = results["label"]["gt_segments"].copy()
        # list[np.array(500, 2)] * num_samples
        if len(segments) > 0:
            segments = resample_segments(segments, n=500)
            # (N, 500, 2)
            segments = np.stack(segments, dim=0)
            if self.denorm:
                h, w = results["ori_shape"][:2]
                segments[..., 0] *= w
                segments[..., 1] *= h
        results["gt_segments"] = segments
        return segments

    def _load_keypoints(self, results):
        """
        Args:
            results (dict)
        Returns:
            results (dict)
        """
        # (num_samples, nl, 2)
        keypoints = results["label"]["gt_keypoints"].copy()
        if self.denorm and len(keypoints) > 0:
            h, w = results["ori_shape"][:2]
            keypoints[..., 0] *= w
            keypoints[..., 1] *= h
        results["gt_keypoints"] = keypoints
        return results


@PIPELINES.register()
class FilterAnnotations:
    def __init__(self, wh_thr=2, ar_thr=20, eps=1e-16) -> None:
        self.wh_thr = wh_thr
        self.ar_thr = ar_thr
        self.eps = eps

    def __call__(self, results):
        # (4, num_bboxes)
        bboxes = results["gt_bboxes"].bboxes.T
        segments = results.get("gt_segments", None)
        keypoints = results.get("gt_keypoints", None)

        w, h = bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]
        ar = np.maximum(w / (h + self.eps), h / (w + self.eps))  # aspect ratio
        index = (w > self.wh_thr) & (h > self.wh_thr) & (ar < self.ar_thr)  # candidates

        bboxes = results["gt_bboxes"][index]
        results["gt_bboxes"] = bboxes
        if segments is not None:
            results["gt_segments"] = segments[index]
        if segments is not None:
            results["gt_keypoints"] = keypoints[index]
        return results

