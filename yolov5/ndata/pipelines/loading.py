from ..builder import PIPELINES
from yolov5.utils.segment import resample_segments
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
    ) -> None:
        self.with_label = with_label
        self.with_bbox = with_bbox
        self.with_seg = with_seg
        self.with_keypoint = with_keypoint
        self.denorm = denorm

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        if self.denorm:
            results["norm"] = False
        if self.with_bbox:
            results = self._load_bboxes(results)
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
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
        bboxes = results["label"].copy()
        num_bboxes = len(bboxes)
        if self.denorm and num_bboxes > 0:
            h, w = results["ori_shape"][:2]
            bboxes[:, 0::2] *= w
            bboxes[:, 1::2] *= h
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

    def _load_masks(self, results):
        """
        Args:
            results (dict)
        Returns:
            results (dict)
        """
        # list[np.array(n, 2)] * N, n is the number of points for each instance,
        # and N is the number of instances.
        segments = results["label"]["gt_segments"].copy()
        # list[np.array(500, 2)] * N
        segments = resample_segments(segments, n=500)
        # (N, 500, 2)
        segments = np.stack(segments, dim=0)
        if self.denorm and len(segments) > 0:
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
        # (N, nl, 2)
        keypoints = results["label"]["gt_keypoints"].copy()
        if self.denorm and len(keypoints) > 0:
            h, w = results["ori_shape"][:2]
            keypoints[..., 0] *= w
            keypoints[..., 1] *= h
        results["gt_keypoints"] = keypoints
        return results
