from ..builder import PIPELINES
import torch
import numpy as np
import cv2
import pycocotools.mask as maskUtils


@PIPELINES.register()
class FormatBundle:
    """The final processing of data.
    Args:
        bbox_type (str): bbox type.
        coord_norm (bool): Whether to normalize the coords.
        mask_ratio (int): The downsample ratio of masks.
    """

    def __init__(self, bbox_type="cxcywh", coord_norm=True, mask_ratio=4) -> None:
        assert bbox_type in [
            "cxcywh",
            "xyxy",
            "ltwh",
        ], f"Support `bbox_type` 'cxcywh', 'xyxy' and 'ltwh', but got {bbox_type}"
        self.bbox_type = bbox_type
        self.coord_norm = coord_norm
        self.mask_ratio = mask_ratio

    def __call__(self, results):
        results = self._format_img(results)
        results = self._format_label(results)
        results = self._format_bboxes(results)
        if results.get("gt_segments", None) is not None:
            results = self._format_segments(results)
        if results.get("gt_keypoints", None) is not None:
            results = self._format_keypoints(results)
        return results

    def _format_img(self, results):
        img = results["img"]
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img = torch.from_numpy(img)
        results["img"] = img
        return results

    def _format_label(self, results):
        labels = torch.from_numpy(results["gt_labels"])
        results["gt_labels"] = labels
        return results

    def _format_bboxes(self, results):
        bboxes = results["gt_bboxes"].convert("cxcywh")
        if self.coord_norm and (not results["norm"]):
            h, w = results["img"].shape[:2]
            bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        results["gt_bboxes"] = torch.from_numpy(bboxes.bboxes)
        return results

    def _format_segments(self, results):
        """convert polygon points to bitmap"""
        segments = results["gt_segments"]
        h, w = results["img"].shape[:2]
        segments[..., 0] *= w
        segments[..., 1] *= h
        if self.mask_ratio > 1:
            segments /= self.mask_ratio
            h /= self.mask_ratio
            w /= self.mask_ratio
        # func: `polygon_to_bitmap` need a list
        masks = [polygon_to_bitmap([segment], h, w) for segment in segments]
        masks = np.stack(masks, axis=0) if len(masks) else np.zeros(0, h, w)
        results["gt_masks"] = torch.from_numpy(masks)
        results.pop("gt_segments")
        return results

    def _format_keypoints(self, results):
        keypoints = results["gt_keypoints"]
        if self.coord_norm and (not results["norm"]):
            h, w = results["img"].shape[:2]
            keypoints[..., 0] /= w
            keypoints[..., 1] /= h
        results["gt_keypoints"] = torch.from_numpy(keypoints)
        return results


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
