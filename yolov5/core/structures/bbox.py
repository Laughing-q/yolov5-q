from lqcv.bbox import (
    xywh2xyxy,
    xyxy2xywh,
    xyxy2ltwh,
    xywh2ltwh,
    ltwh2xywh,
    ltwh2xyxy,
)
from typing import List
from lqcv.utils import to_4tuple
import numpy as np
from numbers import Number

# `xyxy` means left top and right bottom
# `cxcywh` means center x, center y and width, height(yolo format)
# `ltwh` means left top and width, height(coco format)
_formats = ["xyxy", "cxcywh", "ltwh"]

__all__ = ["Bboxes"]


class Bboxes:
    def __init__(self, bboxes, format="xyxy") -> None:
        assert format in _formats
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format):
        assert format in _formats
        if self.format == format:
            bboxes = self.bboxes
        elif self.format == "xyxy":
            if format == "cxcywh":
                bboxes = xyxy2xywh(self.bboxes)
            else:
                bboxes = xyxy2ltwh(self.bboxes)
        elif self.format == "cxcywh":
            if format == "xyxy":
                bboxes = xywh2xyxy(self.bboxes)
            else:
                bboxes = xywh2ltwh(self.bboxes)
        else:
            if format == "xyxy":
                bboxes = ltwh2xyxy(self.bboxes)
            else:
                bboxes = ltwh2xywh(self.bboxes)

        return Bboxes(bboxes, format)

    # def convert(self, format):
    #     assert format in _formats
    #     if self.format == format:
    #         return
    #     if self.format == "xyxy":
    #         if format == "cxcywh":
    #             bboxes = xyxy2xywh(self.bboxes)
    #         else:
    #             bboxes = xyxy2ltwh(self.bboxes)
    #     elif self.format == "cxcywh":
    #         if format == "xyxy":
    #             bboxes = xywh2xyxy(self.bboxes)
    #         else:
    #             bboxes = xywh2ltwh(self.bboxes)
    #     else:
    #         if format == "xyxy":
    #             bboxes = ltwh2xyxy(self.bboxes)
    #         else:
    #             bboxes = ltwh2xywh(self.bboxes)
    #     self.bboxes = bboxes
    #     self.format = format

    def areas(self):
        bboxes = self.convert("xyxy").bboxes
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return area

    # def denormalize(self, w, h):
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #
    # def normalize(self, w, h):
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h

    def mul(self, scale):
        """
        Args:
            scale (tuple | List | int): the scale for four coords.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        Args:
            offset (tuple | List | int): the offset for four coords.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        return len(self.bboxes)

    @classmethod
    def cat(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """
        Concatenates a list of Boxes into a single Bboxes

        Arguments:
            boxes_list (list[Bboxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(np.empty(0))
        assert all([isinstance(box, Bboxes) for box in boxes_list])

        if len(boxes_list) == 1:
            return boxes_list[0]
        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))
        return cat_boxes

    def __getitem__(self, item) -> "Bboxes":
        """
        Args:
            item: int, slice, or a BoolArray

        Returns:
            Boxes: Create a new :class:`Bboxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Bboxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Bboxes might share storage with this Bboxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Bboxes(self.bboxes[item].view(1, -1))
        b = self.bboxes[item]
        assert (
            b.ndim == 2
        ), "Indexing on Bboxes with {} failed to return a matrix!".format(item)
        return Bboxes(b)
