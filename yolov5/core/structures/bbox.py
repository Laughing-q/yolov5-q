from lqcv.bbox import (
    xywh2xyxy,
    xyxy2xywh,
    xyxy2ltwh,
    xywh2ltwh,
    ltwh2xywh,
    ltwh2xyxy,
)

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
        if self.format == "xyxy":
            if format == "cxcywh":
                bboxes = xyxy2xywh(bboxes)
            else:
                bboxes = xyxy2ltwh(bboxes)
        elif self.format == "cxcywh":
            if format == "xyxy":
                bboxes = xywh2xyxy(bboxes)
            else:
                bboxes = xywh2ltwh(bboxes)
        else:
            if format == "xyxy":
                bboxes = ltwh2xyxy(bboxes)
            else:
                bboxes = ltwh2xywh(bboxes)

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

    def denormalize(self, w, h):
        assert (self.bboxes <= 1.0).all()
        self.bboxes[:, 0::2] *= w
        self.bboxes[:, 1::2] *= h

    def normalize(self, w, h):
        assert (self.bboxes > 1.0).any()
        self.bboxes[:, 0::2] /= w
        self.bboxes[:, 1::2] /= h

    def add_offset(self, offset):
        """
        Args:
            offset (tuple | List): the offset for four coords.
        """
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        return len(self.bboxes)
