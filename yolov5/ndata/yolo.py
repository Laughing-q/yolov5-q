from .base import BaseDataset
from .data_utils import IMG_FORMATS, exif_size
from .builder import DATASETS
from ..utils.segment import segments2boxes
from multiprocessing.pool import Pool
from PIL import Image, ImageOps
import glob
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import os
import os.path as osp
import numpy as np
import logging


@DATASETS.register()
class YOLODetectionDataset(BaseDataset):
    """YOLO Dataset.
    Args:
        img_path (str): image path.
        pipeline (dict): a dict of image transforms.
        prefix (str): prefix.
        rect (bool): rect resize, default: False, set True
            when eval or inference, keep False when training.
        batch_size (int): give this argument when rect = True.
    """
    def __init__(
        self,
        img_path,
        pipeline,
        prefix="",
        rect: Optional[bool] = False,
        batch_size: Optional[int] = None,
        test_mode: Optional[bool] = False,
    ):
        super().__init__(pipeline=pipeline, test_mode=test_mode)
        self.prefix = prefix

        self.img_files = self.load_img_files(img_path)
        self.label_files = self._img2label_files()

        self.labels, self.shapes = self.load_labels()
        self.set_rectangle(rect, batch_size)
        self.rect = rect

    def load_img_files(self, img_path):
        """Read image files."""
        p = Path(img_path)  # os-agnostic
        try:
            f = []  # image files
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p, "r") as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [
                        x.replace("./", parent) if x.startswith("./") else x for x in t
                    ]  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise Exception(f"{p} does not exist")
            img_files = sorted(
                [
                    x.replace("/", os.sep)
                    for x in f
                    if x.split(".")[-1].lower() in IMG_FORMATS
                ]
            )
            # img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert img_files, f"No images found"
        except Exception as e:
            raise Exception(f"Error loading data from {str(p)}: {e}\n")
        return img_files

    def load_labels(self):
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = (
                np.load(cache_path, allow_pickle=True).item(),
                True,
            )  # load dict
        except:
            cache, exists = self._cache_labels(cache_path), False  # cache
        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(
                None, desc=self.prefix + d, total=n, initial=n
            )  # display cache results
            if cache["msgs"]:
                logging.info("\n".join(cache["msgs"]))  # display warnings
        assert (
            nf > 0 or not self.augment
        ), f"{self.prefix}No labels in {cache_path}. Can not train without labels"

        # Read cache
        cache.pop("msgs")
        labels, shapes = zip(*cache.values())
        return labels, shapes

    def prepare_train_img(self, index):
        img_file = self.img_files[index]
        label = self.labels[index]
        # NOTE: the shapes are from PIL, so the format is wh.
        w, h = self.shapes[index]
        ori_shape = (h, w)
        # `norm` means the coords are normalized.
        results = dict(
            img_file=img_file,
            label=label,
            ori_shape=ori_shape,
            norm=True,
            bbox_type="cxcywh",
        )
        if self.rect:
            results["rect_shape"] = self.batch_shapes[self.batch_index[index]]
        return self.pipeline(results)

    def prepare_test_img(self, index):
        pass

    def set_rectangle(self, rect, batch_size):
        if not rect:
            return
        assert batch_size is not None
        num_imgs = len(self.shapes)  # number of images
        batch_index = np.floor(np.arange(num_imgs) / batch_size).astype(
            np.int
        )  # batch index
        num_batches = batch_index[-1] + 1  # number of batches
        self.batch_index = batch_index  # batch index of image

        """Update attr if rect is True."""
        # Sort by aspect ratio
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_files = [self.img_files[i] for i in irect]
        self.label_files = [self.label_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * num_batches
        for i in range(num_batches):
            ari = ar[self.batch_index == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.img_size / self.stride).astype(np.int)
            * self.stride
        )

    def _img2label_files(self):
        sa, sb = (
            os.sep + "images" + os.sep,
            os.sep + "labels" + os.sep,
        )  # /images/, /labels/ substrings
        return [
            sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in self.img_files
        ]

    def _check_image_label(self, args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            # verify images
            im = Image.open(img_path)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(img_path, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(img_path)).save(
                            img_path, "JPEG", subsampling=0, quality=100
                        )
                        msg = f"{self.prefix}WARNING: {img_path}: corrupt JPEG restored and saved"

            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f"{self.prefix}WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    bboxes = labels[:, 1:]
                    classes = labels[:, 0]
                else:
                    ne = 1  # label empty
                    bboxes = np.zeros((0, 4), dtype=np.float32)
                    classes = np.zeros((0, 1), dtype=np.float32)
            else:
                nm = 1  # label missing
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0, 1), dtype=np.float32)

            targets = dict(gt_bboxes=bboxes, gt_classes=classes,)
            return img_path, targets, shape, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f"{self.prefix}WARNING: {lb_path}: ignoring invalid labels: {e}"
            return [None, None, None, nm, nf, ne, nc, msg]

    def _cache_labels(self, cache_path):
        num_threads = min(8, os.cpu_count())
        x = {}  # dict
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning '{cache_path.parent / cache_path.stem}' images and labels..."

        with Pool(num_threads) as pool:
            pbar = tqdm(
                pool.imap(
                    self._check_image_label, zip(self.img_files, self.label_files,),
                ),
                desc=desc,
                total=len(self.img_files),
            )
            for (im_file, l, shape, nm_f, nf_f, ne_f, nc_f, msg,) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape]
                if msg:
                    msgs.append(msg)
                pbar.desc = (
                    f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
                )

        pbar.close()
        if msgs:
            logging.info("\n".join(msgs))
        if nf == 0:
            logging.info(f"{self.prefix}WARNING: No labels found in {cache_path}")
        x["results"] = nf, nm, ne, nc, len(self.img_files)
        x["msgs"] = msgs  # warnings
        try:
            np.save(cache_path, x)  # save cache for next time
            cache_path.with_suffix(".cache.npy").rename(
                cache_path
            )  # remove .npy suffix
            logging.info(f"{self.prefix}New cache created: {cache_path}")
        except Exception as e:
            logging.info(
                f"{self.prefix}WARNING: Cache directory {cache_path.parent} is not writeable: {e}"
            )  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)


@DATASETS.register()
class YOLOSegmentDataset(YOLODetectionDataset):
    def __init__(
        self,
        img_path,
        pipeline,
        prefix="",
        rect: Optional[bool] = False,
        batch_size: Optional[int] = None,
        test_mode: Optional[bool] = False,
    ):
        super().__init__(img_path, pipeline, prefix, rect, batch_size, test_mode)

    def _check_image_label(self, args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            # verify images
            im = Image.open(img_path)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(img_path, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(img_path)).save(
                            img_path, "JPEG", subsampling=0, quality=100
                        )
                        msg = f"{self.prefix}WARNING: {img_path}: corrupt JPEG restored and saved"

            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    classes = np.array([x[0] for x in labels], dtype=np.float32)
                    # TODO
                    segments = [
                        np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in labels
                    ]  # (cls, xy1...)
                    bboxes = segments2boxes(segments)
                nl = len(bboxes)
                if len(bboxes):
                    assert (
                        bboxes >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        bboxes <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    bboxes, idx = np.unique(
                        bboxes, axis=0, return_index=True
                    )  # remove duplicate rows
                    # NOTE: `np.unique` will change the order of `l`, so adjust the segments order too.
                    segments = (
                        [segments[i] for i in idx] if len(segments) > 0 else segments
                    )
                    if len(bboxes) < nl:
                        msg = f"{self.prefix}WARNING: {img_path}: {nl - len(bboxes)} duplicate labels removed"
                else:
                    ne = 1  # label empty
                    segments = np.zeros((0, 10), dtype=np.float32)
                    bboxes = np.zeros((0, 4), dtype=np.float32)
                    classes = np.zeros((0, 1), dtype=np.float32)
            else:
                nm = 1  # label missing
                segments = np.zeros((0, 10), dtype=np.float32)
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0, 1), dtype=np.float32)

            targets = dict(gt_bboxes=bboxes, gt_classes=classes, gt_segments=segments)
            return img_path, targets, shape, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f"{self.prefix}WARNING: {lb_path}: ignoring invalid labels: {e}"
            return [None, None, None, nm, nf, ne, nc, msg]


@DATASETS.register()
class YOLOKeypointDataset(YOLODetectionDataset):
    def __init__(
        self,
        img_path,
        pipeline,
        prefix="",
        rect: Optional[bool] = False,
        batch_size: Optional[int] = None,
        test_mode: Optional[bool] = False,
    ):
        super().__init__(img_path, pipeline, prefix, rect, batch_size, test_mode)

    def _check_image_label(self, args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            # verify images
            im = Image.open(img_path)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(img_path, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(img_path)).save(
                            img_path, "JPEG", subsampling=0, quality=100
                        )
                        msg = f"{self.prefix}WARNING: {img_path}: corrupt JPEG restored and saved"

            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    # xyxy, (N, nl, 2)
                    keypoints = np.array(
                        [x[5:] for x in labels], dtype=np.float32
                    ).reshape(len(labels), -1, 2)
                    # cls, xywh
                    labels = np.array([x[:5] for x in labels], dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        keypoints = keypoints[indices]
                        msg += f"{self.prefix}WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    bboxes = labels[:, 1:]
                    classes = labels[:, 0]
                else:
                    ne = 1  # label empty
                    keypoints = np.zeros((0, 10), dtype=np.float32)
                    bboxes = np.zeros((0, 4), dtype=np.float32)
                    classes = np.zeros((0, 1), dtype=np.float32)
            else:
                nm = 1  # label missing
                keypoints = np.zeros((0, 10), dtype=np.float32)
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0, 1), dtype=np.float32)

            targets = dict(
                gt_bboxes=bboxes, gt_classes=classes, gt_keypoints=keypoints,
            )
            return img_path, targets, shape, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f"{self.prefix}WARNING: {lb_path}: ignoring invalid labels: {e}"
            return [None, None, None, nm, nf, ne, nc, msg]
