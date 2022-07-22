from .base import BaseHook
from ..utils.plots import plot_labels
from ..utils.checker import check_anchors
from ..utils.general import colorstr
from ..utils.plots import plot_images, plot_images_masks, plot_images_keypoints
from loguru import logger
import torch.nn.functional as F
from threading import Thread
import numpy as np
import torch
import os.path as osp
import time


class PlotHook(BaseHook):
    def __init__(self, cfg, save_dir, nosave=False) -> None:
        super().__init__()
        self._save_dir = save_dir
        self._nosave = nosave
        self._cfg = cfg

    def before_train(self):
        self._check_index()
        names = (
            ["item"]
            if self.trainer.single_cls and len(self.trainer.data_dict["names"]) != 1
            else self.trainer.data_dict["names"]
        )  # class names

        if self.trainer.rank == 0:
            labels = np.concatenate(self.trainer.dataset.labels, 0)
            # TODO: nosave
            if not self._nosave:
                plot_labels(labels, names, self._save_dir)

            # Anchors
            check_anchors(
                self.trainer.dataset,
                model=self.trainer.model,
                thr=self.trainer.hyp.ANCHOR_T,
                imgsz=self.trainer.img_size,
            )
            self.trainer.model.half().float()  # pre-reduce anchor precision

    def after_iter(self):
        ni = self.trainer.global_iter
        if (
            (not osp.exists(self._save_dir))
            or self._nosave
            or (ni not in self.trainer.plot_idx)
        ):
            return

        targets = self.trainer.targets
        if self._cfg.MODEL.MASK_ON:
            plot_method = plot_images_masks
            if self._cfg.MODEL.MASK_RATIO != 1:
                masks = targets["masks"]
                assert masks is not None
                masks = F.interpolate(
                    masks[None, :],
                    (self.trainer.img_size, self.trainer.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                targets["masks"] = masks
        elif self._cfg.MODEL.KEYPOINT_ON:
            plot_method = plot_images_keypoints
        else:
            plot_method = plot_images

        f = osp.join(self._save_dir, f"train_batch{ni}.jpg")  # filename
        Thread(
            target=plot_method,
            args=(self.trainer.imgs, targets, f),
            daemon=True,
        ).start()

    def _check_index(self):
        # check cls index of labels
        mlc = int(
            np.concatenate(self.trainer.dataset.labels, 0)[:, 0].max()
        )  # max label class
        assert mlc < self.trainer.num_class, (
            f"Label class {mlc} exceeds nc={self.trainer.num_class} in {self.trainer.cfg.DATA.PATH}. "
            "Possible class labels are 0-{self.num_class - 1}"
        )


class BaseLogger(BaseHook):
    def __init__(self, save_dir, nosave=False) -> None:
        super().__init__()
        self._save_dir = save_dir
        self._nosave = nosave
        self.ts = time.time()

    def before_train(self):
        logger.info(
            f"Image sizes {self.trainer.img_size} train, {self.trainer.img_size} val\n"
            f"Using {self.trainer.train_loader.num_workers} dataloader workers\n"
            f"Logging results to {colorstr('bold', None if self._nosave else self._save_dir)}\n"
            f"Starting training for {self.trainer.max_epoch} epochs..."
        )

    def before_epoch(self):
        s = ("\n" + "%10s" * 7) % (
            "Epoch",
            "gpu_mem",
            "box",
            "obj",
            "cls",
            "labels",
            "img_size",
        )
        logger.info(s)

    def after_iter(self):
        if self.trainer.rank != 0:
            return
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        self.trainer.pbar.set_description(
            ("%10s" * 2 + "%10.4g" * 5)
            % (
                f"{self.trainer.epoch}/{self.trainer.max_epoch - 1}",
                mem,
                *list(self.trainer.meter.values()),
                self.trainer.targets["labels"].shape[0],
                self.trainer.imgs.shape[-1],
            )
        )

    def after_train(self):
        logger.info(
            f"\n{self.trainer.epoch - self.trainer.start_epoch + 1} epochs completed in {(time.time() - self.ts) / 3600:.3f} hours."
        )
        logger.info(
            f"Results saved to {colorstr('bold', None if self._nosave else self._save_dir)}"
        )


class MaskLogger(BaseLogger):
    def __init__(self, save_dir, nosave=False, mask_ratio=1) -> None:
        super().__init__(save_dir, nosave)
        self.mask_ratio = mask_ratio

    def before_epoch(self):
        s = ("\n" + "%10s" * 8) % (
            "Epoch",
            "gpu_mem",
            "box",
            "seg",
            "obj",
            "cls",
            "labels",
            "img_size",
        )
        logger.info(s)

    def after_iter(self):
        if self.trainer.rank != 0:
            return
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        self.trainer.pbar.set_description(
            ("%10s" * 2 + "%10.4g" * 6)
            % (
                f"{self.trainer.epoch}/{self.trainer.max_epoch - 1}",
                mem,
                *list(self.trainer.meter.values()),
                self.trainer.targets["labels"].shape[0],
                self.trainer.imgs.shape[-1],
            )
        )


class KeyPointLogger(BaseLogger):
    def __init__(self, save_dir, nosave=False) -> None:
        super().__init__(save_dir, nosave)

    def before_epoch(self):
        s = ("\n" + "%10s" * 8) % (
            "Epoch",
            "gpu_mem",
            "box",
            "key",
            "obj",
            "cls",
            "labels",
            "img_size",
        )
        logger.info(s)

    def after_iter(self):
        if self.trainer.rank != 0:
            return
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        self.trainer.pbar.set_description(
            ("%10s" * 2 + "%10.4g" * 6)
            % (
                f"{self.trainer.epoch}/{self.trainer.max_epoch - 1}",
                mem,
                *list(self.trainer.meter.values()),
                self.trainer.targets["labels"].shape[0],
                self.trainer.imgs.shape[-1],
            )
        )


class NoAugmentHook(BaseHook):
    def __init__(self, save_dir) -> None:
        super().__init__()
        self.last_mosaic = osp.join(save_dir, "weights", "last_mosaic.pt")

    def before_train(self):
        if self.trainer.no_aug_epochs <= 0:
            return
        base_idx = (
            self.trainer.max_epoch - self.trainer.no_aug_epochs
        ) * self.trainer.max_iter
        self.trainer.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

    def before_epoch(self):
        if self.trainer.epoch != (self.trainer.max_epoch - self.trainer.no_aug_epochs):
            return
        logger.info("--->No mosaic aug now!")
        self.trainer.train_loader.close_augment()
        if self.rank == 0:
            self.trainer.save_ckpt(save_file=self.last_mosaic)


class EvalHook(BaseHook):
    def __init__(self, evaluator) -> None:
        super().__init__()
        self.evaluator = evaluator
