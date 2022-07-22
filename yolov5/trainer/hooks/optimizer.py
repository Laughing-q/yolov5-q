from .base import BaseHook
from .builder import HOOKS
from torch.cuda import amp


@HOOKS.register()
class YOLOV5OptimzierHook(BaseHook):
    def __init__(self, accumulate) -> None:
        super().__init__()
        self.accumulate = accumulate
        self.last_iter = -1

    def after_batch(self, trainer):
        trainer.outputs["loss"].backward()
        if trainer.iter - self.last_iter >= self.accumulate:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            self.last_iter = trainer.iter


@HOOKS.register()
class YOLOV5FP16OptimzierHook(BaseHook):
    def __init__(self, accumulate) -> None:
        super().__init__()
        self.accumulate = accumulate
        self.last_iter = -1
        self.scaler = amp.GradScaler(enabled=self.cuda)

    def before_train(self, trainer):
        for m in trainer.model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True

    def after_batch(self, trainer):
        self.scaler.scale(trainer.outputs["loss"]).backward()

        # Optimize
        if trainer.iter - self.last_iter >= self.accumulate:
            self.scaler.step(trainer.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            self.last_iter = trainer.iter
