from .builder import HOOKS
from .base import BaseHook
from yolov5.utils.torch_utils import is_parallel


@HOOKS.register()
class EMAHook(BaseHook):
    """Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook. Note,
    the original model parameters are actually saved in ema field after train.

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = (1-momentum) * ema_param + momentum * cur_param`.
            Defaults to 0.0002.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to False.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        resume_from (str, optional): The checkpoint path. Defaults to None.
        momentum_fun (func, optional): The function to change momentum
            during early iteration (also warmup) to help early training.
            It uses `momentum` as a constant. Defaults to None.
    """

    def __init__(
        self,
        momentum=0.0002,
        interval=1,
        skip_buffers=False,
        resume_from=None,
        momentum_fun=None,
    ):
        assert 0 < momentum < 1
        self.momentum = momentum
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.checkpoint = resume_from
        self.momentum_fun = momentum_fun

    def before_run(self, trainer):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model.
        """
        model = trainer.model
        if is_parallel(model):
            model = model.module
        self.param_ema_buffer = {}
        if self.skip_buffers:
            self.model_parameters = dict(model.named_parameters())
        else:
            self.model_parameters = model.state_dict()
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers())
        if self.checkpoint is not None:
            trainer.resume(self.checkpoint)

    def get_momentum(self, trainer):
        return self.momentum_fun(trainer.iter) if self.momentum_fun else self.momentum

    def after_train_iter(self, trainer):
        """Update ema parameter every self.interval iterations."""
        if (trainer.iter + 1) % self.interval != 0:
            return
        momentum = self.get_momentum(trainer)
        for name, parameter in self.model_parameters.items():
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(1 - momentum).add_(parameter.data, alpha=momentum)

    def after_epoch(self, trainer):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_epoch(self, trainer):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
