from lqcv import Registry, build_from_config
from .get_params import get_yolo_params
import inspect
from typing import List
import torch

OPTIMIZERS = Registry("OPTIMIZERS")


def register_torch_optimizers() -> List:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith("__"):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer(model, cfg):
    cfg["params"] = get_yolo_params(model)
    return build_from_config(cfg, OPTIMIZERS)
