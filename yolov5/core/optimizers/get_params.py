import torch.nn as nn

def get_yolo_params(model, weight_decay):
    """yolov5's params"""
    params = []
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    params.append({"params": g0})
    params.append({"params": g2})
    params.append({"params": g1, "weight_decay": weight_decay})
    return params

