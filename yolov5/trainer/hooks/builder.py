from lqcv import Registry, build_from_config


HOOKS = Registry("HOOKS")

def build_hook(cfg):
    return build_from_config(cfg, HOOKS)
