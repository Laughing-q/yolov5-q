from lqcv import Registry, build_from_config

DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")


def build_datasets(cfg):
    return build_from_config(cfg, DATASETS)

