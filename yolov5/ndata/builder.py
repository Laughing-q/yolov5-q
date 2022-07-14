from lqcv import Registry, build_from_config
from copy import deepcopy

DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")


def build_datasets(cfg):
    if cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = deepcopy(cfg)
        cp_cfg['dataset'] = build_datasets(cp_cfg['dataset'])
        return build_from_config(cp_cfg, DATASETS)
    return build_from_config(cfg, DATASETS)

