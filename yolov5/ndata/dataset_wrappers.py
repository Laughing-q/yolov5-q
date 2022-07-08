from .builder import DATASETS, PIPELINES
from lqcv import build_from_config
import collections
from copy import deepcopy


@DATASETS.register()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Args:
        dataset (:obj:`BaseDataset`): The dataset to be mixed.
        pipline (Sequence[dict]): config dict to be composed.
        skip_transforms(List[str]): List of type string to be skip pipeline.
    """

    def __init__(
        self,
        dataset,
        pipeline,
        skip_transforms=None,
    ) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_transforms is not None:
            assert all(
                [isinstance(skip_transform, str) for skip_transform in skip_transforms]
            )

        self._skip_tranforms = skip_transforms
        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform["type"])
                transform = build_from_config(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError("pipeline must be a dict")

        self.dataset = dataset
        self.num_samples = len(dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        results = deepcopy(self.dataset[index])
        for transform, transform_type in zip(self.pipeline, self.pipeline_types):
            if (
                self._skip_tranforms is not None
                and transform_type in self._skip_tranforms
            ):
                continue

            if hasattr(transform, "get_indexes"):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [deepcopy(self.dataset[index]) for index in indexes]
                results["mix_results"] = mix_results

            results = transform(results)
            if "mix_results" in results:
                results.pop("mix_results")
        return results

    def update_skip_transforms(self, skip_transforms):
        """Update skip_transforms. It is called by an external hook.
        Args:
            skip_transforms (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all(
            [isinstance(skip_transform, str) for skip_transform in skip_transforms]
        )
        self._skip_tranforms = skip_transforms
