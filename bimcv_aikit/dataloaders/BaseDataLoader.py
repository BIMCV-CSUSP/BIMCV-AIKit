import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import Union

import torch
from torch.utils.data.dataloader import default_collate

from ..utils.config import init_obj


class BaseDataLoader(Callable):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        transforms: dict = {},
        dataset_kwargs: dict = {},
        test_run: bool = False,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 1,
        collate_fn: Callable = default_collate,
        **kwargs,
    ):
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            **kwargs,
        }
        self.dataset_kwargs = dataset_kwargs
        self.logger = logging.getLogger("dataloader")
        self.transforms = self.init_transforms(transforms)
        self.test_run = test_run

    @abstractmethod
    def __call__(self, partition: str) -> Union[torch.utils.data.DataLoader, None]:
        """
        Returns the data loader for a given partition
        """
        return NotImplementedError

    def init_transforms(self, transforms_config: dict) -> dict:
        """
        Initializes the transforms from a configuration dictionary.
        """
        if not transforms_config:
            return {}
        transforms = {}
        for partition, transform_config in transforms_config.items():
            if not transform_config:
                transforms[partition] = None
                continue
            try:
                transform_list = [
                    init_obj(
                        transform["module"], transform["type"], **transform["args"]
                    )
                    for transform in transform_config["args"]["transforms"]
                    if isinstance(transform, dict)
                ]
                if len(transform_list) > 0:
                    transform_config["args"]["transforms"] = transform_list
            except Exception as e:
                self.logger.error(
                    f"Error defining transforms for {partition} partition"
                )
                raise e
            transforms[partition] = init_obj(
                transform_config["module"],
                transform_config["type"],
                **transform_config["args"],
            )
        self.logger.debug(f"Transforms initialized: {transforms}")
        return transforms


class BaseClassificationDataLoader(BaseDataLoader):
    _class_weights: Union[list[float], None] = None

    @property
    def class_weights(self) -> Union[list[float], None]:
        """
        Returns the class weights for the dataset.
        """
        return self._class_weights

    @class_weights.setter
    def class_weights(self, class_weights: list[float]):
        self._class_weights = class_weights


class BaseSegmentationDataLoader(BaseDataLoader):

    @property
    def class_weights(self) -> Union[list[float], None]:
        """
        Returns the class weights for the dataset.
        """
        return None
