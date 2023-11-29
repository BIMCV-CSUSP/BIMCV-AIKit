from abc import abstractmethod
from collections.abc import Callable
from typing import Union

from torch.utils.data.dataloader import default_collate

from ..utils.config import init_obj


class BaseDataLoader(Callable):
    """
    Base class for all data loaders
    """

    def __init__(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 1, collate_fn: Callable = default_collate, **kwargs):
        self.dataloader_kwargs = {"batch_size": batch_size, "shuffle": shuffle, "collate_fn": collate_fn, "num_workers": num_workers, **kwargs}

    @abstractmethod
    def __call__(self, partition: str):
        """
        Returns the data loader for a given partition
        """
        return NotImplementedError

    @staticmethod
    def init_transforms(transforms_config: dict):
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
                    init_obj(transform["module"], transform["type"], **transform["args"]) for transform in transform_config["args"]["transforms"]
                ]
            except Exception as e:
                print(f"Error defining transforms for {partition} partition")
                raise e
            transform_config["args"]["transforms"] = transform_list
            transforms[partition] = init_obj(transform_config["module"], transform_config["type"], **transform_config["args"])
        return transforms


class BaseClassificationDataLoader(BaseDataLoader):
    class_weights: Union[list, None] = None

    @property
    def class_weights(self):
        """
        Returns the class weights for the dataset.
        """
        return self._class_weights

    @class_weights.setter
    def class_weights(self, class_weights: list):
        self._class_weights = class_weights
