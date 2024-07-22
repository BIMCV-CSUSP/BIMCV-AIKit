from pathlib import Path
from typing import Any, Union

import pandas as pd
from monai.data import CacheDataset, DataLoader
from torch.utils.data import Dataset as TorchDataset

from .BaseDataLoader import BaseDataLoader


class CSVDataLoader(BaseDataLoader):
    """
    A base data loader for datasets stored in a csv file.
    """

    def __init__(
        self,
        path: Union[str, Path],
        data_column: Union[str, list[str]],
        label_column: str,
        partition_column: str,
        classes: list[Any] = [],
        transforms: dict = {},
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 1,
        dataset_class: TorchDataset = CacheDataset,
        test_run: bool = False,
        read_csv_kwargs: dict = {},
        dataloader_kwargs: dict = {},
        dataset_kwargs: dict = {},
    ):
        """
        Args:
            path (str): Path to the CSV file containing the data.
            data_column (str or list[str]): Column name(s) in the CSV file containing the path/features.
            label_column (str): Column name in the CSV file containing the label.
            partition_column (str): Column name in the CSV file containing the train/dev/test partition label.
            classes (list, optional): List of classes to include in the data. Defaults to None.
            transforms (dict, optional): Dictionary containing the transforms for each partition. Defaults to {}.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the DataLoader. Defaults to False.
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 1.
            dataset_class (torch.utils.data.Dataset, optional): PyTorch/MONAI Dataset class to use. Defaults to monai.data.CacheDataset.
            test_run (bool, optional): If True, only a small subset of the data will be loaded for testing purposes (dry run). Defaults to False.
            read_csv_kwargs (dict, optional): Additional keyword arguments for `pandas.read_csv`. Defaults to {}.
            dataloader_kwargs (dict, optional): Additional keyword arguments for `torch.utils.data.DataLoader`. Defaults to {}.
        """
        super().__init__(
            transforms,
            dataset_kwargs,
            test_run,
            batch_size,
            shuffle,
            num_workers,
            **dataloader_kwargs,
        )
        df = pd.read_csv(path, **read_csv_kwargs)
        if classes:
            df = df.loc[df[label_column].isin(classes)]
            self.logger.debug(
                f"Filtering dataframe with the following classes: {classes}"
            )
        def_classes = pd.unique(df[label_column])
        if type(def_classes[0]) is int:
            map_labels = {class_: i for i, class_ in enumerate(def_classes)}
            df[label_column] = df[label_column].map(map_labels)
            self.logger.warning(
                f"Non-integer labels found. Mapping labels to integers: {map_labels}"
            )
        data_column_list = (
            data_column if isinstance(data_column, list) else [data_column]
        )
        image_col = data_column_list[0]
        df = df[[*data_column_list, label_column, partition_column]]
        df.rename(columns={image_col: "image", label_column: "label"}, inplace=True)
        self.df = df.groupby(partition_column)
        self.dataset_class = dataset_class

    def __call__(self, partition: str):
        """
        Returns a DataLoader object for the specified partition.

        Args:
            partition (str): The partition to load data for (e.g. "train", "val", or "test").

        Returns:
            DataLoarected to our Quick Start Guide. Simply follow theder: A PyTorch DataLoader object containing the specified partition's data.
        """
        try:
            data = self.df.get_group(partition)
        except KeyError:
            self.logger.warning(f"Partition '{partition}' not found in the CSV file.")
            return None
        data = data.to_dict("records")
        if self.test_run:
            data = data[:16]
        dataset = self.dataset_class(
            data=data, transform=self.transforms.get(partition), **self.dataset_kwargs
        )
        return DataLoader(dataset, **self.dataloader_kwargs)
