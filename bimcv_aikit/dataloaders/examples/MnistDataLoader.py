from torch.utils.data import DataLoader
from torchvision import datasets

from ..BaseDataLoader import BaseClassificationDataLoader


class MNIST(datasets.MNIST):
    def __init__(
        self, root, train, transform, target_transform=None, download=False
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        return {"image": item[0], "label": item[1]}


class MnistDataLoader(BaseClassificationDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir: str,
        transforms: dict = {},
        dataset_kwargs: dict = {},
        test_run: bool = False,
        batch_size: int = 16,
        shuffle=False,
        num_workers=1,
    ):
        super().__init__(
            transforms, dataset_kwargs, test_run, batch_size, shuffle, num_workers
        )

        self.class_weights = [1.0] * 10
        self.data_dir = data_dir
        self.train_dataset = MNIST(
            data_dir, train=True, download=True, transform=self.transforms.get("train")
        )
        self.logger.debug(f"Train dataset size: {len(self.train_dataset)}")

    def __call__(self, partition: str) -> DataLoader:
        if partition == "train":
            return DataLoader(self.train_dataset, **self.dataloader_kwargs)
        elif partition == "test":
            test_dataset = MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=self.transforms.get(partition),
            )
            self.logger.debug(f"Test dataset size: {len(test_dataset)}")
            return DataLoader(test_dataset, **self.dataloader_kwargs)
        else:
            self.logger.warning(f'No dataset found for partition "{partition}"')
        return None
