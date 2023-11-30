from torchvision import datasets
from torch.utils.data import DataLoader
from .BaseDataLoader import BaseClassificationDataLoader


class MNIST(datasets.MNIST):
    def __init__(self, root, train, transform, target_transform=None, download=False) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        return {"image": item[0], "label": item[1]}


class MnistDataLoader(BaseClassificationDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir: str, transforms: dict = {}, batch_size: int = 16, shuffle=False, num_workers=1):
        super().__init__(batch_size, shuffle, num_workers)

        self.class_weights = [1.0] * 10
        self.data_dir = data_dir
        transforms_dict = self.init_transforms(transforms)
        self.transform = transforms_dict
        self.train_dataset = MNIST(data_dir, train=True, download=True, transform=transforms_dict["train"])

    def __call__(self, partition: str) -> DataLoader:
        if partition == "train":
            return DataLoader(self.train_dataset, **self.dataloader_kwargs)
        if partition == "test":
            test_dataset = MNIST(self.data_dir, train=False, download=True, transform=self.transform[partition])
            return DataLoader(test_dataset, **self.dataloader_kwargs)
        return None
