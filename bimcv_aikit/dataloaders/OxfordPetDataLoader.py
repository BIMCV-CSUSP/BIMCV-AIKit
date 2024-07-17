from numpy import asarray
from torch import tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .BaseDataLoader import BaseSegmentationDataLoader


class OxfordPet(datasets.OxfordIIITPet):
    def __init__(
        self,
        root,
        split,
        target_types,
        transform,
        target_transform,
        download=False,
    ) -> None:
        super().__init__(
            root,
            split=split,
            target_types=target_types,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        label = asarray(item[1])
        label = tensor(label).unsqueeze(0)
        return {"image": item[0], "label": label}


class OxfordIIITPetDataLoader(BaseSegmentationDataLoader):
    """
    OxfordIIITPet data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir: str,
        transforms: dict = {},
        batch_size: int = 4,
        shuffle=False,
        num_workers=1,
    ):
        super().__init__(batch_size, shuffle, num_workers)

        self.data_dir = data_dir
        transforms_dict = self.init_transforms(transforms)
        self.transform = transforms_dict
        self.train_dataset = Subset(
            OxfordPet(
                self.data_dir,
                split="trainval",
                target_types="segmentation",
                transform=self.transform["train_image"],
                target_transform=self.transform["train_label"],
                download=True,
            ),
            range(10),
        )

    def __call__(self, partition: str) -> DataLoader:
        if partition == "train":
            return DataLoader(self.train_dataset, **self.dataloader_kwargs)
        if partition == "test":
            test_dataset = Subset(
                OxfordPet(
                    self.data_dir,
                    split="test",
                    target_types="segmentation",
                    transform=self.transform["test_image"],
                    target_transform=self.transform["test_label"],
                    download=True,
                ),
                range(10),
            )
            return DataLoader(test_dataset, **self.dataloader_kwargs)
        return None
