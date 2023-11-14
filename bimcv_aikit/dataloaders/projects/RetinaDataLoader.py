import nibabel as nib
import numpy as np
from numpy import unique
from pandas import read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

from bimcv_aikit.monai.transforms import DeleteBlackSlices
from monai import transforms
from monai.data import CacheDataset, DataLoader

config_default = {}


class RetinaDataLoader:
    def __init__(
        self,
        path: str,
        sep: str = ",",
        classes: list = ["no_RD", "RD"],
        test_run: bool = False,
        input_shape: str = "(512, 512)",
        rand_prob: int = 0.5,
        config: dict = config_default,
    ):
        df = read_csv(path, sep=sep)

        n_classes = len(unique(df["label"].values))

        self.groupby = df.groupby("partition")

        self._class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique(self.groupby.get_group("train")["label"].values),
            y=self.groupby.get_group("train")["label"].values,
        )
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys="image", image_only=True),
                transforms.EnsureChannelFirstd(keys="image", channel_dim=-1),
                transforms.CropForegroundd(keys="image", source_key="image"),
                transforms.Resized(keys="image", spatial_size=eval(input_shape)),
                transforms.ScaleIntensityd(keys="image"),
                transforms.RandRotated(keys="image", range_z=np.pi / 12, prob=0.5, keep_size=True),
                transforms.RandFlipd(keys="image", spatial_axis=0, prob=0.5),
                transforms.RandZoomd(keys="image", min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys="image", image_only=True),
                transforms.EnsureChannelFirstd(keys="image", channel_dim=-1),
                transforms.CropForegroundd(keys="image", source_key="image"),
                transforms.Resized(keys="image", spatial_size=eval(input_shape)),
                transforms.ScaleIntensityd(keys="image"),
            ]
        )
        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        if partition == "test":
            return

        data = [
            {"image": image, "label": label}
            for image, label in zip(
                self.groupby.get_group(partition)["path_xnat"].values,
                one_hot(as_tensor(self.groupby.get_group(partition)["label"].values, dtype=int)).float(),
            )
        ]
        if self.test_run:
            data = data[:16]
        if partition == "train":
            dataset = CacheDataset(data=data, transform=self.train_transforms, num_workers=7)
        else:
            dataset = CacheDataset(data=data, transform=self.val_transforms, num_workers=7)
            self.config_args["shuffle"] = False
        return DataLoader(dataset, **self.config_args)

    @property
    def class_weights(self):
        return self._class_weights
