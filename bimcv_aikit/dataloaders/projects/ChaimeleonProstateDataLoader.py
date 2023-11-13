import json

import numpy as np
from monai import transforms
from monai.data import CacheDataset, DataLoader
from numpy import unique
from pandas import read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

from bimcv_aikit.monai.transforms import DeleteBlackSlices

config_default = {}


class ChaimeleonProstateDataLoader:
    def __init__(
        self,
        json_path: str,
        classes: list = ["Low", "High"],
        test_run: bool = False,
        input_shape: str = "(256, 256,30)",
        rand_prob: int = 0.15,
        config: dict = config_default,
    ):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        classes = np.vstack([x["label"] for x in self.data])

        self._class_weights = [2.,1.]#compute_class_weight(class_weight="balanced", classes=[0, 1], y=np.argmax(classes, axis=0))
        print(self._class_weights)
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys="image", reader="NibabelReader", image_only=True),
                transforms.EnsureChannelFirstd(keys="image", channel_dim=None),
                transforms.Orientationd(keys="image", axcodes="RAS"),
                transforms.Resized(
                    keys="image",
                    spatial_size=eval(input_shape),
                    mode=("trilinear"),
                ),
                transforms.NormalizeIntensityd(keys="image"),
                transforms.ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
                transforms.DataStatsd(keys="image"),
                #transforms.RandRotate90d(keys=["image"], spatial_axes=[0, 1], prob=rand_prob, max_k=3),
                #transforms.RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, mode="area", prob=rand_prob),
                # transforms.RandGaussianNoised(keys=["image"], mean=0.1, std=0.25, prob=rand_prob),
                # transforms.RandShiftIntensityd(keys=["image"], offsets=0.2, prob=rand_prob),
                # transforms.RandGaussianSharpend(
                #     keys=["image"],
                #     sigma1_x=[0.5, 1.0],
                #     sigma1_y=[0.5, 1.0],
                #     sigma1_z=[0.5, 1.0],
                #     sigma2_x=[0.5, 1.0],
                #     sigma2_y=[0.5, 1.0],
                #     sigma2_z=[0.5, 1.0],
                #     alpha=[10.0, 30.0],
                #     prob=rand_prob,
                # ),
                #transforms.RandAdjustContrastd(keys=["image"], gamma=2.0, prob=rand_prob),
                transforms.ToTensord(keys=["image", "label","numeric"]),
            ]
        )

        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        if partition == "test" or partition == "val":
            return

        if self.test_run:
            self.data = self.data[:16]
        dataset = CacheDataset(data=self.data, transform=self.train_transforms, num_workers=7)
        return DataLoader(dataset, **self.config_args)

    @property
    def class_weights(self):
        return self._class_weights
