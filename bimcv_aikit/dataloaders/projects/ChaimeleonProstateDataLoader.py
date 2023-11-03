from numpy import unique
import numpy as np
from pandas import read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

from monai.data import CacheDataset, DataLoader
from monai import transforms
from bimcv_aikit.monai.transforms import DeleteBlackSlices
import json

config_default = {}

class ChaimeleonProstateDataLoader:
    def __init__(self, json_path: str, classes: list = ["Low", "High"], test_run: bool = False, input_shape: str = "(256, 256,30)", rand_prob: int = 0.5, config: dict = config_default):

        with open(json_path, "r") as f:
            self.data = json.load(f)

        classes = np.vstack([x['label'] for x in self.data])
        
        self._class_weights = compute_class_weight(
            class_weight="balanced", 
            classes=[0,1], 
            y=np.argmax(classes,axis=0)
        )
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(
                    keys="image", reader="NibabelReader", image_only=True
                ),
                transforms.EnsureChannelFirstd(keys="image",channel_dim = None),
                transforms.Resized(
                    keys="image",
                    spatial_size=eval(input_shape),
                    mode=("trilinear"),
                ),
                transforms.ScaleIntensityd(keys='image', minv=0.0, maxv=1.0),
                transforms.NormalizeIntensityd(keys='image'),
                transforms.ToTensord(keys=['image','label'])
            ]
        )

        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        
        if partition == "test" or partition=="val":
            return 
      
        if self.test_run:
            self.data = self.data[:16]
        dataset = CacheDataset(data=self.data, transform=self.train_transforms, num_workers=7)
        return DataLoader(dataset, **self.config_args)

    @property
    def class_weights(self):
        return self._class_weights