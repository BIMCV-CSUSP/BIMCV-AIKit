import nibabel as nib
from numpy import array, float32, unique
from pandas import read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

from bimcv_aikit.monai.transforms import DeleteBlackSlices

config_default = {}


class ProstateClassDataLoader:
    def __init__(
        self,
        path: str,
        sep: str = ",",
        classes: list = ["noCsPCa", "CsPCa"],
        format_load: str = "cropped",
        img_columns=["t2", "adc", "dwi"],
        test_run: bool = False,
        input_shape: str = "(128, 128, 32)",
        rand_prob: int = 0.5,
        config: dict = config_default,
    ):
        df = read_csv(path, sep=sep)
        self.format_load = format_load
        format_load = "cropped"  #'nifti', mha, cropped

        df["depth"] = df["filepath_t2w_" + self.format_load].apply(lambda path_file: nib.load(path_file).shape[0])
        df["heigth"] = df["filepath_t2w_" + self.format_load].apply(lambda path_file: nib.load(path_file).shape[1])
        df["weigth"] = df["filepath_t2w_" + self.format_load].apply(lambda path_file: nib.load(path_file).shape[2])
        df = df[(df["heigth"] != 0) & (df["depth"] != 0)]
        df = df[df["filepath_t2w_" + self.format_load].notna()].reset_index()

        df = df[(df["heigth"] > 96) & (df["depth"] > 96)]
        n_classes = len(unique(df["label"].values))

        self.groupby = df.groupby("partition")

        self._class_weights = compute_class_weight(
            class_weight="balanced", classes=unique(self.groupby.get_group("tr")["label"].values), y=self.groupby.get_group("tr")["label"].values
        )
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=img_columns + ["zones"], reader="NibabelReader", image_only=True),
                transforms.EnsureChannelFirstd(keys=img_columns + ["zones"]),
                transforms.AsDiscreted(keys="zones", argmax=False, to_onehot=3),
                transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
                transforms.ResampleToMatchd(
                    keys=["adc", "dwi", "zones"],
                    key_dst="t2",
                    mode=("bilinear", "bilinear", "nearest"),
                ),  # Resample images to t2 dimension
                transforms.Resized(
                    keys=img_columns + ["zones"],
                    spatial_size=eval(input_shape),
                    mode=("trilinear", "trilinear", "trilinear", "nearest"),
                ),  # SAMUNETR: Reshape to have the same dimension
                transforms.ScaleIntensityd(keys=img_columns, minv=0.0, maxv=1.0),
                transforms.NormalizeIntensityd(keys=img_columns),
                transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
                # transforms.RandCropByPosNegLabeld(keys=["image"], label_key="zones", spatial_size=[96, 96, 32],num_samples=4,image_key="image",image_threshold=0)
                # transforms.RandSpatialCropSamplesd(keys=['image','label'],roi_size=[96,96,-1],num_samples=8,random_size=False),#For the other models
                # transforms.RandRotate90d(keys=['image'],spatial_axes=0,prob=prob),
                # transforms.RandZoomd(keys=['image'],min_zoom=0.9,max_zoom=1.1,mode='area',prob=prob),
                # transforms.RandGaussianNoised(keys=["image"],mean=0.1,std=0.25,prob=prob),
                # transforms.RandShiftIntensityd(keys=["image"],offsets=0.2,prob=prob),
                # transforms.RandGaussianSharpend(keys=['image'],sigma1_x=[0.5, 1.0],sigma1_y=[0.5, 1.0],sigma1_z=[0.5, 1.0],sigma2_x=[0.5, 1.0],sigma2_y=[0.5, 1.0],sigma2_z=[0.5, 1.0],alpha=[10.0,30.0],prob=prob),
                # transforms.RandAdjustContrastd(keys=['image'],gamma=2.0,prob=prob),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=img_columns + ["zones"], image_only=True),
                transforms.EnsureChannelFirstd(keys=img_columns + ["zones"]),
                transforms.AsDiscreted(keys="zones", argmax=True, to_onehot=3),
                transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
                transforms.ResampleToMatchd(
                    keys=["adc", "dwi", "zones"],
                    key_dst="t2",
                    mode=("bilinear", "bilinear", "nearest"),
                ),  # Resample images to t2 dimensions
                transforms.Resized(
                    keys=img_columns + ["zones"],
                    spatial_size=eval(input_shape),
                    mode=("trilinear", "trilinear", "trilinear", "nearest"),
                ),  # SAMUNETR: Reshape to have the same dimension
                transforms.ScaleIntensityd(keys=img_columns, minv=0.0, maxv=1.0),
                transforms.NormalizeIntensityd(keys=img_columns),
                transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
            ]
        )
        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label, "zones": zone}
            for t2, adc, dwi, label, zone in zip(
                self.groupby.get_group(partition)["filepath_t2w_" + self.format_load].values,
                self.groupby.get_group(partition)["filepath_adc_" + self.format_load].values,
                self.groupby.get_group(partition)["filepath_hbv_" + self.format_load].values,
                one_hot(as_tensor(self.groupby.get_group(partition)["label"].values)).float(),
                self.groupby.get_group(partition)["filepath_seg_zones_cropped"].values,
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


class ProstateMultimodalDataLoader(ProstateClassDataLoader):
    def __init__(
        self,
        path: str,
        sep: str = ",",
        classes: list = ["noCsPCa", "CsPCa"],
        format_load: str = "cropped",
        img_columns=["t2", "adc", "dwi"],
        test_run: bool = False,
        input_shape: str = "(128, 128, 32)",
        rand_prob: int = 0.5,
        config: dict = config_default,
    ):
        super().__init__(path, sep, classes, format_load, img_columns, test_run, input_shape, rand_prob, config)
        self.clinical_cols = ["patient_age", "psa", "psad", "prostate_volume"]
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=img_columns + ["zones"], reader="NibabelReader", image_only=True),
                transforms.EnsureChannelFirstd(keys=img_columns + ["zones"]),
                transforms.AsDiscreted(keys="zones", argmax=False, to_onehot=3),
                transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
                transforms.ResampleToMatchd(
                    keys=["adc", "dwi", "zones"],
                    key_dst="t2",
                    mode=("bilinear", "bilinear", "nearest"),
                ),  # Resample images to t2 dimension
                transforms.Resized(
                    keys=img_columns + ["zones"],
                    spatial_size=eval(input_shape),
                    mode=("trilinear", "trilinear", "trilinear", "nearest"),
                ),  # SAMUNETR: Reshape to have the same dimension
                transforms.ScaleIntensityd(keys=img_columns, minv=0.0, maxv=1.0),
                transforms.NormalizeIntensityd(keys=img_columns),
                transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
                transforms.ToTensord(keys=["numeric"]),
                # transforms.RandCropByPosNegLabeld(keys=["image"], label_key="zones", spatial_size=[96, 96, 32],num_samples=4,image_key="image",image_threshold=0)
                # transforms.RandSpatialCropSamplesd(keys=['image','label'],roi_size=[96,96,-1],num_samples=8,random_size=False),#For the other models
                # transforms.RandRotate90d(keys=['image'],spatial_axes=0,prob=prob),
                # transforms.RandZoomd(keys=['image'],min_zoom=0.9,max_zoom=1.1,mode='area',prob=prob),
                # transforms.RandGaussianNoised(keys=["image"],mean=0.1,std=0.25,prob=prob),
                # transforms.RandShiftIntensityd(keys=["image"],offsets=0.2,prob=prob),
                # transforms.RandGaussianSharpend(keys=['image'],sigma1_x=[0.5, 1.0],sigma1_y=[0.5, 1.0],sigma1_z=[0.5, 1.0],sigma2_x=[0.5, 1.0],sigma2_y=[0.5, 1.0],sigma2_z=[0.5, 1.0],alpha=[10.0,30.0],prob=prob),
                # transforms.RandAdjustContrastd(keys=['image'],gamma=2.0,prob=prob),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=img_columns + ["zones"], image_only=True),
                transforms.EnsureChannelFirstd(keys=img_columns + ["zones"]),
                transforms.AsDiscreted(keys="zones", argmax=True, to_onehot=3),
                transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
                transforms.ResampleToMatchd(
                    keys=["adc", "dwi", "zones"],
                    key_dst="t2",
                    mode=("bilinear", "bilinear", "nearest"),
                ),  # Resample images to t2 dimensions
                transforms.Resized(
                    keys=img_columns + ["zones"],
                    spatial_size=eval(input_shape),
                    mode=("trilinear", "trilinear", "trilinear", "nearest"),
                ),  # SAMUNETR: Reshape to have the same dimension
                transforms.ScaleIntensityd(keys=img_columns, minv=0.0, maxv=1.0),
                transforms.NormalizeIntensityd(keys=img_columns),
                transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
                transforms.ToTensord(keys=["numeric"]),
            ]
        )
        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        clinical_variables = array(self.groupby.get_group(partition)[self.clinical_cols].values, dtype=float32)
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label, "zones": zone, "numeric": clinical}
            for t2, adc, dwi, label, zone, clinical in zip(
                self.groupby.get_group(partition)["filepath_t2w_" + self.format_load].values,
                self.groupby.get_group(partition)["filepath_adc_" + self.format_load].values,
                self.groupby.get_group(partition)["filepath_hbv_" + self.format_load].values,
                one_hot(as_tensor(self.groupby.get_group(partition)["label"].values)).float(),
                self.groupby.get_group(partition)["filepath_seg_zones_cropped"].values,
                clinical_variables,
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
