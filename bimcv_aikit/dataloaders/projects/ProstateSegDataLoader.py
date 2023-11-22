import nibabel as nib
from numpy import unique
from pandas import DataFrame, read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

from bimcv_aikit.monai.transforms import DeleteBlackSlices
from monai import transforms
from monai.data import CacheDataset, DataLoader

config_default = {}


class ProstateSegDataLoader:
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

        data_picai = df

        data_picai_human = data_picai[data_picai["human_labeled"] == 1]
        data_picai.drop(data_picai_human.index, inplace=True)

        data_picai = data_picai[
            [
                "filepath_t2w_" + self.format_load,
                "filepath_adc_" + self.format_load,
                "filepath_hbv_" + self.format_load,
                "filepath_labelAI_cropped",
                "filepath_seg_zones_cropped",
                "partition",
            ]
        ]
        data_picai_human = data_picai_human[
            [
                "filepath_t2w_" + self.format_load,
                "filepath_adc_" + self.format_load,
                "filepath_hbv_" + self.format_load,
                "filepath_label_cropped",
                "filepath_seg_zones_cropped",
                "partition",
            ]
        ]

        df = DataFrame(
            {
                "t2w": list(data_picai["filepath_t2w_" + self.format_load].values)
                + list(data_picai_human["filepath_t2w_" + self.format_load].values),
                "adc": list(data_picai["filepath_adc_" + self.format_load].values)
                + list(data_picai_human["filepath_adc_" + self.format_load].values),
                "dwi": list(data_picai["filepath_hbv_" + self.format_load].values)
                + list(data_picai_human["filepath_hbv_" + self.format_load].values),
                "zones": list(data_picai["filepath_seg_zones_cropped"].values) + list(data_picai_human["filepath_seg_zones_cropped"].values),
                "label": list(data_picai["filepath_labelAI_cropped"].values) + list(data_picai_human["filepath_label_cropped"].values),
                "partition": list(data_picai["partition"].values) + list(data_picai_human["partition"].values),
            }
        )

        n_classes = len(unique(df["label"].values))

        self.groupby = df.groupby("partition")

        self._class_weights = None
        label_column = ["label"]
        prob = rand_prob
        mode = ["bilinear", "nearest"]
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=img_columns + label_column + ["zones"], reader="NibabelReader", image_only=True),
                transforms.AsDiscreted(keys=label_column, threshold=1),  # Convert values greater than 1 to 1
                transforms.EnsureChannelFirstd(keys=img_columns + label_column + ["zones"]),
                transforms.AsDiscreted(keys="zones", argmax=False, to_onehot=3),
                transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
                transforms.ResampleToMatchd(
                    keys=["adc", "dwi", "zones", "label"], key_dst="t2", mode=("bilinear", "bilinear", "nearest", "nearest")
                ),  # Resample images to t2 dimension
                transforms.Resized(
                    keys=img_columns + label_column + ["zones"],
                    spatial_size=eval(input_shape),
                    mode=("trilinear", "trilinear", "trilinear", "nearest", "nearest"),
                ),  # SAMUNETR: Reshape to have the same dimension
                # transforms.ScaleIntensityd(keys=img_columns,minv=0.0, maxv=1.0),
                transforms.NormalizeIntensityd(keys=img_columns),
                transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
                transforms.ConcatItemsd(keys=label_column, name="label", dim=0),
                # transforms.RandSpatialCropSamplesd(keys=['image','label'],roi_size=[96,96,-1],num_samples=8,random_size=False),#For the other models
                transforms.RandRotate90d(keys=["image", "label"], spatial_axes=[0, 1], prob=prob),
                # transforms.RandZoomd(keys=['image','label'],min_zoom=0.9,max_zoom=1.1,mode=['area' if x == 'bilinear' else x for x in mode],prob=prob),
                transforms.RandGaussianNoised(keys=["image"], mean=0.1, std=0.25, prob=prob),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.2, prob=prob),
                transforms.RandGaussianSharpend(
                    keys=["image"],
                    sigma1_x=[0.5, 1.0],
                    sigma1_y=[0.5, 1.0],
                    sigma1_z=[0.5, 1.0],
                    sigma2_x=[0.5, 1.0],
                    sigma2_y=[0.5, 1.0],
                    sigma2_z=[0.5, 1.0],
                    alpha=[10.0, 30.0],
                    prob=prob,
                ),
                transforms.RandAdjustContrastd(keys=["image"], gamma=2.0, prob=prob),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=img_columns + label_column + ["zones"], image_only=True),
                transforms.AsDiscreted(keys=label_column, threshold=1),  # Convert values greater than 1 to 1
                transforms.EnsureChannelFirstd(keys=img_columns + label_column + ["zones"]),
                transforms.AsDiscreted(keys="zones", argmax=True, to_onehot=3),
                transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
                transforms.ResampleToMatchd(
                    keys=["adc", "dwi", "zones", "label"], key_dst="t2", mode=("bilinear", "bilinear", "nearest", "nearest")
                ),  # Resample images to t2 dimensions
                transforms.Resized(
                    keys=img_columns + label_column + ["zones"],
                    spatial_size=eval(input_shape),
                    mode=("trilinear", "trilinear", "trilinear", "nearest", "nearest"),
                ),  # SAMUNETR: Reshape to have the same dimension
                # transforms.ScaleIntensityd(keys=img_columns,minv=0.0, maxv=1.0),
                transforms.NormalizeIntensityd(keys=img_columns),
                transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
                transforms.ConcatItemsd(keys=label_column, name="label", dim=0),
            ]
        )
        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label, "zones": zone}
            for t2, adc, dwi, label, zone in zip(
                self.groupby.get_group(partition)["t2w"].values,
                self.groupby.get_group(partition)["adc"].values,
                self.groupby.get_group(partition)["dwi"].values,
                self.groupby.get_group(partition)["label"].values,
                self.groupby.get_group(partition)["zones"].values,
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
