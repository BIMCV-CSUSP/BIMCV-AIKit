from numpy import array, float32, unique
from pandas import read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

from monai.data import CacheDataset, DataLoader
from monai import transforms
from bimcv_aikit.monai.transforms import DeleteBlackSlices


class ADNIDataLoader:
    """
    A data loader for the ADNI dataset.
    """

    def __init__(self, path: str, sep: str = ",", classes: list = ["CN", "AD"], map_labels_dict: dict = None, test_run: bool = False, input_shape: str = "(96,96,96)", config: dict = {}):
        """
        Initializes an instance of the ADNIDataLoader class.

        Args:
            path (str): Path to the CSV file containing the data.
            sep (str, optional): Separator used in the CSV file. Defaults to ",".
            classes (list, optional): List of classes to include in the data. Defaults to ["CN", "AD"].
            map_labels_dict (dict, optional): Dictionary mapping class names to integer labels. If provided, only the classes in the dictionary will be included in the data. Defaults to None.
            test_run (bool, optional): If True, only a small subset of the data will be loaded for testing purposes. Defaults to False.
            input_shape (str, optional): Spatial size of the input images. Defaults to "(96,96,96)".
            config (dict, optional): Additional configuration options. Defaults to {}.
        """
        df = read_csv(path, sep=sep)
        if map_labels_dict:
            map_labels = map_labels_dict
            df = df.loc[df["Research Group"].isin(list(map_labels.keys()))]
        elif classes:
            df = df.loc[df["Research Group"].isin(classes)]
            map_labels = {class_: i for i, class_ in enumerate(classes)}
        df["intLabel"] = df["Research Group"].map(map_labels)
        n_classes = len(unique(df["intLabel"].values))
        onehot = lambda x: one_hot(as_tensor(x), num_classes=n_classes).float()
        df["onehot"] = df["intLabel"].apply(onehot)
        self.groupby = df.groupby("Partition")
        self._class_weights = compute_class_weight(
            class_weight="balanced", 
            classes=unique(self.groupby.get_group("train")["intLabel"].values), 
            y=self.groupby.get_group("train")["intLabel"].values
        )
        self.transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], reader="NibabelReader", ensure_channel_first=True, image_only=False),
                transforms.ToTensord(keys=["image"]),
                transforms.NormalizeIntensityd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"]),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                DeleteBlackSlices(keys=["image"], threshold=0.5),
                transforms.Resized(keys=["image"], spatial_size=eval(input_shape)),
            ]
        )
        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
            """
            Returns a DataLoader object for the specified partition.

            Args:
                partition (str): The partition to load data for (e.g. "train", "val", or "test").

            Returns:
                DataLoader: A PyTorch DataLoader object containing the specified partition's data.
            """
            data = [
                {"image": img_path, "label": label}
                for img_path, label in zip(self.groupby.get_group(partition)["Path"].values, self.groupby.get_group(partition)["onehot"].values)
            ]
            if self.test_run:
                data = data[:16]
            dataset = CacheDataset(data=data, transform=self.transforms, num_workers=7)
            return DataLoader(dataset, **self.config_args)

    @property
    def class_weights(self):
        """
        Returns the class weights for the dataset.
        """
        return self._class_weights
    
class ADNIMultimodalDataLoader(ADNIDataLoader):
    """
    A data loader for the ADNI dataset that loads both MRI images and clinical variables.
    """

    def __init__(self, path: str, sep: str = ",", classes: list = ["CN", "AD"], map_labels_dict: dict = None, test_run: bool = False, input_shape: str = "(96,96,96)", config: dict = {}):
        """
        Initializes an instance of the ADNIDataLoader class.

        Args:
            path (str): The path to the data file.
            sep (str, optional): The separator used in the data file. Defaults to ",".
            classes (list, optional): A list of class labels. Defaults to ["CN", "AD"].
            map_labels_dict (dict, optional): A dictionary mapping original labels to new labels. Defaults to None.
            test_run (bool, optional): Whether to run the data loader in test mode. Defaults to False.
            input_shape (str, optional): The shape of the input data. Defaults to "(96,96,96)".
            config (dict, optional): A dictionary of configuration options. Defaults to {}.
        """
        super().__init__(path, sep, classes, map_labels_dict, test_run, input_shape, config)
        self.clinical_cols = ["PTGENDER","APOE4","Ventricles","Hippocampus","WholeBrain","Entorhinal","Fusiform","MidTemp","ICV","AGE","CDRSB","ADAS11","ADAS13","ADASQ4","MMSE","RAVLT_immediate","RAVLT_learning","RAVLT_forgetting","RAVLT_perc_forgetting","FAQ","MOCA","LDELTOTAL","TRABSCOR","mPACCdigit","mPACCtrailsB"]
        self.transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], reader="NibabelReader", ensure_channel_first=True, image_only=False),
                transforms.ToTensord(keys=["image", "numeric"]),
                transforms.NormalizeIntensityd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"]),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                DeleteBlackSlices(keys=["image"], threshold=0.5),
                transforms.Resized(keys=["image"], spatial_size=eval(input_shape)),
            ]
        )

    def __call__(self, partition: str):
            """
            Returns a DataLoader object for the specified partition.

            Args:
                partition (str): The partition to load data for (either "train", "val", or "test").

            Returns:
                DataLoader: A PyTorch DataLoader object containing the specified partition's data.
            """
            paths = self.groupby.get_group(partition)["Path"].values
            labels = self.groupby.get_group(partition)["onehot"].values
            clinical_variables = array(self.groupby.get_group(partition)[self.clinical_cols].values, dtype=float32)
            data = [
                {"image": img_path, "label": label, "numeric": clinical}
                for img_path, label, clinical in zip(paths, labels, clinical_variables)
            ]
            if self.test_run:
                data = data[:16]
            dataset = CacheDataset(data=data, transform=self.transforms, num_workers=7)
            return DataLoader(dataset, **self.config_args)
