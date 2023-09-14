from monai.transforms import MapTransform
from numpy import mean, std


class DeleteBlackSlices(MapTransform):
    """
    This transforms deletes black slices from an image based on the image pixels mean and std.
    """

    def __init__(self, keys, threshold: float = 0.5):
        MapTransform.__init__(self, keys)
        self.threshold = threshold

    def __call__(self, x):
        key = self.keys[0]
        data = x[key]
        img_std = std(data, keepdims=False)
        std_per_slice = std(data, axis=(0, 1, 2), keepdims=False)
        img_mean = mean(data, keepdims=False)
        mean_per_slice = mean(data, axis=(0, 1, 2), keepdims=False)
        mask = (std_per_slice > self.threshold * img_std) & (mean_per_slice > self.threshold * img_mean)
        x[key] = data[:, :, :, mask]
        return x


class Labels3Dto2D(MapTransform):
    """
    This transforms recieves a 3D image and its label, and returns a list with all 2D slices
    and a copy of the original label.

    train/dev mode assumes the dataset contains labels, test mode does not
    """

    def __init__(self, keys, mode: str = "train/dev"):
        MapTransform.__init__(self, keys)
        self.mode = mode

    def __call__(self, x):
        img = x[self.keys[0]]
        if self.mode == "train/dev":
            lbl = x[self.keys[1]]
            return [{"image": slice, "label": lbl} for slice in img]
        elif self.mode == "test":
            return [{"image": slice} for slice in img]
