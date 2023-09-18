from monai.transforms import MapTransform
from numpy import ix_, mean, std


class DeleteBlackSlices(MapTransform):
    """
    This transforms deletes black slices from an image based on the image pixels mean and std.

    Input image is expected to be of shape (channels x spatial_dim1 x spatial_dim2 x spatial_dim3)
    """

    def __init__(self, keys, threshold: float = 0.5):
        MapTransform.__init__(self, keys)
        self.threshold = threshold

    def __call__(self, x):
        key = self.keys[0]
        data = x[key]

        img_std = std(data, keepdims=False)
        std_per_slice_axis1 = std(data, axis=(0, 2, 3), keepdims=False)
        std_per_slice_axis2 = std(data, axis=(0, 1, 3), keepdims=False)
        std_per_slice_axis3 = std(data, axis=(0, 1, 2), keepdims=False)

        img_mean = mean(data, keepdims=False)
        mean_per_slice_axis1 = mean(data, axis=(0, 2, 3), keepdims=False)
        mean_per_slice_axis2 = mean(data, axis=(0, 1, 3), keepdims=False)
        mean_per_slice_axis3 = mean(data, axis=(0, 1, 2), keepdims=False)

        mask_axis1 = (std_per_slice_axis1 > self.threshold * img_std) & (mean_per_slice_axis1 > self.threshold * img_mean)
        mask_axis2 = (std_per_slice_axis2 > self.threshold * img_std) & (mean_per_slice_axis2 > self.threshold * img_mean)
        mask_axis3 = (std_per_slice_axis3 > self.threshold * img_std) & (mean_per_slice_axis3 > self.threshold * img_mean)
        data = data[:, mask_axis1, :, :]
        data = data[:,:, mask_axis2, :]
        x[key] = data[:, :, :, mask_axis3]
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
