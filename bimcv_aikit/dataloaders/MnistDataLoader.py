from torchvision import datasets, transforms

from .base_data_loader import BaseDataLoader  # At the moment this line is only for testing purposes


class MNIST(datasets.MNIST):
    def __init__(self, root, train, transform, target_transform=None, download=False) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        return {"image": item[0], "label": item[1]}


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.data_dir = data_dir
        self.dataset = MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    @property
    def class_weights(self):
        """
        Returns the class weights for the dataset.
        """
        return [1.0] * 10
