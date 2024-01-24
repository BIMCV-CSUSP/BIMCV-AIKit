# About DataLoaders

## BaseDataLoader

`BaseDataLoader` is an abstract base class for data loaders. It cannot be instantiated itself and is designed to be subclassed by other data loader classes that implement the specific details of how data should be loaded for different types of datasets.

### Methods

- `__init__(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 1, collate_fn: Callable = default_collate, **kwargs)`: Initializes the object with several parameters such as `batch_size`, `shuffle`, `num_workers`, and `collate_fn`. These parameters are commonly used in PyTorch's DataLoader for controlling the data loading process.

- `__call__(self, partition: str)`: An abstract method that must be implemented by any non-abstract child class. It's designed to return a data loader for a given partition of the data.

- `init_transforms(transforms_config: dict)`: A static method used to initialize data transformations from a configuration dictionary.

## BaseClassificationDataLoader

`BaseClassificationDataLoader` is a subclass of `BaseDataLoader`. It adds a `class_weights` property, which is to be used to handle imbalanced classes in a classification task.

## Usage example

- ### MnistDataLoader

    [`MnistDataLoader`](./MnistDataLoader.py) is a working subclass of `BaseClassificationDataLoader`. It's used to load the MNIST dataset.

    - `__init__(self, data_dir: str, transforms: dict = {}, batch_size: int = 16, shuffle=False, num_workers=1)`: Initializes an instance of the class. It calls the `__init__` method of the superclass with the batch size, whether to shuffle the data, and the number of worker processes. It also initializes the class weights, the data directory, the transformations, and the training dataset.

    - `__call__(self, partition: str)`: Gets a data loader for a given partition of the data. If the partition is "train", it returns a data loader for the training dataset. If the partition is "test", it creates a testing dataset and returns a data loader for it. If the partition is neither "train" nor "test", it returns None.
