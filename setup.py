from setuptools import find_packages, setup

from bimcv_aikit import __version__

setup(
    name="bimcv_aikit",
    version=__version__,
    url="https://github.com/BIMCV-CSUSP/BIMCV-AIKit",
    packages=find_packages(),
    extras_require={
        "all": [
            "monai == 1.2.0",
            "numpy == 1.23.4",
            "pandas == 2.1.0",
            "prettytable == 3.9.0",
            "ptflops == 0.7",
            "pygad == 3.2.0",
            "tensorboard == 2.8.0",
            "torch == 1.12.1",
            "torchmetrics == 1.1.2",
            "torchvision == 0.13.1",
            "tqdm == 4.62.3",
        ]
    },
    entry_points={
        "console_scripts": [
            "bimcv_evaluate=bimcv_aikit.training.evaluate:main",
            "bimcv_train=bimcv_aikit.training.train:main",
            "bimcv_crossval=bimcv_aikit.training.crossval:main",
        ]
    },
)
