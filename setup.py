from setuptools import find_packages, setup

from bimcv_aikit import __version__

setup(
    name='bimcv_aikit',
    version=__version__,
    url='https://github.com/BIMCV-CSUSP/BIMCV-AIKit',
    py_modules=find_packages(),
    extras_require={"all": ["monai == 1.2.0", 
                            "numpy == 1.26.0", 
                            "pandas == 2.1.0", 
                            "prettytable == 3.9.0",
                            "ptflops == 0.7", 
                            "pygad == 3.2.0", 
                            "torch == 1.12.1", 
                            "torchmetrics == 1.1.2", 
                            "torchvision == 0.13.1",
                            "tqdm == 4.62.3",]},
)