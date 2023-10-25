from setuptools import find_packages, setup

from bimcv_aikit import __version__

setup(
    name='bimcv_aikit',
    version=__version__,
    url='https://github.com/BIMCV-CSUSP/BIMCV-AIKit',
    py_modules=find_packages(),
)