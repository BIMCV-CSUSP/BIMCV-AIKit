# BIMCV code utils

This repository contains useful methods to handle medical images in Python, and implementations of Deep Learning models used by the [Medical Imaging Databank of the Valencia Region](https://bimcv.cipf.es/) team.

## Modules

- [Data](/bimcv_aikit/data/): methods to handle datasets for developing DL models
- [Dataloaders](/bimcv_aikit/dataloaders/): PyTorch dataloader classes used by the training module
- [Loss Functions](/bimcv_aikit/loss_functions/): loss function definitions used by the training module
- [Medical Imaging](/bimcv_aikit/medical_imaging/): methods to transform or extract relevant information from medical imaging formats (DICOM, NIFTI, ...)
- [Metrics](/bimcv_aikit/metrics/): metric definitions used by the training module
- [Models](/bimcv_aikit/models/): model definitions used by the training module
- [MONAI](/bimcv_aikit/monai/): utils generated on top of the [MONAI](https://github.com/Project-MONAI/MONAI) framework
- [PyTorch](/bimcv_aikit/pytorch/): utils generated on top of the PyTorch framework
- [Training](/bimcv_aikit/training/): core module of the package, used to handle and log DL experiments. See [usage instructions](/bimcv_aikit/training/README.md) for more details

## Installation

There are three options:

- Clone the repository and add the folder to the `PYTHONPATH` (recommended if you wish to contribute, or develop on top of the existing methods):

```bash
git clone https://github.com/BIMCV-CSUSP/BIMCV-AIKit.git
export PYTHONPATH="<PATH>/BIMCV-AIKit:$PYTHONPATH"
```

Replacing `<PATH>` with the proper value to generate a global path to the `BIMCV-AIKit` folder. The configuration of the `PYTHONPATH` may vary for your system.

- Clone the repository, and install from source as a pip package:

```bash
git clone https://github.com/BIMCV-CSUSP/BIMCV-AIKit.git
cd BIMCV-AIKit
pip install -e .
```

- Install as a standalone pip package using:

```bash
pip install git+https://github.com/BIMCV-CSUSP/BIMCV-AIKit.git#egg=bimcv_aikit
```

### Dependencies

The current release depends on the following Python libraries:

- monai == 1.2.0
- numpy == 1.23.4
- pandas == 2.1.0
- prettytable == 3.9.0
- ptflops == 0.7
- pygad == 3.2.0
- tensorboard == 2.8.0
- torch == 1.12.1
- torchmetrics == 1.1.2
- torchvision == 0.13.1
- tqdm == 4.62.3

Install all dependencies using:

```bash
pip install git+https://github.com/BIMCV-CSUSP/BIMCV-AIKit.git#egg=bimcv_aikit[all]
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

- [Jesús Alejandro Alzate-Grisales](https://github.com/jesusalzate) - Data Scientist
- [Alejandro Mora-Rubio](https://github.com/MoraRubio) - Data Scientist
