import argparse
import collections
import importlib
import json
from functools import partial

import numpy as np
import torch
from monai.transforms import Compose

from .. import dataloaders as data_loader_module
from ..metrics.BaseMetric import BaseMetric
from ..metrics.segmentation.metrics_segmentation import metrics_segmentation_constructor_monai
from . import trainer as module_trainer
from .parse_config import ConfigParser
from .utils import prepare_device

# fix random seeds for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
    ]
    config = ConfigParser.from_args(args, options)

    SEED = config["seed"] if config["seed"] else torch.ranint(1, 10000)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader")

    # build model architecture, then print to console
    model = config.init_obj("arch")
    logger.debug(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    if data_loader.class_weights is None:
        criterion = config.init_obj("loss")
    else:
        criterion = config.init_obj("loss", **{"weight": torch.tensor(data_loader.class_weights).to(device)})

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer) if config["lr_scheduler"] else None

    train_loader = data_loader(config["data_loader"]["partitions"]["train"])
    valid_loader = data_loader(config["data_loader"]["partitions"]["val"])

    metrics = {}
    for name, met in config["metrics"].items():
        metric = partial(getattr(importlib.import_module(met["module"]), met["type"]), **met["args"])

        if "monai" in met["module"]:
            metrics[name] = metrics_segmentation_constructor_monai(original_metric=metric)
        else:
            metrics[name] = BaseMetric(metric)

    Trainer = getattr(module_trainer, config["trainer"]["type"])
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        train_data_loader=train_loader,
        valid_data_loader=valid_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()

    results = {}

    train_predictions, train_results = trainer.evaluate(train_loader)
    results["Train Metrics"] = train_results
    results["Train Predictions"] = train_predictions.tolist()

    if valid_loader:
        val_predictions, val_results = trainer.evaluate(valid_loader)
        results["Validation Metrics"] = val_results
        results["Validation Predictions"] = val_predictions.tolist()

    del train_loader, valid_loader

    test_loader = data_loader(config["data_loader"]["partitions"]["test"])
    if test_loader:
        test_predictions, test_results = trainer.evaluate(test_loader)
        results["Test Metrics"] = test_results
        results["Test Predictions"] = test_predictions.tolist()

    with open(f"{config.log_dir}/results.json", "w") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
