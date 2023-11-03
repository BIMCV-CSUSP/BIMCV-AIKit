import argparse
import collections
import importlib
import json

import numpy as np
import torch

import bimcv_aikit.dataloaders as data_loader_module
import trainer as module_trainer
from parse_config import ConfigParser
from utils import prepare_device

# fix random seeds for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    SEED = config["seed"] if config["seed"] else torch.ranint(1, 10000)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", data_loader_module)

    # build model architecture, then print to console
    module_arch = importlib.import_module(config["arch"]["module"])
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = config.init_obj("loss", importlib.import_module(config["loss"]["module"]), **{"weight": torch.tensor(data_loader.class_weights).to(device)})
    metrics = {name: getattr(importlib.import_module(met["module"]), met["type"])(**met["args"]) for name, met in config["metrics"].items()}

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer) if config["lr_scheduler"] else None

    train_loader = data_loader(config["data_loader"]["partitions"]["train"])
    valid_loader = data_loader(config["data_loader"]["partitions"]["val"])

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

    train_predictions, train_results = trainer.evaluate(train_loader)
    if valid_loader:
        val_predictions, val_results = trainer.evaluate(valid_loader)

    del train_loader#, valid_loader

    test_loader = data_loader(config["data_loader"]["partitions"]["test"])
    if test_loader:
        test_predictions, test_results = trainer.evaluate(test_loader)

    results = {
        "Train Metrics": train_results,
        "Val Metrics": val_results if valid_loader else None,
        "Test Metrics": test_results if test_loader else None,
        "Train Predictions": train_predictions.tolist(),
        "Val Predictions": val_predictions.tolist() if valid_loader else None,
        "Test Predictions": test_predictions.tolist() if test_loader else None,
    }
    with open(f"{config.log_dir}/results.json", "w") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
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
    main(config)
