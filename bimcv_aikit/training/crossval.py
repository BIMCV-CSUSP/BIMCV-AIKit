import argparse
import importlib
import json
import types
from functools import partial

import numpy as np
import torch
from prettytable import PrettyTable

from ..metrics.BaseMetric import BaseMetric
from . import trainer as module_trainer
from .parse_config import ConfigParser, CustomArgs
from .utils import prepare_device

# fix random seeds for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config: ConfigParser):
    SEED = config["seed"] if config["seed"] else torch.ranint(1, 10000)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger = config.get_logger("train")

    results = {}

    for i, fold in enumerate(config["data_loader"]["partitions"]["folds"]):
        logger.info(f"{'-' * 20}\nStarting fold {i}\n{'-' * 20}")

        # setup data_loader instances
        data_loader = config.init_obj(
            "data_loader", **{config["data_loader"]["partitions"]["crossval_arg"]: fold}
        )

        # build model architecture, then print to console
        module_arch = importlib.import_module(config["arch"]["module"])
        model = config.init_obj("arch", module_arch)
        if i == 0:
            logger.debug(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config["n_gpu"])
        model = model.to(device)  # type: ignore
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        if data_loader.class_weights is None:  # type: ignore
            criterion = config.init_obj("loss")
        else:
            criterion = config.init_obj(
                "loss", **{"weight": torch.tensor(data_loader.class_weights).to(device)}  # type: ignore
            )

        metrics = {}
        for name, met in config["metrics"].items():
            element = getattr(importlib.import_module(met["module"]), met["type"])
            if isinstance(element, types.FunctionType):
                metric = partial(
                    getattr(importlib.import_module(met["module"]), met["type"]),
                    **met["args"],
                )
            else:
                metric = element(**met["args"])
            metrics[name] = BaseMetric(metric)

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
        lr_scheduler = (
            config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)
            if config["lr_scheduler"]
            else None
        )

        train_loader = data_loader(config["data_loader"]["partitions"]["train"])  # type: ignore

        Trainer = getattr(module_trainer, config["trainer"]["type"])
        trainer = Trainer(
            model,
            criterion,
            metrics,
            optimizer,
            fold=fold,
            config=config,
            device=device,
            train_data_loader=train_loader,
            valid_data_loader=None,
            lr_scheduler=lr_scheduler,
        )

        trainer.train()

        results = {fold: {}}
        train_predictions, train_results = trainer.evaluate(train_loader)
        results[fold] = {
            "Train Metrics": train_results,
            "Train Predictions": train_predictions.tolist(),
        }
        del train_loader

        test_loader = data_loader(config["data_loader"]["partitions"]["test"])  # type: ignore
        if test_loader:
            test_predictions, test_results = trainer.evaluate(test_loader)
            results[fold].update(
                {
                    "Test Metrics": test_results,
                    "Test Predictions": test_predictions.tolist(),
                }
            )

    train_metrics: dict[str, list] = {}
    test_metrics: dict[str, list] = {}
    for fold, fold_results in results.items():
        for metric, value in fold_results["Train Metrics"].items():
            if not train_metrics.get(metric, False):
                train_metrics[metric] = []
            train_metrics[metric].append(value)
        for metric, value in fold_results["Test Metrics"].items():
            if not test_metrics.get(metric, False):
                test_metrics[metric] = []
            test_metrics[metric].append(value)
    results["Aggregates"] = {
        "Train Metrics": {
            metric: {"mean": np.mean(values), "std": np.std(values)}
            for metric, values in train_metrics.items()
        },
        "Test Metrics": {
            metric: {"mean": np.mean(values), "std": np.std(values)}
            for metric, values in test_metrics.items()
        },
    }
    with open(f"{config.log_dir}/cross_val_results.json", "w") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    table = PrettyTable()
    table.title = "Final Performance Metrics"
    metrics = list(results["Aggregates"]["Train Metrics"].keys())
    table.add_column("Metrics", metrics)
    train_values = [
        f"{value['mean']:.4f} +/- {value['std']:.4f}"
        for _, value in results["Aggregates"]["Train Metrics"].items()
    ]
    table.add_column("Train", train_values)
    test_values = [
        f"{value['mean']:.4f} +/- {value['std']:.4f}"
        for _, value in results["Aggregates"]["Test Metrics"].items()
    ]
    table.add_column("Test", test_values)
    logger.info(table)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch model cross-validation script")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
