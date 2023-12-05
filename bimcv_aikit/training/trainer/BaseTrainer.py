import pathlib
import signal
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Union

import torch
from numpy import inf
from prettytable import PrettyTable

from ...utils.config import init_obj
from ..logger import TensorboardWriter
from ..parse_config import ConfigParser


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metric_ftns: dict,
        optimizer: torch.optim.Optimizer,
        config: ConfigParser,
        device: torch.device,
        lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None],
        fold: str = "",
    ):
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"] if cfg_trainer["save_period"] else (self.epochs + 1)
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir / fold if fold else config.save_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer["tensorboard"])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.killer = GracefulKiller()

    @abstractmethod
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> tuple:
        """
        Evaluates the PyTorch model using the given data loader.

        :param data_loader: torch.utils.data.DataLoader, the PyTorch DataLoader object to use for evaluation.
        :return: A tuple containing the predicted values (torch.Tensor) and the computed metrics (dict).
        """
        return NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        :param epoch: Current epoch number.
        :return: A dictionary containing the computed metrics for that epoch.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            table = PrettyTable()
            table.title = f"Performance epoch {epoch}"
            metrics = list(self.metric_ftns.keys()) + ["loss"]
            train_values = [f"{value:.4f}" for key, value in log.items() if ("val" not in key and key != "epoch")]
            val_values = [f"{value:.4f}" for key, value in log.items() if ("val" in key and key != "epoch")]
            table.add_column("Metrics", metrics)
            table.add_column("Train", train_values)
            if len(val_values) > 0:
                table.add_column(" Validation", val_values)
            self.logger.info(table)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {} epochs. " "Training stops.".format(self.early_stop))
                    break

            self._save_checkpoint(epoch, save_best=best)

            if self.killer.kill_now:
                self.logger.info(f"Terminating training at epoch {epoch} after receiving SIGTERM or SIGINT...")
                break

    def _save_checkpoint(self, epoch: int, save_best: bool = False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            # "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        weights = {"state_dict": self.model.state_dict()}
        if epoch % self.save_period == 0:
            torch.save(state, str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth"))
            torch.save(weights, str(self.checkpoint_dir / f"model-weights-epoch{epoch}.pth"))
            self.logger.info(f"Saving checkpoint epoch {epoch} ...")
        if save_best:
            torch.save(state, str(self.checkpoint_dir / "best-checkpoint.pth"))
            torch.save(weights, str(self.checkpoint_dir / "best-model-weights.pth"))
            self.logger.info("Saving current best: best-model-weights.pth ...")

    def _resume_checkpoint(self, resume_path: pathlib.Path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = pathlib.Path(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(str(resume_path))
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            raise ValueError("Architecture configuration given in config file is different from that of checkpoint.")
        weights_path = resume_path.parent.joinpath(resume_path.stem.replace("checkpoint", "model-weights"))
        if not str(weights_path).endswith(".pth"):
            weights_path = weights_path.with_suffix(".pth")
        print(weights_path)
        self.model.load_state_dict(torch.load(str(weights_path))["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            self.logger.warning("Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    @staticmethod
    def _init_transforms(transforms_config: Union[dict, list]) -> Union[Any, dict]:
        """
        Initializes the transforms from a configuration dictionary.
        """
        if not transforms_config:
            return {}
        if isinstance(transforms_config, list):
            try:
                transform_list = [
                    init_obj(transform["module"], transform["type"], **transform["args"]) for transform in transforms_config["args"]["transforms"]
                ]
            except Exception as e:
                print(f"Error defining transforms")
                raise e
            transforms_config["args"]["transforms"] = transform_list
            return init_obj(transforms_config["module"], transforms_config["type"], **transforms_config["args"])
        transforms = {}
        for partition, transform_config in transforms_config.items():
            if not transform_config:
                transforms[partition] = None
                continue
            try:
                transform_list = [
                    init_obj(transform["module"], transform["type"], **transform["args"]) for transform in transform_config["args"]["transforms"]
                ]
            except Exception as e:
                print(f"Error defining transforms for {partition} partition")
                raise e
            transform_config["args"]["transforms"] = transform_list
            transforms[partition] = init_obj(transform_config["module"], transform_config["type"], **transform_config["args"])
        return transforms


class GracefulKiller:
    """
    A class that allows for graceful termination of a program.

    Attributes:
        kill_now (bool): Flag indicating whether the program should be terminated.
        cont (int): Counter for the number of times the exit_gracefully method has been called.
    """

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.cont = 0

    def exit_gracefully(self, *args):
        """
        Method to handle the graceful termination of the program.

        Args:
            *args: Variable number of arguments passed to the signal handler.
        """
        self.kill_now = True
        self.cont += 1
        if self.cont > 1:
            raise KeyboardInterrupt
