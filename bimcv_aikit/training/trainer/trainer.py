from time import sleep

import numpy as np
import torch
from torch.nn.functional import softmax
from torchvision.utils import make_grid
from tqdm import tqdm

from base import BaseTrainer
from utils import MetricTracker, inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self, model, criterion, metric_ftns, optimizer, config, device, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _aggregate_metrics_per_epoch(self, stage, epoch):
        if not self.metric_ftns:
            return {}, []
        metrics_dict = {}
        values = []
        for name, metric_fct in self.metric_ftns.items():
            metrics_dict[name] = metric_fct.compute()
            self.writer.add_scalar(f"{name}/{stage.lower()}", metrics_dict[name], epoch)
            metric_fct.reset()
            values.append(f"{metrics_dict[name]:.4f}")
        return metrics_dict

    def _compute_metrics(self, predictions, labels):
        if not self.metric_ftns:
            return {}
        metrics_dict = {}
        if any(predictions.sum(dim=1) != 1.0):  # if predictions are not probabilities
            predictions = softmax(predictions, dim=1)
        if len(predictions.shape) == 2:  # if predictions are one-hot
            predictions = predictions.argmax(dim=1)
        if len(labels.shape) == 2:  # if predictions are one-hot
            labels = labels.argmax(dim=1)
        for name, metric_fct in self.metric_ftns.items():
            metric_fct(predictions.to("cpu"), labels.to("cpu"))
            metrics_dict[name] = f"{metric_fct.compute():.4f}"
        return metrics_dict

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        with tqdm(self.data_loader, unit="batch") as tepoch:
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Train Epoch {epoch}")
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                metrics_dict = self._compute_metrics(output, target)
                epoch_loss += loss.item()
                if batch_idx % self.log_step == 0:
                    if not self.metric_ftns:
                        tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1))
                    else:
                        tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1), metrics=metrics_dict)
                    sleep(0.001)

                #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:  # iteration-based training
                    break

        metrics_dict = self._aggregate_metrics_per_epoch("train", epoch)

        if not self.metric_ftns:
            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1))
        else:
            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1), metrics=metrics_dict)
        sleep(0.001)
        metrics_dict["loss"] = epoch_loss / (batch_idx + 1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            metrics_dict.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return metrics_dict

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.valid_data_loader, unit="batch") as tepoch:
                epoch_loss = 0.0
                for batch_idx, (data, target) in enumerate(tepoch):
                    tepoch.set_description(f"Validation Epoch {epoch}")
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    loss = self.criterion(output, target)

                    metrics_dict = self._compute_metrics(output, target)
                    epoch_loss += loss.item()
                    if batch_idx % self.log_step == 0:
                        if not self.metric_ftns:
                            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1))
                        else:
                            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1), metrics=metrics_dict)
                        sleep(0.001)
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        metrics_dict = self._aggregate_metrics_per_epoch("valid", epoch)
        metrics_dict["loss"] = epoch_loss / (batch_idx + 1)
        return metrics_dict
