from time import sleep

import numpy as np
import torch
from monai import visualize
from torch.nn.functional import softmax
from tqdm import tqdm

from ..utils import inf_loop
from .BaseTrainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        train_data_loader,
        fold="",
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold)
        self.config = config
        self.device = device
        self.data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

    def evaluate(self, data_loader):
        """
        Evaluates the PyTorch model using the given data loader.

        :param data_loader: torch.utils.data.DataLoader, the PyTorch DataLoader object to use for evaluation.
        :return: A tuple containing the predicted values and the computed metrics.
        """

        path = str(self.checkpoint_dir / "model_best.pth")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.logger.info(f"Checkpoint {path} loaded.")
        self.model = self.model.to(self.device)
        self.model.eval()

        outputs = []
        labels = []

        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as tepoch:
                for batch_data in tepoch:
                    tepoch.set_description("Progress")
                    data, target = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                    labels.append(target)
                    outputs.append(self.model(data))

        predictions, labels = torch.cat(outputs, 0), torch.cat(labels, 0)
        metrics_dict = {}
        if any(predictions.sum(dim=1) != 1.0):  # if predictions are not probabilities
            predictions = softmax(predictions, dim=1)
        predict_proba = predictions.cpu().numpy()
        if len(predictions.shape) == 2:  # if predictions are one-hot
            predictions = predictions.argmax(dim=1)
        if len(labels.shape) == 2:  # if predictions are one-hot
            labels = labels.argmax(dim=1)
        for name, metric_fct in self.metric_ftns.items():
            result = metric_fct(predictions.to("cpu"), labels.to("cpu"))
            try:
                metrics_dict[name] = result.item()
            except:
                metrics_dict[name] = result.numpy()
            metric_fct.reset()
        return predict_proba, metrics_dict

    def _aggregate_metrics_per_epoch(self, stage, epoch):
        """
        Aggregates metrics for the given stage and epoch, and logs to tensorboard.

        :param stage: The stage of the training process. Must be a string.
        :param epoch: The epoch number. Must be an integer.
        :return: A dictionary containing the aggregated metrics.
        """

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
        """
        Computes metrics for the given predictions and labels.

        :param predictions: torch.Tensor, the predicted values.
        :param labels: torch.Tensor, the true values.
        :return: A dictionary containing the computed metrics.
        """

        if not self.metric_ftns:
            return {}
        metrics_dict = {}
        if any(predictions.sum(dim=1) != 1.0):  # if predictions are not probabilities
            predictions = softmax(predictions, dim=1)
        if len(predictions.shape) == 2:  # if predictions are one-hot
            predictions = predictions.argmax(dim=1, keepdim=True)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        elif len(labels.shape) == 2:  # if predictions are one-hot
            labels = labels.argmax(dim=1, keepdim=True)
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

        self.model = self.model.to(self.device)
        self.model.train()

        with tqdm(self.data_loader, unit="batch") as tepoch:
            epoch_loss = 0.0
            for batch_idx, batch_data in enumerate(tepoch):
                tepoch.set_description(f"Train Epoch {epoch}")
                data, target = batch_data["image"].to(self.device), batch_data["label"].to(self.device)

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

                if batch_idx == self.len_epoch:  # iteration-based training
                    break

        metrics_dict = self._aggregate_metrics_per_epoch("train", epoch)
        # if epoch % 1 == 0:
        #     self.writer.add_image("input_image", data.cpu()[0, :, :, :, 16])
        #     self.writer.add_video("input_video", data.cpu().transpose(4, 1), global_step=epoch)

        if not self.metric_ftns:
            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1))
        else:
            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1), metrics=metrics_dict)
        sleep(0.001)
        metrics_dict["loss"] = epoch_loss / (batch_idx + 1)
        self.writer.add_scalar("loss/train", metrics_dict["loss"], epoch)

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
                for batch_idx, batch_data in enumerate(tepoch):
                    tepoch.set_description(f"Validation Epoch {epoch}")
                    data, target = batch_data["image"].to(self.device), batch_data["label"].to(self.device)

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
                        self.writer.add_image("input", data.cpu())

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        metrics_dict = self._aggregate_metrics_per_epoch("validation", epoch)
        metrics_dict["loss"] = epoch_loss / (batch_idx + 1)
        self.writer.add_scalar("loss/validation", metrics_dict["loss"], epoch)
        return metrics_dict
