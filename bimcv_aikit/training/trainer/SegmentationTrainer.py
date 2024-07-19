import logging
from time import sleep

import numpy as np
import torch
from tqdm import tqdm

from .BaseTrainer import BaseTrainer


class SegmentationTrainer(BaseTrainer):
    """
    Trainer segmentation
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
    ):
        super().__init__(
            model,
            train_data_loader,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            lr_scheduler,
            valid_data_loader,
            fold,
        )

    def _evaluate(
        self, data_loader: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, dict]:
        """
        Evaluates the PyTorch model using the given data loader.

        :param data_loader: torch.utils.data.DataLoader, the PyTorch DataLoader object to use for evaluation.
        :return: A tuple containing the predicted values and the computed metrics.
        """

        outputs = []
        labels = []

        with torch.no_grad():
            with tqdm(
                data_loader,
                unit="batch",
                disable=(self.logger.level >= logging.WARNING),
            ) as tepoch:
                for batch_data in tepoch:
                    tepoch.set_description("Evaluation progress")
                    data, target = batch_data["image"].to(self.device), batch_data[
                        "label"
                    ].to(self.device)
                    labels.append(target)
                    if self.inferer:
                        out = self.inferer(data, self.model)
                    else:
                        out = self.model(data)
                    if not isinstance(out, torch.Tensor):  # for torchvision models
                        out = out["out"]
                    outputs.append(out)

        predictions, labels = torch.cat(outputs, 0), torch.cat(labels, 0)

        if self.post_transforms.get("pred"):
            predictions = self.post_transforms["pred"](predictions)
        if self.post_transforms.get("label"):
            labels = self.post_transforms["label"](labels)

        metrics_dict = {}
        for name, metric_fct in self.metric_ftns.items():
            result = metric_fct(predictions, labels)
            try:
                metrics_dict[name] = result.item()
            except Exception:
                metrics_dict[name] = result.numpy()
            metric_fct.reset()
        return np.array([]), metrics_dict

    def _aggregate_metrics_per_epoch(self, stage: str, epoch: int) -> dict:
        """
        Aggregates metrics for the given stage and epoch, and logs to tensorboard.

        :param stage: The stage of the training process. Must be a string.
        :param epoch: The epoch number. Must be an integer.
        :return: A dictionary containing the aggregated metrics.
        """

        if not self.metric_ftns:
            return {}
        metrics_dict = {}
        values = []
        for name, metric_fct in self.metric_ftns.items():
            metrics_dict[name] = metric_fct.compute()
            self.writer.add_scalar(f"{name}/{stage.lower()}", metrics_dict[name], epoch)
            metric_fct.reset()
            values.append(f"{metrics_dict[name]:.4f}")
        return metrics_dict

    def _compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Computes metrics for the given predictions and labels.

        :param predictions: torch.Tensor, the predicted values.
        :param labels: torch.Tensor, the true values.
        :return: A dictionary containing the computed metrics.
        """

        if not self.metric_ftns:
            return {}
        metrics_dict = {}

        if self.post_transforms.get("pred"):
            predictions = self.post_transforms["pred"](predictions)
        if self.post_transforms.get("label"):
            labels = self.post_transforms["label"](labels)

        for name, metric_fct in self.metric_ftns.items():
            metric_fct(predictions, labels)
            metrics_dict[name] = f"{metric_fct.compute():.4f}"
        return metrics_dict

    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model = self.model.to(self.device)
        self.model.train()
        batch_idx = 0
        with tqdm(
            self.data_loader,
            unit="batch",
            disable=(self.logger.level >= logging.WARNING),
        ) as tepoch:
            epoch_loss = 0.0
            for batch_idx, batch_data in enumerate(tepoch):
                tepoch.set_description(f"Train Epoch {epoch}")
                data, target = batch_data["image"].to(self.device), batch_data[
                    "label"
                ].to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                if not isinstance(output, torch.Tensor):  # for torchvision models
                    output = output["out"]
                if epoch == 1 and batch_idx == 0:
                    self.logger.debug(f"Model input shape: {data.shape}")
                    self.logger.debug(f"Model ground truth shape: {target.shape}")
                    self.logger.debug(f"Model output shape: {output.shape}")
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                metrics_dict = self._compute_metrics(output, target)
                epoch_loss += loss.item()
                if batch_idx % self.log_step == 0:
                    if not self.metric_ftns:
                        tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1))
                    else:
                        tepoch.set_postfix(
                            loss=epoch_loss / (batch_idx + 1), metrics=metrics_dict
                        )
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

    def _valid_epoch(self, epoch: int) -> dict:
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        batch_idx = 0
        with torch.no_grad():
            with tqdm(
                self.valid_data_loader,
                unit="batch",
                disable=(self.logger.level >= logging.WARNING),
            ) as tepoch:
                epoch_loss = 0.0
                for batch_idx, batch_data in enumerate(tepoch):
                    tepoch.set_description(f"Validation Epoch {epoch}")
                    data, target = batch_data["image"].to(self.device), batch_data[
                        "label"
                    ].to(self.device)

                    if self.inferer:
                        output = self.inferer(data, self.model)
                    else:
                        output = self.model(data)

                    loss = self.criterion(output, target)

                    metrics_dict = self._compute_metrics(output, target)
                    epoch_loss += loss.item()
                    if batch_idx % self.log_step == 0:
                        if not self.metric_ftns:
                            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1))
                        else:
                            tepoch.set_postfix(
                                loss=epoch_loss / (batch_idx + 1), metrics=metrics_dict
                            )
                        sleep(0.001)
                        # self.writer.add_image("input", data.cpu())

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        metrics_dict = self._aggregate_metrics_per_epoch("validation", epoch)
        metrics_dict["loss"] = epoch_loss / (batch_idx + 1)
        self.writer.add_scalar("loss/validation", metrics_dict["loss"], epoch)
        return metrics_dict
