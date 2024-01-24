from time import sleep

import numpy as np
import torch
from monai import visualize
from torch.nn.functional import softmax
from tqdm import tqdm

from ..utils import inf_loop
from .ClassificationTrainer import ClassificationTrainer


class MultimodalClassificationTrainer(ClassificationTrainer):
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
        super().__init__(
            model, criterion, metric_ftns, optimizer, config, device, train_data_loader, fold, valid_data_loader, lr_scheduler, len_epoch
        )

    def _evaluate(self, data_loader):
        """
        Evaluates the PyTorch model using the given data loader.

        :param data_loader: torch.utils.data.DataLoader, the PyTorch DataLoader object to use for evaluation.
        :return: A tuple containing the predicted values and the computed metrics.
        """

        outputs = []
        labels = []

        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as tepoch:
                for batch_data in tepoch:
                    tepoch.set_description("Progress")
                    img_data, num_data, target = (
                        batch_data["image"].to(self.device),
                        batch_data["numeric"].to(self.device),
                        batch_data["label"].to(self.device),
                    )
                    labels.append(target)
                    outputs.append(self.model(img_data, num_data))

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
                img_data, num_data, target = (
                    batch_data["image"].to(self.device),
                    batch_data["numeric"].to(self.device),
                    batch_data["label"].to(self.device),
                )

                self.optimizer.zero_grad()
                output = self.model(img_data, num_data)
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
        # if epoch % 5 == 0:
        #     self.writer.add_image("input_image", img_data.cpu()[0, :, :, :, 16], global_step=epoch)
            # self.writer.add_video('input_video', img_data.cpu().transpose(4,1), global_step=epoch)

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
                    img_data, num_data, target = (
                        batch_data["image"].to(self.device),
                        batch_data["numeric"].to(self.device),
                        batch_data["label"].to(self.device),
                    )

                    output = self.model(img_data, num_data)
                    loss = self.criterion(output, target)

                    metrics_dict = self._compute_metrics(output, target)
                    epoch_loss += loss.item()
                    if batch_idx % self.log_step == 0:
                        if not self.metric_ftns:
                            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1))
                        else:
                            tepoch.set_postfix(loss=epoch_loss / (batch_idx + 1), metrics=metrics_dict)
                        sleep(0.001)
                        # self.writer.add_image('input', img_data.cpu())

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        metrics_dict = self._aggregate_metrics_per_epoch("validation", epoch)
        metrics_dict["loss"] = epoch_loss / (batch_idx + 1)
        self.writer.add_scalar("loss/validation", metrics_dict["loss"], epoch)
        return metrics_dict
