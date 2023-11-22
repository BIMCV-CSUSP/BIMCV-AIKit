import torch
from torchmetrics import Metric

import monai
from monai.data import decollate_batch

class metrics_segmentation_constructor_monai:
    """
    metrics_segmentation_multitask: A custom metric for multitask segmentation problems.
                                    This class handles extraction of segmented predictions and targets,
                                    post-processes the predictions, and then computes the desired metric
                                    using the provided original metric (e.g., Dice Score).

    Args:
        original_metric (Metric): An instance of a torchmetrics Metric that will be used to compute
                                  the final metric value based on the post-processed data.

    Example Usage:
        DiceScore_multitask = metrics_segmentation_multitask(original_metric=SomeSegmentationMetric(...))
    """

    def __init__(self, original_metric: Metric):
        """
        Constructor for metrics_segmentation_multitask.

        Args:
            original_metric (Metric): The metric instance from torchmetrics to compute the desired metric.
        """

        # Initialize the base metric from torchmetrics
        self.metric = original_metric

        # Compose post-processing transforms for predictions
        self.post_pred = monai.transforms.Compose(
            monai.transforms.AsDiscrete(argmax=True),
        )

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the metric value for the given predictions and targets.

        Args:
            preds (torch.Tensor): Predicted tensor, typically the output from a neural network.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Computed metric value.
        """

        # Move tensors to CPU for further processing
        preds_, targets = preds.to("cpu"), target.to("cpu")

        # Decollate batched tensor data
        labels_list = decollate_batch(targets)
        output_list = decollate_batch(preds_)

        # Apply post-processing transforms on predictions
        output_convert = [self.post_pred(output_tensor) for output_tensor in output_list]

        # Compute metric
        self.metric(output_convert, labels_list)
        return self.compute()

    def compute(self) -> torch.Tensor:
        """
        Compute and return the final metric result after aggregation.

        Returns:
            torch.Tensor: Aggregated metric result.
        """
        return self.metric.aggregate()[0]

    def reset(self):
        """
        Reset the internal states or accumulations of the metric instance.
        This is typically used between epochs or different batches of validation data.
        """
        self.metric.reset()


class metrics_segmentation_multitask_logits:
    """
    metrics_segmentation_multitask_logits: A custom metric for multitask segmentation problems when using logits.
                                    This class handles extraction of segmented predictions and targets,
                                    post-processes the predictions, and then computes the desired metric
                                    using the provided original metric (e.g., Dice Score).

    Args:
        original_metric (Metric): An instance of a torchmetrics Metric that will be used to compute
                                  the final metric value based on the post-processed data.

    Example Usage:
        DiceScore_multitask = metrics_segmentation_multitask(original_metric=SomeSegmentationMetric(...))
    """

    def __init__(self, original_metric: Metric):
        """
        Constructor for metrics_segmentation_multitask.

        Args:
            original_metric (Metric): The metric instance from torchmetrics to compute the desired metric.
        """

        # Initialize the base metric from torchmetrics
        self.metric = original_metric

        # Compose post-processing transforms for predictions
        self.post_pred = monai.transforms.Compose(
            monai.transforms.AsDiscrete(argmax=True),
        )

        self.post_label = monai.transforms.Compose(
            monai.transforms.AsDiscrete(to_onehot=2),
        )

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the metric value for the given predictions and targets.

        Args:
            preds (torch.Tensor): Predicted tensor, typically the output from a neural network.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Computed metric value.
        """

        # Move tensors to CPU for further processing
        preds_, targets = preds.to("cpu"), target.to("cpu")

        # Decollate batched tensor data
        labels_list = decollate_batch(targets)
        output_list = decollate_batch(preds_)

        # Apply post-processing transforms on predictions
        labels_convert = [self.post_pred(labels_tensor) for labels_tensor in labels_list]

        # Compute metric
        self.metric(output_list, labels_convert)
        return self.compute()

    def compute(self) -> torch.Tensor:
        """
        Compute and return the final metric result after aggregation.

        Returns:
            torch.Tensor: Aggregated metric result.
        """
        return self.metric.aggregate()[0]

    def reset(self):
        """
        Reset the internal states or accumulations of the metric instance.
        This is typically used between epochs or different batches of validation data.
        """
        self.metric.reset()

