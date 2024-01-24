from collections.abc import Callable
from typing import Union

import torch
from monai.metrics.metric import Cumulative, IterationMetric
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction


class BaseMetric(Cumulative, IterationMetric):
    def __init__(
        self,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.reduction = reduction

    def __call__(self, y_pred: Union[torch.Tensor, list], y: Union[torch.Tensor, list, None] = None, **kwargs) -> Union[torch.Tensor, list]:
        """
        Execute basic computation for model prediction and ground truth.
        It can support  both `list of channel-first torch.Tensor` and `batch-first Tensor`.
        Users call this API to execute computation on every batch of data, then accumulate the results,
        or accumulate the original `y_pred` and `y`, then execute on the accumulated data.

        Args:
            y_pred: the model prediction data to compute, must be a list of `channel-first` torch.Tensor
                or a `batch-first` torch.Tensor.
            y: the ground truth to compute, must be a list of `channel-first` torch.Tensor
                or a `batch-first` torch.Tensor.
            kwargs: additional parameters for specific metric computation logic (e.g. ``spacing`` for SurfaceDistanceMetric, etc.).

        Returns:
            The computed metric values at the iteration level. The output shape should be
            a `batch-first` tensor (BC[HWD]) or a list of `batch-first` tensors.
        """
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        if isinstance(ret, (tuple, list)):  # First two cases account for image metrics (more than one dimension)
            self.extend(*ret)
        elif isinstance(ret, torch.Tensor) and ret.ndim > 1:
            self.extend(ret)
        elif isinstance(ret, torch.Tensor) and ret.ndim <= 1:  # This case accounts for single float value metric
            self.append(ret)
        return ret

    def _compute_tensor(self, predictions: torch.Tensor, labels: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            predictions: input data to compute, typical model output.
            labels: ground truth to compare the predictions

        Raises:
            ValueError: when predictions and labels are of different shape.
        """
        if predictions.shape != labels.shape:
            raise ValueError(f"Predictions and labels must have same shapes, got {predictions.shape} and {labels.shape}.")
        return self.metric(predictions, labels, *args, **kwargs)

    def compute(self, reduction: Union[MetricReduction, str, None] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Execute reduction logic for the output of `metric`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be PyTorch torch.Tensor.")
        if data.ndim <= 1:  # For single float value metrics
            f = do_1d_metric_reduction(data, reduction or self.reduction)
        else:  # For metrics with more dimensions
            f = do_metric_reduction(data, reduction or self.reduction)
        return f


def do_1d_metric_reduction(f: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    This function is to do the metric reduction for calculated `not-nan` metrics of each sample's each class.
    The function also returns `not_nans`, which counts the number of not nans for the metric.

    Args:
        f: a tensor that contains the calculated metric scores per batch and
            per class. The first two dims should be batch and class.
        reduction: define the mode to reduce metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``}, default to ``"mean"``.
            if "none", return the input f tensor and not_nans.

    Raises:
        ValueError: When ``reduction`` is not one of
            ["mean", "sum", "none"].
    """

    if reduction == "none":
        return f
    elif reduction == "mean":
        f = torch.mean(f)
    elif reduction == "sum":
        f = torch.sum(f)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}, available options are " '["mean", "sum", "none"].')
    return f
