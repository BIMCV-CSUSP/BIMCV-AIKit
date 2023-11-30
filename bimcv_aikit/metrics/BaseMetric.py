from collections.abc import Callable
from typing import Union

from monai.metrics.metric import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction
from torch import Tensor


class BaseMetric(CumulativeIterationMetric):
    def __init__(
        self, metric: Callable[[Tensor, Tensor], Tensor], post_transforms: dict = None, reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ) -> None:
        super().__init__()
        self.metric = metric
        self.post_transforms = self._init_transforms(post_transforms)
        self.reduction = reduction

    def _compute_tensor(self, predictions: Tensor, labels: Tensor, *args, **kwargs) -> Tensor:
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

    def compute(self, reduction: Union[MetricReduction, str, None] = None) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Execute reduction logic for the output of `metric`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, Tensor):
            raise ValueError("The data to aggregate must be PyTorch Tensor.")

        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f
