import torch
from torch import nn


class seg_and_class_loss_multitask(nn.Module):
    """
    seg_and_class_loss_multitask: A custom loss module designed for multitasking involving a segmentation task and a classification task.

    The loss module calculates and combines the losses from both tasks based on provided weights.

    Args:
        segmentation_loss (nn.Module): Loss function for the segmentation task.
        classification_loss (nn.Module): Loss function for the classification task.
        weight_seg (float): Weight given to the segmentation loss. Defaults to 0.5.
        weight_class (float): Weight given to the classification loss. Defaults to 0.5.

    Forward Args:
        net_output (torch.Tensor): Predictions from the network.
        target (torch.Tensor): Ground truth labels.

    Forward Return:
        result (torch.Tensor): Weighted sum of segmentation and classification losses.

    Example:
        loss_function = seg_and_class_loss_multitask(segmentation_loss=nn.CrossEntropyLoss(),
                                                    classification_loss=nn.BCEWithLogitsLoss())
    """

    def __init__(self, segmentation_loss, classification_loss, weight_seg: float = 0.5, weight_class: float = 0.5):
        super(seg_and_class_loss_multitask, self).__init__()
        self.segloss = segmentation_loss
        self.classloss = classification_loss
        self.weight_seg = weight_seg
        self.weight_class = weight_class

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_segmentation = self.segloss(net_output[0], target[0])
        loss_classification = self.classloss(net_output[1], target[1])
        result = self.weight_seg * loss_segmentation + self.weight_class * loss_classification
        return result
