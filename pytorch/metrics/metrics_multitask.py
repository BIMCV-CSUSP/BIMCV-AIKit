
from torchmetrics import Metric
import torch

class metrics_classification_multitask(Metric):
    """
    metrics_classification_multitask: A custom metric for multitask classification problems. 
                                    It focuses on extracting the second task's predictions 
                                    and targets, then computing accuracy using the provided 
                                    original metric (e.g., Accuracy).

    Args:
        original_metric (Metric): A torchmetrics Metric instance that will be used to 
                                compute the actual metric value based on extracted data.

    Example:
        Accuracy_multitask = metrics_classification_multitask(original_metric=Accuracy(task="multiclass", 
                                                average="weighted", num_classes=2))
    """

    def __init__(self,original_metric):
        super().__init__()
        
        self.metric=original_metric

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        
        preds_,targets=preds[1].argmax(dim=-1).to('cpu'), target[1].argmax(dim=-1).to('cpu')
       
        return self.metric(preds_,targets)
        

    def compute(self) -> torch.Tensor:
        # compute final result
        return self.metric.compute()
    
    def reset(self):
        self.metric.reset()


def test():
    # Create synthetic data
    preds = [1,torch.tensor([
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7],
        [0.9, 0.1]
    ])]
    
    targets = [1, torch.tensor([
        [1, 0],
        [1, 0],
        [0, 1],
        [1, 0]
    ])]
    
    # Update metric
    Accuracy_multitask(preds, targets)
    
    # Compute accuracy
    acc = Accuracy_multitask.compute()
    
    # Expected accuracy
    # For this synthetic data, we expect 2 predictions to be correct out of 4.
    expected_acc = 3 / 4
    
    assert acc.item() == expected_acc, f"Expected {expected_acc}, but got {acc.item()}"
    print(f"Expected {expected_acc}, got {acc.item()}")
    
    print("Test passed!")
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    from torchmetrics import Accuracy
    n_classes=2
    Accuracy_multitask=metrics_classification_multitask(original_metric=Accuracy(task="multiclass", average="weighted", num_classes=n_classes))
    test()