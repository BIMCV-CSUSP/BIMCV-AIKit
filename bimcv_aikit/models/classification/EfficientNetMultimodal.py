import torch
import torch.nn as nn
from monai.networks.nets import ViT, EfficientNetBN

class EfficientNetMultimodal(nn.Module):
    """
    EfficientNetMultimodal: An experimental architecture combining EfficientNet and a fully connected layer
    for multimodal classification, designed for grayscale 3D volumes (e.g., structural brain MRIs) and a 1D array 
    of numerical values (e.g., clinical variables).

    The model uses an EfficientNet backbone for feature extraction and a fully connected layer for classification.

    Args:
        model_name (str): Model version/name to use from the EfficientNet variants. Defaults to "efficientnet-b0".
        n_classes (int): Number of output classes. Defaults to 2.
        in_channels_eff (int): Number of input channels for efficientnet. Defaults to 1.
        in_num_features (int): Number of input features for the fully connected layer. Defaults to 25.
        pretrained_weights_path (str): Path to the pretrained weights of EfficientNet (Optional). Defaults to None.

    Forward Return:
        x (Tensor): The classification logits.

    Example:
        model = EfficientNetMultimodal(model_name='efficientnet-b0', n_classes=2, pretrained_weights_path=None)
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        n_classes: int = 2,
        in_channels_eff: int = 1,
        in_num_features: int = 25,
        pretrained_weights_path: str = None,
    ):
        """
        Initialize the EfficientNetMultimodal model with the given parameters.
        """
        super().__init__()

        # Instantiate the EfficientNet model
        self.vision_backbone = EfficientNetBN(
            model_name=model_name,
            pretrained=True,
            progress=False,
            spatial_dims=3,
            in_channels=in_channels_eff,
            num_classes=n_classes,
        )

        # Load pretrained weights into EfficientNet if provided
        if pretrained_weights_path:
            self.vision_backbone.load_state_dict(torch.load(pretrained_weights_path)["state_dict"])

        # Define the fully connected layer
        self.fc = nn.Linear(in_num_features + n_classes, n_classes)

    def forward(self, x_img, x_num):
        """
        Forward pass through the model.

        Args:
            x_img (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width).
            x_num (Tensor): Input tensor of shape (batch_size, in_num_features).

        Returns:
            x (Tensor): Classification logits of shape (batch_size, n_classes).
        """
        # Extract features using EfficientNet
        x = self.vision_backbone(x_img)

        # Concatenate the features with the input tensor
        x = torch.cat((x, x_num), dim=1)
        

        # Pass the concatenated tensor through the fully connected layer
        x = self.fc(x)
        return x


def test():
    batch_size = 2
    in_channels = 1
    depth = 96
    height = 96
    width = 96
    in_num_features = 25
    n_classes = 2
    model = EfficientNetMultimodal(model_name="efficientnet-b0", n_classes=n_classes, in_channels_eff=in_channels, in_num_features=in_num_features)
    x_img = torch.randn(batch_size, in_channels, depth, height, width)
    x_num = torch.randn(batch_size, in_num_features)
    output = model(x_img, x_num)
    assert output.shape == (batch_size, n_classes)

if __name__ == "__main__":
    test()
