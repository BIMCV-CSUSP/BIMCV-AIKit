import torch
import torch.nn as nn
from monai.networks.nets import ViT, EfficientNetBN


class EfficientViT(nn.Module):
    """
    EfficientViT: An experimental architecture combining EfficientNet and Vision Transformer (ViT) 
    for 3D image classification, designed for grayscale 3D volumes (e.g., structural brain MRIs).

    The model uses an EfficientNet backbone for feature extraction and a ViT for classification.

    Args:
        model_name (str): Model version/name to use from the EfficientNet variants. Defaults to "efficientnet-b0".
        n_classes (int): Number of output classes. Defaults to 2.
        pretrained_weights_path (str): Path to the pretrained weights of EfficientNet (Optional). Defaults to None.

    Forward Return:
        x (Tensor): The classification logits.

    Example:
        model = EfficientViT(model_name='efficientnet-b0', n_classes=2, pretrained_weights_path=None)
    """

    def __init__(self, model_name: str = "efficientnet-b0", n_classes: int = 2, pretrained_weights_path: str = None):
        """
        Initialize the EfficientViT model with the given parameters.
        """
        super(EfficientViT, self).__init__()

        # Instantiate the EfficientNet model
        EfficientNet = EfficientNetBN(
            model_name=model_name, 
            pretrained=False, 
            progress=False, 
            spatial_dims=3, 
            in_channels=1, 
            num_classes=n_classes
        )

        # Load pretrained weights into EfficientNet if provided
        if pretrained_weights_path:
            EfficientNet.load_state_dict(torch.load(pretrained_weights_path))

        # Use certain layers from EfficientNet for feature extraction
        layers = list(EfficientNet.children())[:3] + list(EfficientNet._blocks.children())[:2]
        self.features = nn.Sequential(*layers)

        # Define the Vision Transformer model for classification
        self.vit = ViT(
            in_channels=24, 
            img_size=(23, 23, 23), 
            patch_size=4, 
            num_classes=n_classes, 
            classification=True
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            x (Tensor): Classification logits of shape (batch_size, n_classes).
        """
        # Extract features using EfficientNet
        x = self.features(x)

        # Pass the features through the ViT model and retrieve the classification logits
        x = self.vit(x)[0]

        return x


def test():
    model = EfficientViT(n_classes=3, pretrained_weights_path=None)
    # print(model)
    input = torch.randn(3, 1, 91, 91, 91)
    out = model(input)
    print(out.shape)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()
