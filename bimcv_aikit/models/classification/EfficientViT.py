import torch
import torch.nn as nn

from monai.networks.nets import EfficientNetBN, ViT


class EfficientViT(nn.Module):
    """
    EfficientViT: An experimental architecture combining EfficientNet and Vision Transformer (ViT)
    for 3D image classification, designed for grayscale 3D volumes (e.g., structural brain MRIs).

    The model uses an EfficientNet backbone for feature extraction and a ViT for classification.

    Args:
        model_name (str): Model version/name to use from the EfficientNet variants. Defaults to "efficientnet-b0".
        n_classes (int): Number of output classes. Defaults to 2.
        in_channels_eff (int): Number of input channels for efficientnet. Defaults to 1.
        in_channels_vit (int): Number of input channels for ViT. Depends on efficientnet final layer. Defaults to 24.
        pretrained_weights_path (str): Path to the pretrained weights of EfficientNet (Optional). Defaults to None.

    Forward Return:
        x (Tensor): The classification logits.

    Example:
        model = EfficientViT(model_name='efficientnet-b0', n_classes=2, pretrained_weights_path=None)
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        n_classes: int = 2,
        in_channels_eff: int = 1,
        in_channels_vit: int = 24,
        pretrained_weights_path: str = None,
    ):
        """
        Initialize the EfficientViT model with the given parameters.
        """
        super(EfficientViT, self).__init__()

        # Instantiate the EfficientNet model
        EfficientNet = EfficientNetBN(
            model_name=model_name,
            pretrained=True,
            progress=False,
            spatial_dims=3,
            in_channels=in_channels_eff,
            num_classes=n_classes,
        )

        # Load pretrained weights into EfficientNet if provided
        if pretrained_weights_path:
            EfficientNet.load_state_dict(torch.load(pretrained_weights_path))

        # Use certain layers from EfficientNet for feature extraction
        layers = list(EfficientNet.children())[:3] + list(EfficientNet._blocks.children())[:2]
        self.features = nn.Sequential(*layers)

        # Define the Vision Transformer model for classification
        self.vit = ViT(
            in_channels=in_channels_vit,
            img_size=(32, 32, 8),  # Input image size // 4
            patch_size=4,
            num_classes=n_classes,
            classification=True,
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
        # print(x.shape)
        # Pass the features through the ViT model and retrieve the classification logits
        x = self.vit(x)[0]

        return x


def test():
    model = EfficientViT(
        model_name="efficientnet-b7",
        n_classes=3,
        in_channels_eff=5,
        in_channels_vit=48,
        pretrained_weights_path=None,
    )
    # print(model)
    input = torch.randn(3, 5, 128, 128, 32)
    out = model(input)
    print(out.shape)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()
