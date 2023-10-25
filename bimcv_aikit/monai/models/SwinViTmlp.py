import torch.nn as nn
import torch.nn.functional as F
import monai
import torch


class SwinViTmlp(nn.Module):
    """
    SwinViTmlp: A custom neural network module for image classification using a SwinUnetr backbone 
    and additional fully connected layers for classification. 


    Args:
        n_classes (int): Number of output classes. Defaults to 2.
        img_size (tuple): Image size (Depth, Height, Width) that should be divisible by 32. Defaults to (96, 96, 96).
        in_channels (int): Number of input image channels. Defaults to 1.
        pretrained_weights (str): Path to the pretrained weights of the SwinUnetr model (Optional). Defaults to None.

    Forward Return:
        out (Tensor): The classification logits.

    Example:
        model = SwinViTmlp(n_classes=3, img_size=(128, 128, 128), in_channels=1, pretrained_weights=None)
    """

    def __init__(self, n_classes: int = 2, img_size: tuple = (96,96,96), in_channels: int = 1, pretrained_weights: str = None):
        """
        Initialize the SwinViTmlp model with the given parameters.
        """
        super(SwinViTmlp, self).__init__()

        # Define Backbone (SwinUNETR)
        # The SwinUNETR model is used as a backbone to extract features from the input image.
        backbone = monai.networks.nets.SwinUNETR(
            img_size=img_size, 
            in_channels=in_channels, 
            out_channels=14, 
            feature_size=48, 
            use_v2=True
        )

        # Load pretrained weights if provided
        if pretrained_weights:
            backbone.load_from(weights=torch.load(pretrained_weights))

        # Use the swinViT part of the SwinUNETR as the backbone for feature extraction
        self.backbone = backbone.swinViT

        # Define 'necks' for each head
        # The extracted features are passed through a fully connected layer to reduce dimensions.
        self.fc = nn.Linear(768, 32)

        # Define heads
        # An additional fully connected layer to produce the classification logits.
        self.out = nn.Linear(32, n_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width)

        Returns:
            out (Tensor): Classification logits of shape (batch_size, n_classes)
        """
        # Extract features using the backbone
        x = self.backbone(x)[4]

        # Apply Global Average Pooling on the extracted features
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)

        # Pass through the fully connected 'neck'
        x = F.silu(self.fc(x))

        # Compute the classification logits using the head
        out = self.out(x)

        return out


def test():
    model = SwinViTmlp(n_classes=2)
    input = torch.randn(3, 1, 96, 96, 96)
    out = model(input)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()
