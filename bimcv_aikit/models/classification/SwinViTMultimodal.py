import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinUNETR

from .SwinViTMLP import SwinViTMLP_v3


class SwinViTMultimodal(nn.Module):
    """
    SwinViTMultimodal: A novel architecture that merges SwinViT and a dense layer
    for multimodal classification. It's tailored for grayscale 3D volumes (like structural brain MRIs) and a 1D array
    of numerical data (such as clinical variables).

    This model employs a SwinViT backbone for feature extraction and a dense layer for classification.

    Args:
        n_classes (int): The number of output classes. Defaults to 2.
        img_size (tuple): The size of the input image. Defaults to (96, 96, 96).
        in_channels (int): The number of input channels for the image. Defaults to 1.
        in_num_features (int): The number of input features for the dense layer. Defaults to 25.
        pretrained_weights (str): The path to the pretrained weights of SwinViT (Optional). Defaults to None.

    Forward Return:
        x (Tensor): The classification logits.

    Example:
        model = SwinViTMultimodal(n_classes=3, img_size=(128, 128, 128), in_channels=1, pretrained_weights=None)
    """

    def __init__(
        self,
        n_classes: int = 2,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 1,
        in_num_features: int = 25,
        pretrained_weights: str = None,
    ):
        """
        Initialize the SwinViTMultimodal model with the provided parameters.
        """
        super().__init__()

        # Define Backbone (SwinUNETR)
        # The SwinUNETR model is used as a backbone to extract features from the input image.
        backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=14,
            feature_size=48,
            use_v2=True,
        )

        # Load pretrained weights if provided
        if pretrained_weights:
            backbone.load_from(weights=torch.load(pretrained_weights))

        # Use the swinViT part of the SwinUNETR as the backbone for feature extraction
        self.vision_backbone = backbone.swinViT

        # Define 'necks' for each head
        # The extracted features are passed through a fully connected layer to reduce dimensions.
        self.fc = nn.Linear(768, 32)

        # Define heads
        # An additional fully connected layer to produce the classification logits.
        self.out_backbone = nn.Linear(32, n_classes)

        # Define the fully connected layer
        self.fc_out = nn.Linear(in_num_features + n_classes, n_classes)

    def forward(self, x_img, x_num):
        """
        Forward pass through the model.

        Args:
            x_img (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width).
            x_num (Tensor): Input tensor of shape (batch_size, in_num_features).

        Returns:
            x (Tensor): Classification logits of shape (batch_size, n_classes).
        """
        # Extract features using the backbone
        x = self.vision_backbone(x_img)[4]

        # Apply Global Average Pooling on the extracted features
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)

        # Pass through the fully connected 'neck'
        x = F.silu(self.fc(x))

        # Compute the classification logits using the head
        x = self.out_backbone(x)

        # Concatenate the features with the input tensor
        x = torch.cat((x, x_num), dim=1)

        # Pass the concatenated tensor through the fully connected layer
        x = self.fc_out(x)
        return x


class SwinViTMultimodal_v3(SwinViTMLP_v3):
    """
    SwinViTMultimodal_v3: A novel architecture that merges SwinViT and a dense layer
    for multimodal classification. It's tailored for grayscale 3D volumes (like structural brain MRIs) and a 1D array
    of numerical data (such as clinical variables).

    This model employs a SwinViT backbone for feature extraction and a dense layer for classification.

    Args:
        n_classes (int): The number of output classes. Defaults to 2.
        img_size (tuple): The size of the input image. Defaults to (96, 96, 96).
        in_channels (int): The number of input channels for the image. Defaults to 1.
        in_num_features (int): The number of input features for the dense layer. Defaults to 25.
        pretrained_weights (str): The path to the pretrained weights of SwinViT (Optional). Defaults to None.

    Forward Return:
        x (Tensor): The classification logits.

    Example:
        model = SwinViTMultimodal(n_classes=3, img_size=(128, 128, 128), in_channels=1, pretrained_weights=None)
    """

    def __init__(
        self,
        n_classes: int = 2,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 1,
        in_num_features: int = 25,
        pretrained_weights: str = None,
    ):
        """
        Initialize the SwinViTMultimodal model with the provided parameters.
        """
        super().__init__(
            n_classes=n_classes,
            img_size=img_size,
            in_channels=in_channels,
            pretrained_weights=pretrained_weights,
        )
        feature_size = 48
        # Define fully connected layers for classification.
        self.fc1 = nn.Linear((16 * feature_size) + in_num_features, feature_size)
        self.fc2 = nn.Linear((8 * feature_size) + in_num_features, feature_size)
        self.fc3 = nn.Linear((4 * feature_size) + in_num_features, feature_size)
        self.fc4 = nn.Linear((2 * feature_size) + in_num_features, feature_size)

        # Define head for classification.
        self.fc = nn.Linear((5 * feature_size), feature_size)
        self.out_class = nn.Linear(feature_size, n_classes)
        # Define the fully connected layer
        # self.fc_out = nn.Linear(in_num_features + n_classes, n_classes)

    def forward(self, x_img, x_num):
        """
        Forward pass through the model.

        Args:
            x_img (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width).
            x_num (Tensor): Input tensor of shape (batch_size, in_num_features).

        Returns:
            x (Tensor): Classification logits of shape (batch_size, n_classes).
        """
        # Extract features using the backbone
        hidden_states_out = self.base_model.swinViT(x_img, self.base_model.normalize)
        enc1 = self.base_model.encoder2(hidden_states_out[0])
        enc2 = self.base_model.encoder3(hidden_states_out[1])
        enc3 = self.base_model.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        enc5 = self.base_model.encoder10(hidden_states_out[4])

        out1 = F.adaptive_avg_pool3d(enc5, (1, 1, 1)).view(enc5.size(0), -1)
        out1 = F.silu(self.fc1(torch.cat((out1, x_num), dim=1)))

        out2 = F.adaptive_avg_pool3d(enc4, (1, 1, 1)).view(enc4.size(0), -1)
        out2 = F.silu(self.fc2(torch.cat((out2, x_num), dim=1)))

        out3 = F.adaptive_avg_pool3d(enc3, (1, 1, 1)).view(enc3.size(0), -1)
        out3 = F.silu(self.fc3(torch.cat((out3, x_num), dim=1)))

        out4 = F.adaptive_avg_pool3d(enc2, (1, 1, 1)).view(enc2.size(0), -1)
        out4 = F.silu(self.fc4(torch.cat((out4, x_num), dim=1)))

        out5 = F.adaptive_avg_pool3d(enc1, (1, 1, 1)).view(enc1.size(0), -1)

        out = torch.cat((out1, out2, out3, out4, out5), dim=1)

        out = F.silu(self.fc(out))

        x = self.out_class(out)

        return x


def test():
    batch_size = 2
    in_channels = 1
    depth = 96
    height = 96
    width = 96
    in_num_features = 25
    n_classes = 2
    model = SwinViTMultimodal(
        n_classes=n_classes,
        img_size=(depth, height, width),
        in_channels=in_channels,
        in_num_features=in_num_features,
    )
    x_img = torch.randn(batch_size, in_channels, depth, height, width)
    x_num = torch.randn(batch_size, in_num_features)
    output = model(x_img, x_num)
    assert output.shape == (batch_size, n_classes)


if __name__ == "__main__":
    test()
