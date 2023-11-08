import monai
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinUnetr_multitask(nn.Module):
    """
    SwinUnetr_multitask: A multi-task learning model using the SwinUnetr architecture, which is designed
    to provide both classification and segmentation outputs given an input image.

    Args:
        n_classes (int): Number of output classes for the classification task. Defaults to 2.
        img_size (tuple): Image size (Depth, Height, Width). Must be divisible by 32. Defaults to (96, 96, 96).
        in_channels (int): Number of input image channels. Defaults to 1.
        seg_channels (int): Number of output segmentation channels. Defaults to 14.
        pretrained_weights (str): Path to the pretrained weights of the SwinUnetr model (Optional). Defaults to None.
        feature_size (int): Feature size for the SwinUnetr model. Defaults to 48.
        drop_rate (float): Dropout rate applied in SwinUnetr and the fully connected layers. Defaults to 0.0.

    Forward Return:
        class_out (Tensor): The classification logits.
        logits_seg (Tensor): The segmentation map.

    Example:
        model = SwinUnetr_multitask(n_classes=3, img_size=(128, 128, 128), in_channels=1, pretrained_weights=None)
    """

    def __init__(
        self,
        n_classes: int = 2,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 1,
        seg_channels: int = 14,
        pretrained_weights: str = None,
        feature_size: int = 48,
        drop_rate: float = 0.0,
    ):
        """
        Initialize the SwinUnetr_multitask model with the given parameters.
        """
        super(SwinUnetr_multitask, self).__init__()

        self.drop_rate = drop_rate

        # Define the SwinUNETR model for feature extraction and segmentation.
        self.model = monai.networks.nets.SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=seg_channels,
            feature_size=feature_size,
            use_v2=True,
            drop_rate=drop_rate,
        )

        # Load pretrained weights if provided.
        if pretrained_weights:
            self.model.load_from(weights=torch.load(pretrained_weights))

        # Extract the swinViT part of the SwinUNETR for feature extraction in classification.
        self.backbone = self.model.swinViT

        # Define fully connected layers for classification.
        self.fc1 = nn.Linear(16 * feature_size, 8 * feature_size)
        self.fc2 = nn.Linear(8 * feature_size, 4 * feature_size)
        self.fc3 = nn.Linear(4 * feature_size, 2 * feature_size)
        self.fc4 = nn.Linear(2 * feature_size, feature_size)

        # Define head for classification.
        self.out_class = nn.Linear(feature_size, n_classes)

    def forward(self, x_in):
        """
        Forward pass through the model.

        Args:
            x_in (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            class_out (Tensor): Classification logits of shape (batch_size, n_classes).
            logits_seg (Tensor): Segmentation map.
        """
        # Extract features using the SwinViT backbone.
        hidden_states_out = self.backbone(x_in)

        # Classification task
        # Perform Global Average Pooling and pass through the classification head.
        x1 = F.adaptive_avg_pool3d(hidden_states_out[4], (1, 1, 1)).view(hidden_states_out[4].size(0), -1)
        x1 = F.dropout(F.relu(self.fc1(x1)), self.drop_rate)
        x1 = F.dropout(F.relu(self.fc2(x1)), self.drop_rate)
        x1 = F.dropout(F.relu(self.fc3(x1)), self.drop_rate)
        x1 = F.dropout(F.relu(self.fc4(x1)), self.drop_rate)
        class_out = self.out_class(x1)

        # Segmentation task
        # Pass through the SwinUNETR layers to produce the segmentation map.
        enc0 = self.model.encoder1(x_in)
        enc1 = self.model.encoder2(hidden_states_out[0])
        enc2 = self.model.encoder3(hidden_states_out[1])
        enc3 = self.model.encoder4(hidden_states_out[2])
        dec4 = self.model.encoder10(hidden_states_out[4])
        dec3 = self.model.decoder5(dec4, hidden_states_out[3])
        dec2 = self.model.decoder4(dec3, enc3)
        dec1 = self.model.decoder3(dec2, enc2)
        dec0 = self.model.decoder2(dec1, enc1)
        out = self.model.decoder1(dec0, enc0)
        logits_seg = self.model.out(out)

        return logits_seg, class_out


def test():
    model = SwinUnetr_multitask(n_classes=2)
    input = torch.randn(3, 1, 96, 96, 96)
    out = model(input)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()
