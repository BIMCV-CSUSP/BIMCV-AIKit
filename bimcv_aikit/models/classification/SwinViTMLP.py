from __future__ import annotations

from collections.abc import Sequence

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, get_conv_layer
from monai.networks.nets.swin_unetr import SwinUNETR


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

    def __init__(self, n_classes: int = 2, img_size: tuple = (96, 96, 96), in_channels: int = 1, pretrained_weights: str = None):
        """
        Initialize the SwinViTmlp model with the given parameters.
        """
        super(SwinViTmlp, self).__init__()

        # Define Backbone (SwinUNETR)
        # The SwinUNETR model is used as a backbone to extract features from the input image.
        backbone = monai.networks.nets.SwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=14, feature_size=48, use_v2=True)

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


class SwinViTMLP_v2(nn.Module):
    """
    SwinViTMLP_v2: A custom neural network module for image classification using a SwinUNETR backbone
    and additional fully connected layers for classification.

    Args:
        n_classes (int): Number of output classes. Defaults to 2.
        img_size (tuple): Image size (Depth, Height, Width) that should be divisible by 32. Defaults to (96, 96, 96).
        in_channels (int): Number of input image channels. Defaults to 1.
        pretrained_weights (str): Path to the pretrained weights of the SwinUNETR model (Optional). Defaults to None.

    Forward Return:
        out (Tensor): The classification logits.

    Example:
        model = SwinViTMLP_v2(n_classes=3, img_size=(128, 128, 128), in_channels=1, pretrained_weights=None)
    """

    def __init__(self, n_classes: int = 2, img_size: tuple = (96, 96, 96), in_channels: int = 1, pretrained_weights: str = None):
        """
        Initialize the SwinViTMLP_v2 model with the given parameters.
        """
        super().__init__()

        feature_size = 48

        # Define Backbone (SwinUNETR)
        # The SwinUNETR model is used as a backbone to extract features from the input image.
        self.base_model = SwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=14, feature_size=feature_size, use_v2=True)

        # Load pretrained weights if provided
        if pretrained_weights:
            self.base_model.load_from(weights=torch.load(pretrained_weights))

        # Define decoder blocks
        # The decoder blocks are used to upsample the extracted features.
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.decoder1 = UnetDownBlock(
            spatial_dims=3, in_channels=feature_size, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name="instance", stride=1
        )
        self.decoder2 = UnetDownBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            stride=1,
        )
        self.decoder3 = UnetDownBlock(
            spatial_dims=3,
            in_channels=2 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            stride=1,
        )
        self.decoder4 = UnetDownBlock(
            spatial_dims=3,
            in_channels=4 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            stride=1,
        )
        self.decoder5 = UnetDownBlock(
            spatial_dims=3,
            in_channels=8 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            stride=1,
        )

        # Define fully connected layers
        # The extracted features are passed through a fully connected layer to reduce dimensions.
        self.fc = nn.Linear(768, 32)

        # Define heads
        # An additional fully connected layer to produce the classification logits.
        self.out = nn.Linear(32, n_classes)

    def forward(self, x):
        """
        Forward pass through the SwinViTMLP_v2 model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width)

        Returns:
            out (Tensor): Classification logits of shape (batch_size, n_classes)
        """
        # Extract features using the SwinUNETR backbone
        hidden_states_out = self.base_model.swinViT(x, self.base_model.normalize)
        enc0 = self.base_model.encoder1(x)
        enc1 = self.base_model.encoder2(hidden_states_out[0])
        enc2 = self.base_model.encoder3(hidden_states_out[1])
        enc3 = self.base_model.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        enc5 = self.base_model.encoder10(hidden_states_out[4])

        # Upsample the extracted features using the decoder blocks
        dec1 = self.decoder1(enc0, enc1)
        dec2 = self.decoder2(dec1, enc2)
        dec3 = self.decoder3(dec2, enc3)
        dec4 = self.decoder4(dec3, enc4)
        dec5 = self.decoder5(dec4, enc5)

        # Apply Global Average Pooling on the extracted features
        out = F.adaptive_avg_pool3d(dec5, (1, 1, 1)).view(dec5.size(0), -1)

        # Pass through the fully connected 'neck'
        out = F.silu(self.fc(out))

        # Compute the classification logits using the head
        out = self.out(out)

        return out


class UnetDownBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=False,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class SwinViTMLP_v3(nn.Module):
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

    def __init__(self, n_classes: int = 2, img_size: tuple = (96, 96, 96), in_channels: int = 1, pretrained_weights: str = None):
        """
        Initialize the SwinViTmlp model with the given parameters.
        """
        super().__init__()

        feature_size = 48

        # Define Backbone (SwinUNETR)
        # The SwinUNETR model is used as a backbone to extract features from the input image.
        self.base_model = monai.networks.nets.SwinUNETR(
            img_size=img_size, in_channels=in_channels, out_channels=14, feature_size=feature_size, use_v2=True
        )

        # Load pretrained weights if provided
        if pretrained_weights:
            self.base_model.load_from(weights=torch.load(pretrained_weights))

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        # Define fully connected layers for classification.
        self.fc1 = nn.Linear(16 * feature_size, feature_size)
        self.fc2 = nn.Linear(8 * feature_size, feature_size)
        self.fc3 = nn.Linear(4 * feature_size, feature_size)
        self.fc4 = nn.Linear(2 * feature_size, feature_size)

        # Define head for classification.
        self.fc = nn.Linear(5 * feature_size, feature_size)
        self.out_class = nn.Linear(feature_size, n_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width)

        Returns:
            out (Tensor): Classification logits of shape (batch_size, n_classes)
        """
        hidden_states_out = self.base_model.swinViT(x, self.base_model.normalize)
        enc1 = self.base_model.encoder2(hidden_states_out[0])
        enc2 = self.base_model.encoder3(hidden_states_out[1])
        enc3 = self.base_model.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        enc5 = self.base_model.encoder10(hidden_states_out[4])

        out1 = F.adaptive_avg_pool3d(enc5, (1, 1, 1)).view(enc5.size(0), -1)
        out1 = F.silu(self.fc1(out1))

        out2 = F.adaptive_avg_pool3d(enc4, (1, 1, 1)).view(enc4.size(0), -1)
        out2 = F.silu(self.fc2(out2))

        out3 = F.adaptive_avg_pool3d(enc3, (1, 1, 1)).view(enc3.size(0), -1)
        out3 = F.silu(self.fc3(out3))

        out4 = F.adaptive_avg_pool3d(enc2, (1, 1, 1)).view(enc2.size(0), -1)
        out4 = F.silu(self.fc4(out4))

        out5 = F.adaptive_avg_pool3d(enc1, (1, 1, 1)).view(enc1.size(0), -1)

        out = torch.cat((out1, out2, out3, out4, out5), dim=1)

        out = F.silu(self.fc(out))

        out = self.out_class(out)

        return out


def test():
    model = SwinViTmlp(n_classes=2)
    input = torch.randn(3, 1, 96, 96, 96)
    out = model(input)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()
