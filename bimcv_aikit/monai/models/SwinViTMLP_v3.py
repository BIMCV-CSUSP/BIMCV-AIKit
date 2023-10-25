import torch.nn as nn
import torch.nn.functional as F
import monai
import torch
from monai.networks.blocks import UnetrBasicBlock

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

    def __init__(self, n_classes: int = 2, img_size: tuple = (96,96,96), in_channels: int = 1, pretrained_weights: str = None):
        """
        Initialize the SwinViTmlp model with the given parameters.
        """
        super().__init__()

        feature_size=48

        # Define Backbone (SwinUNETR)
        # The SwinUNETR model is used as a backbone to extract features from the input image.
        self.base_model = monai.networks.nets.SwinUNETR(
            img_size=img_size, 
            in_channels=in_channels, 
            out_channels=14, 
            feature_size=feature_size, 
            use_v2=True
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
    model = SwinViTMLP_v3(n_classes=2, pretrained_weights="/home/amora/alzheimer/model_swinvit.pt")
    input = torch.randn(2, 1, 96, 96, 96)
    out = model(input)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()
