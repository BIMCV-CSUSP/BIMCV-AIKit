import torch
import torch.nn as nn
from monai.networks.nets import ViT, EfficientNetBN


class EfficientViT(nn.Module):
    """
    Experimental architecture for image classification with an EfficientNet backbone for feature extraction
    and a vanilla Vision Transformer for classification.

    Designed for grayscale 3D volumes (i.e. estructural brain MRI)

    Args:
        model_name (str): model version (defaults to efficientnet-b0)
        n_classes (int): number of output classes (defaults to 2)
        pretrained_weights_path (str): path to EfficientNet pretrained weights (Optional, defaults to None)
    """

    def __init__(self, model_name: str = "efficientnet-b0", n_classes: int = 2, pretrained_weights_path: str = None):
        super(EfficientViT, self).__init__()
        EfficientNet = EfficientNetBN(model_name=model_name, pretrained=False, progress=False, spatial_dims=3, in_channels=1, num_classes=n_classes)
        if pretrained_weights_path:
            EfficientNet.load_state_dict(torch.load(pretrained_weights_path))
        layers = list(EfficientNet.children())[:3] + list(EfficientNet._blocks.children())[:2]
        self.features = nn.Sequential(*layers)
        self.vit = ViT(in_channels=24, img_size=(23, 23, 23), patch_size=4, num_classes=n_classes, classification=True)

    def forward(self, x):
        x = self.features(x)
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
