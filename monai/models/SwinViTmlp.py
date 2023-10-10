import torch.nn as nn
import torch.nn.functional as F
import monai
import torch


class SwinViTmlp(nn.Module):
    def __init__(self, n_classes: int = 2, pretrained_weights: str =None):
        super(SwinViTmlp, self).__init__()

        # Define Backbone (SwinUNETR)
        backbone = monai.networks.nets.SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48, use_v2=True)
        if pretrained_weights:
            backbone.load_from(weights=torch.load(pretrained_weights))

        self.backbone = backbone.swinViT
        # Remove FC layer

        # Define 'necks' for each head
        self.fc = nn.Linear(768, 32)

        # Define heads

        self.out = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.backbone(x)[4]
        # print(x.shape)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)  # Global Average Pooling
        # print(x.shape)

        x = F.silu(self.fc(x))

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
