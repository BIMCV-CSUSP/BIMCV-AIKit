import monai
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinUnetr_multitask(nn.Module):
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
        super(SwinUnetr_multitask, self).__init__()

        # Define Backbone (SwinUNETR)
        self.drop_rate = drop_rate
        self.model = monai.networks.nets.SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=seg_channels,
            feature_size=feature_size,
            use_v2=True,
            drop_rate=drop_rate,
        )
        if pretrained_weights:
            self.model.load_from(weights=torch.load(pretrained_weights))

        self.backbone = self.model.swinViT

        self.fc1 = nn.Linear(16 * feature_size, 8 * feature_size)
        self.fc2 = nn.Linear(8 * feature_size, 4 * feature_size)
        self.fc3 = nn.Linear(4 * feature_size, 2 * feature_size)
        self.fc4 = nn.Linear(2 * feature_size, feature_size)
        # Define heads

        self.out_class = nn.Linear(feature_size, n_classes)

    def forward(self, x_in):
        hidden_states_out = self.backbone(x_in)
        # print(x.shape)
        # Classification task
        x1 = F.adaptive_avg_pool3d(hidden_states_out[4], (1, 1, 1)).view(
            hidden_states_out[4].size(0), -1
        )  # Global Average Pooling
        x1 = F.dropout(F.relu(self.fc1(x1)), self.drop_rate)
        x1 = F.dropout(F.relu(self.fc2(x1)), self.drop_rate)
        x1 = F.dropout(F.relu(self.fc3(x1)), self.drop_rate)
        x1 = F.dropout(F.relu(self.fc4(x1)), self.drop_rate)
        class_out = self.out_class(x1)

        # Segmentation task
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

        return class_out, logits_seg


def test():
    model = SwinUnetr_multitask(n_classes=2)
    input = torch.randn(3, 1, 96, 96, 96)
    out = model(input)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()
