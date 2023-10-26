import monai
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


######################################################################
# Preconfigured MONAI MODELS
######################################################################
class DenseNet(BaseModel):
    def __init__(self, num_classes):
        self.model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=4, out_channels=num_classes, dropout_prob=0.2)

    def forward(self, image):
        return self.model(image)


######################################################################
# Manual Models PyTorch
######################################################################


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Bilinear3D(nn.Module):
    def __init__(self, num_classes):
        super(Bilinear3D, self).__init__()
        self.conv3d_5_2 = nn.ModuleList(
            [nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(5, 5, 5), stride=(2, 2, 2), padding="valid") for _ in range(2)]
        )
        self.conv3d_3_1 = nn.ModuleList(
            [nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="valid") for _ in range(12)]
        )
        self.bn3d = nn.ModuleList([nn.BatchNorm3d(num_features=32) for _ in range(14)])
        self.avg = nn.ModuleList([nn.AvgPool3d(kernel_size=(2, 2, 2)) for _ in range(6)])
        self.dense_100 = nn.Linear(in_features=512, out_features=100)
        self.dense = nn.Linear(in_features=100, out_features=num_classes)
        self.bn = nn.BatchNorm1d(num_features=100)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv3d_5_2[0](x)
        x1 = self.bn3d[0](x1)
        x1 = self.conv3d_3_1[0](x1)
        x1 = self.bn3d[1](x1)
        x1 = self.conv3d_3_1[1](x1)
        x1 = self.bn3d[2](x1)
        x1 = self.avg[0](x1)
        x1 = self.conv3d_3_1[2](x1)
        x1 = self.bn3d[3](x1)
        x1 = self.conv3d_3_1[3](x1)
        x1 = self.bn3d[4](x1)
        x1 = self.avg[1](x1)
        x1 = self.conv3d_3_1[4](x1)
        x1 = self.bn3d[5](x1)
        x1 = self.conv3d_3_1[5](x1)
        x1 = self.bn3d[6](x1)
        x1 = self.avg[2](x1)
        x1 = torch.flatten(x1, start_dim=1)

        x2 = self.conv3d_5_2[1](x)
        x2 = self.bn3d[7](x2)
        x2 = self.conv3d_3_1[6](x2)
        x2 = self.bn3d[8](x2)
        x2 = self.conv3d_3_1[7](x2)
        x2 = self.bn3d[9](x2)
        x2 = self.avg[3](x2)
        x2 = self.conv3d_3_1[8](x2)
        x2 = self.bn3d[10](x2)
        x2 = self.conv3d_3_1[9](x2)
        x2 = self.bn3d[11](x2)
        x2 = self.avg[4](x2)
        x2 = self.conv3d_3_1[10](x2)
        x2 = self.bn3d[12](x2)
        x2 = self.conv3d_3_1[11](x2)
        x2 = self.bn3d[13](x2)
        x2 = self.avg[5](x2)
        x2 = torch.flatten(x2, start_dim=1)

        x = torch.cat((x1, x2), dim=1)

        x = self.dense_100(x)
        x = self.bn(x)
        x = self.dense(x)
        return self.softmax(x)
