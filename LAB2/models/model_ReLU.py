# TODO implement EEGNet model
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_classes=2, channels=2, samples=750, alpha=0.1, dropout=0.35):
        super(EEGNet, self).__init__()

        # ---------- Block 1 ----------
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1),
                      padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )

        # ---------- Block 2 ----------
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(channels, 1), stride=(1, 1),
                      groups=8, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropout)
        )

        # ---------- Block 3 ----------
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1),
                      padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout)
        )

        # ---------- Flatten ----------
        # 自動計算 flatten 維度
        with torch.no_grad():
            x = torch.zeros(1, 1, channels, samples)
            x = self._forward_features(x)
            flat_dim = x.shape[1]

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, num_classes)
        )

    def _forward_features(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x


# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self, num_classes=2, channels=2, samples=750):
        super(DeepConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1,5), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1,5), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1,5), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4)
        )

        self.fc = nn.Linear(8600, num_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x



