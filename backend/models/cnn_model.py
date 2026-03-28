import torch
import torch.nn as nn
import torch.nn.functional as F


class HPF(nn.Module):
    """High-Pass Filter layer — standard in steganalysis nets (KV kernel from Ye et al.)"""
    def __init__(self):
        super().__init__()
        # KV kernel - suppresses image content, amplifies residual noise
        kernel = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0
        self.register_buffer('weight', kernel.view(1, 1, 5, 5))

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.hpf = HPF()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
