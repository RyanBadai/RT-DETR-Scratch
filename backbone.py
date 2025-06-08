import torch
import torch.nn as nn

# ResNet-like Backbone
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 128, 2, stride=2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 1024, 2, stride=2)

        # Add 1x1 convolutions to adjust channel sizes
        self.adjust_s4 = nn.Conv2d(256, 128, kernel_size=1)
        self.adjust_s5 = nn.Conv2d(512, 256, kernel_size=1)
        self.adjust_s6 = nn.Conv2d(1024, 512, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        s3 = self.layer1(x)
        s4 = self.layer2(s3)
        s5 = self.layer3(s4)
        s6 = self.layer4(s5)

        # Adjust channel sizes
        s4 = self.adjust_s4(s4)
        s5 = self.adjust_s5(s5)
        s6 = self.adjust_s6(s6)

        return [s4, s5, s6]