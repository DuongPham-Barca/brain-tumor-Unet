import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x

    

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, \
            kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = nn.Module()
        self.encoder.conv1 = nn.Module()
        self.encoder.conv1.conv = ConvBlock(n_channels, 64)
        self.encoder.conv2 = nn.Module()
        self.encoder.conv2.conv = ConvBlock(64, 128)
        self.encoder.conv3 = nn.Module()
        self.encoder.conv3.conv = ConvBlock(128, 256)
        self.encoder.conv4 = nn.Module()
        self.encoder.conv4.conv = ConvBlock(256, 512)

        self.bottleneck = nn.Module()
        self.bottleneck.conv = ConvBlock(512, 512)

        self.decoder = nn.Module()
        self.decoder.up1 = Decoder(512, 256)
        self.decoder.up2 = Decoder(256, 128)
        self.decoder.up3 = Decoder(128, 64)
        self.decoder.up4 = Decoder(64, 32)

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder.conv1.conv(x)
        x1 = F.max_pool2d(skip1, 2)

        skip2 = self.encoder.conv2.conv(x1)
        x2 = F.max_pool2d(skip2, 2)

        skip3 = self.encoder.conv3.conv(x2)
        x3 = F.max_pool2d(skip3, 2)

        skip4 = self.encoder.conv4.conv(x3)
        x4 = F.max_pool2d(skip4, 2)

        x5 = self.bottleneck.conv(x4)

        x = self.decoder.up1(x5, skip4)
        x = self.decoder.up2(x, skip3)
        x = self.decoder.up3(x, skip2)
        x = self.decoder.up4(x, skip1)

        x = self.final(x)

        return x
    

