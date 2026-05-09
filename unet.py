import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)   # (64, 256, 256)
        x2 = self.conv2(self.pool1(x1))  # (128, 128, 128)
        x3 = self.conv3(self.pool2(x2))  # (256, 64, 64)
        x4 = self.conv4(self.pool3(x3))  # (512, 32, 32)

        out = self.pool4(x4)

        return x1, x2, x3, x4, out

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = ConvBlock(512, 512) # was 512→1024, now 512→512
        self.dropout = nn.Dropout2d(p=0.3) # not normal Dropout

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # FIX: match size
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = UpBlock(512, 512, 256)   # was UpBlock(1024, 512, 512)
        self.up2 = UpBlock(256, 256, 128)   # was UpBlock(512,  256, 256)
        self.up3 = UpBlock(128, 128, 64)    # was UpBlock(256,  128, 128)
        self.up4 = UpBlock(64,   64,  32)   # was UpBlock(128,   64,  64)

    def forward(self, x, x1, x2, x3, x4):
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

        self.final = nn.Conv2d(32, 1, kernel_size=1) # was Conv2d(64, 1, ...)

    def forward(self, x):
        
        x1, x2, x3, x4, out = self.encoder(x)
        
        out = self.bottleneck(out)
        
        out = self.decoder(out, x1, x2, x3, x4) 

        return self.final(out)