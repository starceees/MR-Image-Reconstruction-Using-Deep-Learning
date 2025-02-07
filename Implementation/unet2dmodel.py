import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class Down2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down2D, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv2D(in_channels, out_channels)
        con
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up2D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up2D, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Note: in_channels is assumed to be from the concatenation, so use half for transposed conv.
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to have the same size as x2 (in case of odd dimensions)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, bilinear=True):
        super(UNet2D, self).__init__()
        self.inc = DoubleConv2D(in_channels, base_filters)
        self.down1 = Down2D(base_filters, base_filters * 2)
        self.down2 = Down2D(base_filters * 2, base_filters * 4)
        self.down3 = Down2D(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down2D(base_filters * 8, base_filters * 16 // factor)
        
        self.up1 = Up2D(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up2D(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up2D(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up2D(base_filters * 2, base_filters, bilinear)
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
