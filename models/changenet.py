import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ChangeNet(nn.Module):
    def __init__(self, backbone="resnet18"):
        super(ChangeNet, self).__init__()

        # Load ResNet backbone (no classifier head)
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_channels = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_channels = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_channels = 2048
        else:
            raise ValueError(f"Backbone '{backbone}' not supported.")

        # Remove the final FC + pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ChangeNet input has 6 channels (A + B)
        self.input_proj = nn.Conv2d(6, 3, kernel_size=1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, a, b):
        # Merge A and B images into 6-channel tensor
        x = torch.cat([a, b], dim=1)  # [B, 6, H, W]

        # Project to 3-channel (to fit pretrained backbone)
        x = self.input_proj(x)  # [B, 3, H, W]

        x = self.backbone(x)    # [B, in_channels, H/32, W/32]
        x = self.decoder(x)     # [B, 32, H/2, W/2]
        x = self.out(x)         # [B, 1, H/2, W/2]

        # Upsample to match original input resolution
        x = F.interpolate(x, size=a.shape[2:], mode="bilinear", align_corners=False)

        return torch.sigmoid(x)
