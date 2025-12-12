# BIDR Relight
# 12/11/25
# CS7180 Advanced Perception
# Contributors: Max Huber, Adharsh Kandula, Richard Zhao

# This file creates a UNet model using ResNet50 as the backbone for ISD estimation as 
# seen done through Michael Massone's thesis work. 
# Several components were coded/modified with the help of GPT-5 and Claude Sonnet 4.5


import logging
import torch
import torch.nn as nn
from torchvision.models import resnet50


# ============================================================================================================
# ResNet UNet
# ============================================================================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale.to(dtype=x.dtype, device=x.device)


################################################################################################################################################################


class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, se_block=False, dropout=0.0):
        super().__init__()
        self.se_block = se_block
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        if se_block:
            self.se = SEBlock(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if self.se_block:
            return self.se(x)
        return x


################################################################################################################################################################


class ResNet50UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        pretrained=True,
        checkpoint=None,
        se_block=True,
        dropout=0.0,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initializing {self.__class__.__name__} | "
            f"in_channels={in_channels}, out_channels={out_channels}, "
            f"pretrained={pretrained}, checkpoint={checkpoint}, se_block={se_block}"
        )
        # Load backbone
        if checkpoint:
            self.logger.info(f"Loading ResNet50 weights from checkpoint: {checkpoint}")
            resnet = resnet50(weights=None)
            state_dict = torch.load(checkpoint, map_location="cpu")

            # OLD CODE
            # resnet.load_state_dict(
            #     state_dict.get("state_dict", state_dict), strict=False
            # )
        else:
            weights = None
            resnet = resnet50(weights=weights)
            state_dict = None

        # Encoder
        self.in_conv = nn.Sequential(
            resnet.conv1,  # 64 x H/2
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool  # H/4
        self.enc1 = resnet.layer1  # 256 x H/4
        self.enc2 = resnet.layer2  # 512 x H/8
        self.enc3 = resnet.layer3  # 1024 x H/16
        self.enc4 = resnet.layer4  # 2048 x H/32

        # Decoder
        self.up4 = UpBlock(2048, 1024, 512, se_block=se_block, dropout=dropout)
        self.up3 = UpBlock(512, 512, 256, se_block=se_block, dropout=dropout)
        self.up2 = UpBlock(256, 256, 128, se_block=se_block, dropout=dropout)
        self.up1 = UpBlock(128, 64, 64, se_block=se_block, dropout=dropout)
        self.up0 = UpBlock(
            64, 0, 32, se_block=se_block, dropout=dropout
        )  # No skip connection here
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        # Optionally re-train input conv to accept different channels
        if in_channels != 3:
            self.in_conv[0] = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self._log_parameter_count()

        # Loading pretrained weights
        if state_dict is not None:
            self.load_state_dict(state_dict["model_state_dict"])

    def _log_parameter_count(self):
        """
        Logs total and trainable parameters in the model, summarized by top-level modules.
        """
        self.logger.info(
            f"{self.__class__.__name__} Parameter Summary (Top-Level Modules):"
        )
        self.logger.info("-" * 80)
        total_params = 0
        trainable_params = 0

        for name, module in self.named_children():  # Only top-level children
            mod_total = sum(p.numel() for p in module.parameters())
            mod_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            total_params += mod_total
            trainable_params += mod_trainable
            self.logger.info(
                f"{name:<25} | Total: {mod_total:<20} | Trainable: {mod_trainable:,}"
            )

        self.logger.info("-" * 80)
        self.logger.info(f"Total Parameters:     {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")

    def forward(self, x):
        # Encoder
        self.logger.debug(f"{self.__class__.__name__} input shape = {x.shape}")
        x0 = self.in_conv(x)  # conv1
        x1 = self.enc1(self.maxpool(x0))  # layer1
        x2 = self.enc2(x1)  # layer2
        x3 = self.enc3(x2)  # layer3
        x4 = self.enc4(x3)  # layer4
        self.logger.debug(f"{self.__class__.__name__} bottlneck shape = {x4.shape}")
        # Decoder with skip connections
        d4 = self.up4(x4, x3)  # skip enc layer4
        d3 = self.up3(d4, x2)  # skip enc layer2`
        d2 = self.up2(d3, x1)  # skip enc layer1
        d1 = self.up1(d2, x0)
        d0 = self.up0(d1, None)  # no skip

        out = self.final_conv(d0)  # regression output
        self.logger.debug(f"{self.__class__.__name__} output shape = {out.shape}")

        return out
