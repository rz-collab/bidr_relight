# BIDR Relight
# 12/11/25
# CS7180 Advanced Perception
# Contributors: Max Huber, Adharsh Kandula, Richard Zhao

# This file creates a mock model for estimating the ISD of an image using a neural network.
# Several components were coded/modified with the help of GPT-5 and Claude Sonnet 4.5


import torch
import torch.nn as nn


# ============================================================================================================
# Mock ISD Estimator
# ============================================================================================================
class MockISDModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # A physically plausible constant ISD direction for outdoor afternoon daylight
        isd = torch.tensor([0.58, 0.56, 0.59], dtype=torch.float32)
        isd = isd / torch.norm(isd)  # normalize to unit vector

        # Register as buffer so it moves to CUDA with the model but is not updated
        self.register_buffer("isd_vec", isd.view(1, 3, 1, 1))

    def forward(self, x):
        """
        x: (B, 3, H, W)
        Returns (B, 3, H, W) constant ISD map
        """
        B, _, H, W = x.shape
        return self.isd_vec.expand(B, 3, H, W)
