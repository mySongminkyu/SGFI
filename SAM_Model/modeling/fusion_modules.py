import torch
import torch.nn as nn
from .sam2_utils import LayerNorm2d


class SemanticInjector(nn.Module):
    """
    Semantic Masking Injector for high-resolution feature fusion.

    Generates a soft attention mask from low-resolution features and uses it
    to selectively inject high-resolution detail into the decoder upsampling path.

    Fused = Low + (High * Mask * scale_factor)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )
        nn.init.constant_(self.mask_generator[-1].bias, 0.0)
        self.scale_factor = nn.Parameter(torch.tensor(2.0))

    def forward(self, low_res, high_res):
        mask  = torch.sigmoid(self.mask_generator(low_res))
        fused = low_res + (high_res * mask * self.scale_factor)
        return fused, mask
