import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Callable, Optional, Union, Dict
from torch import Tensor


class DABPool(nn.Module):
    """Depth Adaptive Blur-Pool."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                 depth_index=0, dab_controller=None):
        super().__init__()
        if dab_controller is None:
            raise ValueError("DABPool requires a dab_controller")
        self.controller = dab_controller
        self.depth_index = depth_index
        self.blur = GaussianBlur2d(in_channels, kernel_size=3)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        current_sigma = self.controller.get_sigma(self.depth_index)
        x = self.blur(x, current_sigma)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DABSigmaController(nn.Module):
    """
    Manages the learnable sigma parameters for the DAB-Pool layers.
    Enforces the paper's constraint: G_sigma_1 < G_sigma_2 < ...
    Initialization follows the paper: sigma_D = D / 2
    """
    def __init__(self, num_downsample_layers):
        super().__init__()
        self.num_layers = num_downsample_layers
        self.base_sigma = 0.0
        
        # Calculate init value to satisfy sigma[d] = (d+1) / 2
        # softplus(x) = 0.5  =>  x = ln(e^0.5 - 1)
        init_val = math.log(math.exp(0.5) - 1)
        self.deltas = nn.Parameter(torch.full((num_downsample_layers,), init_val))

    def get_sigma(self, depth_index):
        positive_increments = F.softplus(self.deltas)
        cumulative_sigmas = torch.cumsum(positive_increments, dim=0)
        return cumulative_sigmas[depth_index]


class GaussianBlur2d(nn.Module):
    """Applies a 2D Gaussian Blur with 3x3 kernel."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = (kernel_size // 2)

    def forward(self, x, sigma):
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=x.device, dtype=x.dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        return F.conv2d(x, weight=kernel, groups=self.channels, padding=self.padding)


class AARelu(nn.Module):
    """Anti-Aliasing ReLU (AA-ReLU) from Equation 9."""
    def __init__(self, alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        alpha = self.alpha
        t1 = 0.0
        t2 = alpha
        t3 = alpha * math.exp(math.pi / 2)
        
        mask_neg = (x < t1)
        mask_linear = (x >= t1) & (x < t2)
        mask_roll = (x >= t2) & (x <= t3)
        mask_sat = (x > t3)
        
        out = torch.zeros_like(x)
        out[mask_linear] = x[mask_linear]
        
        if mask_roll.any():
            x_roll = x[mask_roll]
            out[mask_roll] = alpha * torch.sin(torch.log(x_roll / alpha)) + alpha
            
        out[mask_sat] = 2.0 * alpha
        return out

