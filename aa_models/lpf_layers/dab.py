import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Callable, Optional, Union, Dict
from torch import Tensor

"""
Note: AA-Relu is not implemented here, since only low-pass filters are used in the final experiments
"""
class DABPool(nn.Module):
    """Depth Adaptive Blur-Pool: blur with learnable sigma, then stride-subsample."""
    def __init__(self, channels, stride=2, depth_index=0, dab_controller=None):
        super().__init__()

        if dab_controller is None:
            raise ValueError("DABPool requires a dab_controller")

        self.controller = dab_controller
        self.depth_index = depth_index
        self.stride = stride
        self.blur = GaussianBlur2d(channels, kernel_size=3)

    def forward(self, x):
        sigma = self.controller.get_sigma(self.depth_index)
        x = self.blur(x, sigma)
        return x[:, :, ::self.stride, ::self.stride]
    

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
    """Applies a 2D Gaussian Blur."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = (kernel_size // 2)

    def forward(self, x, sigma):
        ax = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1., device=x.device, dtype=x.dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2 + 1e-8))
        kernel = kernel / kernel.sum()
        
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        
        return F.conv2d(x, weight=kernel, groups=self.channels, padding=self.padding)
