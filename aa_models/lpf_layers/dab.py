import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DABPool(nn.Module):
    def __init__(self, channels, stride=2, depth_index=0, dab_controller=None, filter_size=3, padding=None):
        super().__init__()
        
        # If padding is not provided, calculate it automatically ("same" padding)
        if padding is None:
            padding = filter_size // 2
            
        # FIX: Bypass nn.Module's __setattr__ to prevent registering the shared 
        # dab_controller as a submodule of this specific DABPool instance.
        # This avoids the "module already has a parent" conflict while still 
        # allowing us to access it via self.controller in forward().
        object.__setattr__(self, 'controller', dab_controller)
        
        self.depth_index = depth_index
        self.stride = stride
        
        # Use the calculated or provided padding
        self.blur = GaussianBlur2d(channels, kernel_size=filter_size, padding=padding)

    def forward(self, x):
        sigma = self.controller.get_sigma(self.depth_index)
        return self.blur(x, sigma, stride=self.stride)
    

class DABSigmaController(nn.Module):
    """
    Manages the learnable sigma parameters for the DAB-Pool layers.
    Enforces the paper's constraint: G_sigma_1 < G_sigma_2 < ...
    Initialization follows the paper: sigma_D = D / 2
    """
    def __init__(self, num_downsample_layers):
        super().__init__()
        self.num_layers = num_downsample_layers
        
        # Calculate init value to satisfy sigma[0] = 0.5 (D=1)
        # softplus(x) = 0.5  =>  x = ln(e^0.5 - 1)
        init_val = math.log(math.exp(0.5) - 1)
        self.deltas = nn.Parameter(torch.full((num_downsample_layers,), init_val))

    def get_sigma(self, depth_index):
        positive_increments = F.softplus(self.deltas)
        cumulative_sigmas = torch.cumsum(positive_increments, dim=0)
        return cumulative_sigmas[depth_index]


class GaussianBlur2d(nn.Module):
    """Depthwise Gaussian blur with dynamically learned sigma."""

    def __init__(self, channels, kernel_size=3, padding=0):
        super().__init__()

        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding

        ax = (
            torch.arange(kernel_size, dtype=torch.float32)
            - (kernel_size - 1) // 2
        )

        xx, yy = torch.meshgrid(
            ax,
            ax,
            indexing='ij'
        )

        # Squared distance from the kernel center.
        self.register_buffer(
            'd_sq',
            xx.square() + yy.square()
        )

    def forward(self, x, sigma, stride=1):
        # sigma is guaranteed to be > 0 by
        # softplus + cumulative sum in DABSigmaController.
        kernel = torch.exp(
            -self.d_sq / (2.0 * sigma.square())
        )

        kernel = kernel / kernel.sum()

        kernel = kernel.view(
            1,
            1,
            self.kernel_size,
            self.kernel_size
        ).expand(
            self.channels,
            1,
            self.kernel_size,
            self.kernel_size
        )

        return F.conv2d(
            x,
            weight=kernel,
            stride=stride,
            padding=self.padding,
            groups=self.channels
        )