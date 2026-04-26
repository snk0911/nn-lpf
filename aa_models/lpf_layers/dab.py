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
        
        # Calculate init value to satisfy sigma[0] = 0.5 (D=1)
        # softplus(x) = 0.5  =>  x = ln(e^0.5 - 1)
        init_val = math.log(math.exp(0.5) - 1)
        self.deltas = nn.Parameter(torch.full((num_downsample_layers,), init_val))

    def get_sigma(self, depth_index):
        positive_increments = F.softplus(self.deltas)
        cumulative_sigmas = torch.cumsum(positive_increments, dim=0)
        return cumulative_sigmas[depth_index]


class GaussianBlur2d(nn.Module):
    """Applies a 2D Gaussian Blur with dynamic sigma and selectable kernel size."""
    def __init__(self, channels, kernel_size=3, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding
        
        # Create grid coordinates.
        ax = torch.arange(0, kernel_size, dtype=torch.float32) - (kernel_size - 1) // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # Stack and register buffer (shape: 2 x k x k)
        self.register_buffer('grid', torch.stack([xx, yy], dim=0))

    def forward(self, x, sigma):
        # grid shape: (2, k, k)
        
        # Calculate squared distances from center
        d_sq = self.grid[0]**2 + self.grid[1]**2
        
        # Gaussian formula
        # FIX: Using clamp(min=1e-4) as a minimal, non-intrusive safety net 
        # against division-by-zero and gradient explosion if sigma collapses.
        kernel = torch.exp(-d_sq / (2 * sigma.clamp(min=1e-4)**2))
        
        # Normalize the kernel so it sums to 1
        kernel = kernel / kernel.sum()
        
        # Reshape for conv2d: (out_channels, in_channels/groups, kH, kW)
        # Since groups=channels, we repeat the kernel for each channel.
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        
        return F.conv2d(x, weight=kernel, groups=self.channels, padding=self.padding)