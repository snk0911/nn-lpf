import torch
import torch.nn as nn
import torch.nn.functional as F


class PerChannelSoftPool(nn.Module):
    """
    Pure PyTorch implementation of SoftPool.
    This version processes channels independently (Spatial SoftPool).
    """
    def __init__(self, channels, kernel_size=3, stride=2):
        super(PerChannelSoftPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        B, C, H, W = x.shape

        # 1. Unfold extracts sliding patches
        # Output shape: (Batch, Channels * Kernel * Kernel, L)
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )

        # 2. Reshape to separate Channels and Kernel dimensions
        # We infer L (number of patches) from the last dimension
        L = x_unfold.size(-1)
        K = self.kernel_size
        
        # Reshape (B, C*K*K, L) -> (B, C, K*K, L)
        x_unfold = x_unfold.view(B, C, K * K, L)
        
        # Reshape to (B, C, Kernel_H, Kernel_W, L) for spatial weighting
        x_unfold = x_unfold.view(B, C, K, K, L)

        # 3. Calculate SoftMax weights Spatially
        #    Dimensions 2 and 3 are Height/Width of kernel
        exp_x = torch.exp(x_unfold)
        sum_exp = torch.sum(exp_x, dim=[2, 3], keepdim=True) + 1e-12
        weights = exp_x / sum_exp

        # 4. Weighted Sum
        #    Sum over the kernel dimensions (2 and 3)
        out = torch.sum(x_unfold * weights, dim=[2, 3])

        # 5. Reshape back to image format
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        return out.view(B, C, H_out, W_out)
