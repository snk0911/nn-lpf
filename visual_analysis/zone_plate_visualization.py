import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
from torch.autograd import Function
from torch.nn import Module


def generate_zone_plate(size=64):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    r_squared = xv**2 + yv**2
    # Increased K to push frequency higher to force aliasing
    K = 0.5 * np.pi * size * 2.0 
    img = 0.5 + 0.5 * np.cos(K * r_squared / size)
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def visualize_comparison():
    input_img = generate_zone_plate(size=64)
    
    # 1. Naive Stride 2 (No filtering)
    naive_out = input_img[:, :, ::2, ::2]
    
    # 2. Average Pooling (Box Filter)
    # Kernel Size 2, Stride 2
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_out = avgpool(input_img)
    
    # 3. BlurPool (Binomial/Gaussian approximation)
    blurpool = BlurPool(channels=1, filter_size=4, stride=2)
    bp_out = blurpool(input_img)
    
    # 4. DWT (Haar)
    dwt = DWT_2D_tiny('haar')
    dwt_out = dwt(input_img)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(input_img.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Input Zone Plate"); axes[0].axis('off')
    
    axes[1].imshow(naive_out.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("1. Naive Stride 2\n(Severe Aliasing)"); axes[1].axis('off')
    
    axes[2].imshow(avg_out.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("2. Avg Pool (k=2)\n(Box Filter Aliasing)"); axes[2].axis('off')
    
    axes[3].imshow(bp_out.squeeze(), cmap='gray', cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("3. BlurPool (k=4)\n(Gaussian-like)"); axes[3].axis('off')
    
    axes[4].imshow(dwt_out.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[4].set_title("4. DWT (Haar)\n(Wavelet LPF)"); axes[4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_comparison()