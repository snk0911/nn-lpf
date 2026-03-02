import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

def get_blurpool_coeffs(filter_size):
    """
    Replicates the coefficient generation from the BlurPool class.
    Returns 1D and 2D filter coefficients.
    """
    if filter_size == 1: a = np.array([1.,])
    elif filter_size == 2: a = np.array([1., 1.])
    elif filter_size == 3: a = np.array([1., 2., 1.])
    elif filter_size == 4: a = np.array([1., 3., 3., 1.])
    elif filter_size == 5: a = np.array([1., 4., 6., 4., 1.])
    elif filter_size == 6: a = np.array([1., 5., 10., 10., 5., 1.])
    elif filter_size == 7: a = np.array([1., 6., 15., 20., 15., 6., 1.])
    else: raise ValueError("Invalid filter size")
    
    # Normalize
    filt_1d = a / np.sum(a)
    # Create 2D kernel via outer product (separable filter)
    filt_2d = np.outer(filt_1d, filt_1d)
    
    return filt_1d, filt_2d

def get_dwt_coeffs(wavelet_type):
    """
    Replicates the coefficient generation from the DWT_2D_tiny class.
    Note: Your code uses wavelet.rec_lo (reconstruction low pass).
    """
    wavelet = pywt.Wavelet(wavelet_type)
    # Your implementation specifically pulls rec_lo for the forward pass matrix
    coeffs = np.array(wavelet.rec_lo)
    
    # Normalize for visualization (energy scaling)
    # Wavelet filters usually have norm sqrt(2). We normalize to sum=1 for consistent plotting.
    filt_1d = coeffs / np.sum(coeffs)
    filt_2d = np.outer(filt_1d, filt_1d)
    
    return filt_1d, filt_2d

def plot_freq_response_1d(ax, kernel_1d, label):
    """
    Plots the 1D magnitude frequency response.
    """
    # Compute frequency response using FFT
    # Pad to 512 points for a smooth curve
    n_pts = 512
    h = np.abs(np.fft.fft(kernel_1d, n_pts))
    h = np.fft.fftshift(h)
    
    # Normalize so DC gain is 1 (0 dB)
    h = h / np.max(h)
    
    # Frequency axis
    freq = np.linspace(-0.5, 0.5, n_pts)
    
    ax.plot(freq, 20 * np.log10(h + 1e-10), label=label, linewidth=2)
    ax.set_xlabel('Normalized Frequency (cycles/pixel)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_title('1D Frequency Response')

def plot_freq_response_2d(ax, kernel_2d, title):
    """
    Plots the 2D magnitude frequency response as a heatmap.
    """
    n_pts = 256
    # Pad kernel to n_pts x n_pts
    pad_w = (n_pts - kernel_2d.shape[0]) // 2
    padded = np.pad(kernel_2d, ((pad_w, n_pts - kernel_2d.shape[0] - pad_w), 
                                (pad_w, n_pts - kernel_2d.shape[1] - pad_w)), 
                    'constant')
    
    # Compute 2D FFT and shift to center
    f = np.fft.fft2(padded)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log10(np.abs(fshift) + 1e-10)
    
    # Normalize to 0 dB peak
    magnitude = magnitude - np.max(magnitude)
    
    # Display
    im = ax.imshow(magnitude, cmap='jet', extent=[-0.5, 0.5, -0.5, 0.5], vmin=-60, vmax=0)
    ax.set_title(title)
    ax.set_xlabel('Frequency u')
    ax.set_ylabel('Frequency v')
    return im

# ==========================================
# Main Execution
# ==========================================

# 1. Setup Plot
fig_1d, ax_1d = plt.subplots(figsize=(8, 6))
fig_2d, axs_2d = plt.subplots(1, 2, figsize=(12, 5))

# 2. Plot BlurPool (Filter Size 3)
bp_1d, bp_2d = get_blurpool_coeffs(filter_size=3)
plot_freq_response_1d(ax_1d, bp_1d, label='BlurPool (Size 3)')
im = plot_freq_response_2d(axs_2d[0], bp_2d, 'BlurPool (Size 3)')

# 3. Plot DWT (Haar - Default in your code)
dwt_1d, dwt_2d = get_dwt_coeffs('haar')
plot_freq_response_1d(ax_1d, dwt_1d, label='DWT (Haar)')
plot_freq_response_2d(axs_2d[1], dwt_2d, 'DWT (Haar)')

# 4. Final Touches
ax_1d.set_ylim(-60, 5) # Limit y-axis to see stopband attenuation
ax_1d.set_title('Comparison of Low Pass Filters')

# Add colorbar to 2D plot
fig_2d.colorbar(im, ax=axs_2d.ravel().tolist(), shrink=0.6, label='Magnitude (dB)')

plt.show()