# -*- coding: utf-8 -*-
"""
ReLU High-Frequency Reintroduction Visualization
Demonstrates that ReLU reintroduces high-frequency content after convolution,
supporting the theoretical argument from Chaman and Dokmanic (2021).

Layout:
    Col 1: Baseline Before ReLU (top) / Baseline After ReLU (bottom)
    Col 2: Bin-5 Before ReLU (top)    / Bin-5 After ReLU (bottom)
    Row 3: Radial frequency profile (full width)

Usage:
    python relu_reintroduction.py --model_baseline path/to/baseline.pth
                                  --model_bin5 path/to/bin5.pth
                                  --image path/to/image.jpeg
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
import sys
from PIL import Image
import torchvision.transforms as transforms

# ---- CONFIG ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT     = os.path.join(BASE_DIR, 'saved_checkpoints')

BASELINE_WEIGHTS = os.path.join(CKPT, 'baseline.pth.tar')
BIN5_WEIGHTS     = os.path.join(CKPT, 'blur4.pth.tar')

OUT_DIR  = os.path.join(BASE_DIR, 'plots')
GPU      = None

MEAN = [0.4802, 0.4481, 0.3975]
STD  = [0.2764, 0.2689, 0.2816]
# ------------------------------------------------------------------------------

sys.path.insert(0, BASE_DIR)


# ---- Load model and register hooks -------------------------------------------
def load_model_with_hooks(weights_path, aa_type='none', filter_size=1, gpu=None):
    import aa_models
    model = aa_models.resnet18(
        aa_type=aa_type,
        filter_size=filter_size,
        num_classes=200
    )
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if gpu is not None:
        model = model.cuda(gpu)

    activations = {}

    def hook_fn(name):
        def fn(module, input, output):
            activations[name] = output.detach().cpu()
        return fn

    model.layer1[0].bn1.register_forward_hook(hook_fn('before_relu'))
    model.layer1[0].relu.register_forward_hook(hook_fn('after_relu'))

    return model, activations


# ---- Load and preprocess image -----------------------------------------------
def load_image(image_path, gpu=None):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    if gpu is not None:
        tensor = tensor.cuda(gpu)
    return tensor


# ---- Feature map to spatial --------------------------------------------------
def to_spatial(fm_tensor):
    fm_np = fm_tensor[0].numpy()
    spatial = fm_np.mean(axis=0)
    s_min, s_max = spatial.min(), spatial.max()
    if s_max > s_min:
        spatial = (spatial - s_min) / (s_max - s_min)
    return spatial


# ---- Radial frequency profile ------------------------------------------------
def compute_radial_profile(fm_tensor):
    fm_np = fm_tensor[0].numpy()
    C, H, W = fm_np.shape
    window_2d = np.outer(np.hanning(H), np.hanning(W))

    fft_mags = []
    for c in range(C):
        fm_windowed = fm_np[c] * window_2d
        F = np.fft.fft2(fm_windowed)
        F_shifted = np.fft.fftshift(F)
        mag = np.log1p(np.abs(F_shifted))
        fft_mags.append(mag)

    fft_mean = np.mean(fft_mags, axis=0)

    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cy, cx)

    profile = np.array([
        fft_mean[r == i].mean() if (r == i).any() else 0.0
        for i in range(max_r)
    ])

    radii = np.linspace(0, 0.5, max_r)
    return radii, profile


# ---- Main --------------------------------------------------------------------
def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading image: {args.image}")
    image = load_image(args.image, gpu=GPU)

    print("Loading Baseline model...")
    baseline_model, baseline_acts = load_model_with_hooks(
        args.model_baseline, aa_type='none', filter_size=1, gpu=GPU
    )

    print("Loading Bin-5 model...")
    bin5_model, bin5_acts = load_model_with_hooks(
        args.model_bin5, aa_type='blur', filter_size=4, gpu=GPU
    )

    print("Running forward passes...")
    with torch.no_grad():
        baseline_model(image)
        bin5_model(image)

    # Spatial feature maps
    baseline_before = to_spatial(baseline_acts['before_relu'])
    baseline_after  = to_spatial(baseline_acts['after_relu'])
    bin5_before     = to_spatial(bin5_acts['before_relu'])
    bin5_after      = to_spatial(bin5_acts['after_relu'])

    # Radial profiles
    radii_bb, prof_bb = compute_radial_profile(baseline_acts['before_relu'])
    radii_ba, prof_ba = compute_radial_profile(baseline_acts['after_relu'])
    radii_5b, prof_5b = compute_radial_profile(bin5_acts['before_relu'])
    radii_5a, prof_5a = compute_radial_profile(bin5_acts['after_relu'])

    # ---- Plot ----------------------------------------------------------------
    # Layout:
    # Row 0: Baseline Before ReLU | Bin-5 Before ReLU
    # Row 1: Baseline After ReLU  | Bin-5 After ReLU
    # Row 2: Radial profile (full width)

    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(
        'ReLU High-Frequency Reintroduction After Low-Pass Filtering',
        fontsize=13, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.15,
                           height_ratios=[1, 1, 1.4])

    # Row 0 -- Before ReLU
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(baseline_before, cmap='gray', vmin=0, vmax=1)
    ax00.set_title('Baseline\nBefore ReLU (after conv+bn)', fontsize=10)
    ax00.axis('off')

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(bin5_before, cmap='gray', vmin=0, vmax=1)
    ax01.set_title('Bin-5\nBefore ReLU (after conv+bn)', fontsize=10)
    ax01.axis('off')

    # Row 1 -- After ReLU
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(baseline_after, cmap='gray', vmin=0, vmax=1)
    ax10.set_title('Baseline\nAfter ReLU', fontsize=10)
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(bin5_after, cmap='gray', vmin=0, vmax=1)
    ax11.set_title('Bin-5\nAfter ReLU', fontsize=10)
    ax11.axis('off')

    # Row 2 -- Radial frequency profile
    ax_p = fig.add_subplot(gs[2, :])

    ax_p.plot(radii_bb, prof_bb, color='black',     linestyle='--', linewidth=2,
              label='Baseline -- Before ReLU')
    ax_p.plot(radii_ba, prof_ba, color='black',     linestyle='-',  linewidth=2,
              label='Baseline -- After ReLU')
    ax_p.plot(radii_5b, prof_5b, color='royalblue', linestyle='--', linewidth=2,
              label='Bin-5 -- Before ReLU')
    ax_p.plot(radii_5a, prof_5a, color='royalblue', linestyle='-',  linewidth=2,
              label='Bin-5 -- After ReLU')

    ax_p.axvline(x=0.25, color='red', linestyle=':', linewidth=1.5,
                 alpha=0.7, label='Nyquist limit after stride-2 (0.25)')
    ax_p.axvspan(0.25, 0.5, alpha=0.07, color='red')
    ax_p.text(0.38, 0.85, 'Aliasing\nregion',
              color='red', fontsize=8, ha='center',
              transform=ax_p.get_xaxis_transform(), alpha=0.7)

    ax_p.set_xlabel('Normalized Frequency (cycles/pixel)', fontsize=11)
    ax_p.set_ylabel('Mean FFT Magnitude', fontsize=11)
    ax_p.set_title(
        'Radial Frequency Profile -- Bin-5 Reduces High Frequencies, ReLU Reintroduces in Both',
        fontsize=10, fontweight='bold'
    )
    ax_p.legend(fontsize=9, loc='upper right')
    ax_p.grid(True, alpha=0.3)
    ax_p.set_xlim(0, 0.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'relu_reintroduction.jpg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='jpg')
    print(f"Saved to {out_path}")
    print("Done.")


# ---- Entry point -------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReLU Reintroduction Visualization')
    parser.add_argument('--model_baseline', default=BASELINE_WEIGHTS,
                        help='Path to baseline checkpoint')
    parser.add_argument('--model_bin5', default=BIN5_WEIGHTS,
                        help='Path to Bin-5 checkpoint')
    parser.add_argument('--image', required=True,
                        help='Path to input image (will be resized to 64x64)')
    args = parser.parse_args()
    main(args)