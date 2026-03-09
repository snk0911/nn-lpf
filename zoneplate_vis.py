"""
Zone Plate Visualization — Frequency Response of all Anti-Aliasing Methods
Passes zone plate directly through the isolated pool layer (no conv1/bn1)
to show pure filter frequency response without learned feature interference.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import aa_models


# CONFIG 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT     = os.path.join(BASE_DIR, 'saved_checkpoints')

BASELINE_WEIGHTS     = os.path.join(CKPT, 'baseline.pth.tar')
AVG_WEIGHTS          = os.path.join(CKPT, 'avg.pth.tar')
TRI3_WEIGHTS         = os.path.join(CKPT, 'tri3.pth.tar')
BLUR4_WEIGHTS        = os.path.join(CKPT, 'blur4.pth.tar')
BIN5_WEIGHTS         = os.path.join(CKPT, 'bin5.pth.tar')
PASA_WEIGHTS         = os.path.join(CKPT, 'pasa.pth.tar')
WAVELET_HAAR_WEIGHTS = os.path.join(CKPT, 'wavelet_haar.pth.tar')
WAVELET_DB4_WEIGHTS  = os.path.join(CKPT, 'wavelet_db4.pth.tar')
WAVELET_BIOR_WEIGHTS = os.path.join(CKPT, 'wavelet_bior.pthv')

GPU                  = None   # set to 0 if you want to use GPU
OUT_DIR              = os.path.join(BASE_DIR, 'plots')
ZONE_PLATE_SIZE      = 128    # 512 for presentation, 256 for thesis PDF# 


# Generate Zone Plate
def generate_zone_plate(size=512):
    """
    Generates a zone plate of given size — contains all spatial frequencies
    from low (center) to high (edges).
    f_max is scaled relative to size so the pattern looks correct at any resolution.
    Returns:
        plate_np:     numpy (size, size) normalized to [0,1] for display
        plate_tensor: torch (1, 64, size, size) — 64 channels to match pool input
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2

    # f_max scaled so that the highest frequency ring is near the Nyquist limit
    # f_max is set to size / (4 * stride) so that the highest frequency at the zone plate boundary corresponds to the Nyquist frequency after downsampling with stride 2.
    stride = 2 # downsampling factor, simulates stride 2
    f_max = size / (4 * stride)
    plate = np.cos(np.pi * f_max * r2)

    plate_np = (plate - plate.min()) / (plate.max() - plate.min())

    # Pool layer expects (B, C, H, W) with C=64 (output of conv1 in ResNet-18)
    plate_tensor = torch.from_numpy(plate).float()
    plate_tensor = plate_tensor.unsqueeze(0).repeat(64, 1, 1)  # (64, H, W)
    plate_tensor = plate_tensor.unsqueeze(0)                    # (1, 64, H, W)

    return plate_np, plate_tensor


def load_aa_model(weights_path, aa_type, filter_size=5, wavelet_type='haar',
                  pasa_group=2, gpu=None):
    model = aa_models.resnet18(
        aa_type=aa_type,
        wavelet_type=wavelet_type,
        filter_size=filter_size,
        pasa_group=pasa_group,
        num_classes=200
    )
    weights = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights['state_dict'])
    model.eval()
    if gpu is not None:
        model = model.cuda(gpu)
    return model


# Run zone plate through isolated pool layer
def run_through_pool(model, zone_plate_tensor, gpu=None):
    """
    Extracts pool layer from model and runs zone plate directly through it.
    Returns: numpy (H, W) normalized to [0,1]
    """
    pool = model.pool
    pool.eval()

    x = zone_plate_tensor
    if gpu is not None:
        x = x.cuda(gpu)

    with torch.no_grad():
        out = pool(x)  # (1, 64, H/2, W/2)

    out = out[0].cpu().numpy()          # (64, H, W)
    out = np.mean(out, axis=0)          # average over channels
    out = out - out.min()
    out = out / (out.max() + 1e-8)
    return out


def main():
    # Zone plate — size controlled by ZONE_PLATE_SIZE in config
    # 512 for presentation, 256 for thesis PDF
    zone_plate_np, zone_plate_tensor = generate_zone_plate(size=ZONE_PLATE_SIZE)

    if GPU is not None:
        zone_plate_tensor = zone_plate_tensor.cuda(GPU)

    # Load all models
    print("Loading models...")
    model_configs = [
        ('Baseline', load_aa_model(BASELINE_WEIGHTS, aa_type='none')),
        ('Blur-4', load_aa_model(BLUR4_WEIGHTS, aa_type='blur', filter_size=4)),
        # ('Rect-2',        load_aa_model(AVG_WEIGHTS,          aa_type='avg',
        #                                 filter_size=2,
        #                                 gpu=GPU)),
        # ('Tri-3',         load_aa_model(TRI3_WEIGHTS,         aa_type='blur',
        #                                 filter_size=3, wavelet_type='haar',
        #                                 gpu=GPU)),
        # ('Bin-5',         load_aa_model(BIN5_WEIGHTS,         aa_type='blur',
        #                                 filter_size=5, wavelet_type='haar',
        #                                 gpu=GPU)),
        # ('PASA',          load_aa_model(PASA_WEIGHTS,         aa_type='pasa',
        #                                 filter_size=5, wavelet_type='haar',
        #                                 pasa_group=8, gpu=GPU)),
        # ('WaveCNet-Haar', load_aa_model(WAVELET_HAAR_WEIGHTS, aa_type='wavelet',
        #                                 filter_size=5, wavelet_type='haar',
        #                                 gpu=GPU)),
        # ('WaveCNet-DB4',  load_aa_model(WAVELET_DB4_WEIGHTS,  aa_type='wavelet',
        #                                 filter_size=5, wavelet_type='db4',
        #                                 gpu=GPU)),
        # ('WaveCNet-Bior', load_aa_model(WAVELET_BIOR_WEIGHTS, aa_type='wavelet',
        #                                 filter_size=5, wavelet_type='bior',
        #                                 gpu=GPU)),
    ]

    # Run zone plate through isolated pool layer of each model
    print("Running zone plate through isolated pool layers...")
    outputs = {}
    for name, model in model_configs:
        out = run_through_pool(model, zone_plate_tensor, GPU)
        outputs[name] = out
        print(f"  {name}: output shape {out.shape}")

    # Plot
    n_models = len(model_configs)
    fig, axes = plt.subplots(1, n_models + 1,
                             figsize=(3.5 * (n_models + 1), 4))
    fig.suptitle('Zone Plate: Frequency Response of Isolated Pool Layer',
                 fontsize=14, fontweight='bold', y=1.02)

    # Input zone plate
    axes[0].imshow(zone_plate_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Input\nZone Plate\n({ZONE_PLATE_SIZE}×{ZONE_PLATE_SIZE})', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for col, (name, _) in enumerate(model_configs):
        axes[col + 1].imshow(outputs[name], cmap='gray', vmin=0, vmax=1)
        axes[col + 1].set_title(f'{name}\nPool output', fontsize=10, fontweight='bold')
        axes[col + 1].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'zone_plate.jpg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='jpg')
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()