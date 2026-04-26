"""
Zone Plate Visualization — Frequency Response of all Anti-Aliasing Methods
Passes zone plate directly through the isolated pool layer (no conv1/bn1)
to show pure filter frequency response without learned feature interference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import aa_models


# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT     = os.path.join(BASE_DIR, '/home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_none_seed1')

BASELINE_WEIGHTS     = os.path.join(CKPT, 'resnet18_none_seed1_best.pth.tar')
# BLUR5_WEIGHTS        = os.path.join(CKPT, 'blur5.pth.tar')
# AVG_WEIGHTS          = os.path.join(CKPT, 'avg.pth.tar')
# TRI3_WEIGHTS         = os.path.join(CKPT, 'tri3.pth.tar')
# BIN5_WEIGHTS         = os.path.join(CKPT, 'bin5.pth.tar')
# PASA_WEIGHTS         = os.path.join(CKPT, 'pasa.pth.tar')
# WAVELET_HAAR_WEIGHTS = os.path.join(CKPT, 'wavelet_haar.pth.tar')
# WAVELET_DB4_WEIGHTS  = os.path.join(CKPT, 'wavelet_db4.pth.tar')
# WAVELET_BIOR_WEIGHTS = os.path.join(CKPT, 'wavelet_bior.pth.tar')

GPU             = None  # set to 0 if you want to use GPU
OUT_DIR         = os.path.join(BASE_DIR, 'plots')
ZONE_PLATE_SIZE = 256   # 512 for presentation, 256 for thesis PDF


def generate_zone_plate(size=256):
    """
        Generates a zone plate following Gonzalez & Woods (2018, p. 190):

            z(x, y) = 0.5 * (1 + cos(x^2 + y^2))

        Coordinates are centered at the image origin, ranging from
        -sqrt(32*pi) to +sqrt(32*pi), such that the zone plate contains
        16 cycles from center to edge along each axis. This ensures
        frequencies exceed the Nyquist limit of the stride-2 downsampled
        output, making aliasing artifacts visible in the baseline.

        Returns:
            plate:        numpy (size, size) in [0, 1] for display
            plate_tensor: torch (1, 64, size, size) — 64 channels to match pool input
    """
    r_max = np.sqrt(32 * np.pi)
    x = np.linspace(-r_max, r_max, size)
    y = np.linspace(-r_max, r_max, size)
    xx, yy = np.meshgrid(x, y)

    plate = 0.5 * (1 + np.cos(xx**2 + yy**2))  # Gonzalez & Woods (2018, p. 190)

    plate_tensor = torch.from_numpy(plate).float()
    plate_tensor = plate_tensor.unsqueeze(0).repeat(64, 1, 1)  # (64, H, W)
    plate_tensor = plate_tensor.unsqueeze(0)                    # (1, 64, H, W)

    return plate, plate_tensor


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


def run_through_pool(model, zone_plate_tensor, gpu=None):
    """
    Extracts pool layer from model and runs zone plate directly through it.
    Returns: numpy (H, W) normalized to [0, 1]
    """
    pool = model.pool
    pool.eval()

    x = zone_plate_tensor
    if gpu is not None:
        x = x.cuda(gpu)

    with torch.no_grad():
        out = pool(x)  # (1, 64, H/2, W/2)

    out = out[0].cpu().numpy()   # (64, H, W)
    out = np.mean(out, axis=0)   # average over channels
    out = out - out.min()
    out = out / (out.max() + 1e-8)
    return out


def main():
    zone_plate, zone_plate_tensor = generate_zone_plate(size=ZONE_PLATE_SIZE)

    if GPU is not None:
        zone_plate_tensor = zone_plate_tensor.cuda(GPU)

    print("Loading models...")
    model_configs = [
        ('Baseline', load_aa_model(BASELINE_WEIGHTS, aa_type='none')),
        # ('Blur-5',   load_aa_model(BLUR5_WEIGHTS,    aa_type='blur', filter_size=5)),
        # ('Rect-2',        load_aa_model(AVG_WEIGHTS,          aa_type='avg',     filter_size=2)),
        # ('Tri-3',         load_aa_model(TRI3_WEIGHTS,         aa_type='blur',    filter_size=3)),
        # ('Bin-5',         load_aa_model(BIN5_WEIGHTS,         aa_type='blur',    filter_size=5)),
        # ('PASA',          load_aa_model(PASA_WEIGHTS,         aa_type='pasa',    filter_size=5, pasa_group=8)),
        # ('WaveCNet-Haar', load_aa_model(WAVELET_HAAR_WEIGHTS, aa_type='wavelet', wavelet_type='haar')),
        # ('WaveCNet-DB4',  load_aa_model(WAVELET_DB4_WEIGHTS,  aa_type='wavelet', wavelet_type='db4')),
        # ('WaveCNet-Bior', load_aa_model(WAVELET_BIOR_WEIGHTS, aa_type='wavelet', wavelet_type='bior')),
    ]

    print("Running zone plate through isolated pool layers...")
    outputs = {}
    for name, model in model_configs:
        out = run_through_pool(model, zone_plate_tensor, GPU)
        outputs[name] = out
        print(f"  {name}: output shape {out.shape}")

    n_models = len(model_configs)
    fig, axes = plt.subplots(1, n_models + 1,
                             figsize=(3.5 * (n_models + 1), 4))
    fig.suptitle('Zone Plate: Frequency Response of Isolated Pool Layer',
                 fontsize=14, fontweight='bold', y=1.02)

    axes[0].imshow(zone_plate, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Input\nZone Plate\n({ZONE_PLATE_SIZE}×{ZONE_PLATE_SIZE})',
                      fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for col, (name, _) in enumerate(model_configs):
        axes[col + 1].imshow(outputs[name], cmap='gray', vmin=0, vmax=1)
        axes[col + 1].set_title(f'{name}\nPool output', fontsize=10, fontweight='bold')
        axes[col + 1].axis('off')

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'zone_plate.jpg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='jpg')
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()