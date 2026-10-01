import torch
import torch.nn.functional as F
import numpy as np
import csv

# ---------------------------------------------------------
# Checkpoints for each kernel size
# ---------------------------------------------------------

checkpoint_paths = {
    3: [
        r'saved_checkpoints\k3\checkpoint_seed_1.pth',
        r'saved_checkpoints\k3\checkpoint_seed_2.pth',
        r'saved_checkpoints\k3\checkpoint_seed_3.pth',
        r'saved_checkpoints\k3\checkpoint_seed_4.pth',
        r'saved_checkpoints\k3\checkpoint_seed_5.pth',
    ],

    5: [
        r'saved_checkpoints\k5\checkpoint_seed_1.pth',
        r'saved_checkpoints\k5\checkpoint_seed_2.pth',
        r'saved_checkpoints\k5\checkpoint_seed_3.pth',
        r'saved_checkpoints\k5\checkpoint_seed_4.pth',
        r'saved_checkpoints\k5\checkpoint_seed_5.pth',
    ],

    7: [
        r'saved_checkpoints\k7\checkpoint_seed_1.pth',
        r'saved_checkpoints\k7\checkpoint_seed_2.pth',
        r'saved_checkpoints\k7\checkpoint_seed_3.pth',
        r'saved_checkpoints\k7\checkpoint_seed_4.pth',
        r'saved_checkpoints\k7\checkpoint_seed_5.pth',
    ]
}


rows = []


# ---------------------------------------------------------
# Process each kernel size separately
# ---------------------------------------------------------

for kernel_size, paths in checkpoint_paths.items():

    all_sigmas = []

    print(f"\n{'=' * 60}")
    print(f"Kernel size: {kernel_size}")
    print(f"{'=' * 60}")

    # -----------------------------------------------------
    # Extract sigma values for every seed
    # -----------------------------------------------------

    for seed_idx, path in enumerate(paths, start=1):

        checkpoint = torch.load(path, map_location='cpu')

        state_dict = (
            checkpoint['state_dict']
            if 'state_dict' in checkpoint
            else checkpoint
        )

        deltas = state_dict['dab_controller.deltas']

        # Reconstruct the effective Gaussian standard deviations exactly as in
        # DABSigmaController.get_sigma(). The controller learns unconstrained
        # parameters ("deltas"), which are transformed into positive increments
        # using softplus. Their cumulative sum yields the monotonically increasing
        # sigma values used by the individual DABPool layers.
        sigmas = (
            torch.cumsum(F.softplus(deltas), dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        all_sigmas.append(sigmas)

        # Print individual seed
        sigma_text = "  ".join(
            f"sigma_{j + 1}={sigma:.4f}"
            for j, sigma in enumerate(sigmas)
        )

        print(f"Seed {seed_idx}: {sigma_text}")

        # CSV row
        row = {
            "kernel_size": kernel_size,
            "run": f"Seed {seed_idx}"
        }

        for j, sigma in enumerate(sigmas):
            row[f"sigma_{j + 1}"] = sigma

        rows.append(row)

    # -----------------------------------------------------
    # Statistics over all five seeds
    # -----------------------------------------------------

    all_sigmas = np.stack(all_sigmas)

    mean = all_sigmas.mean(axis=0)

    # Sample standard deviation across five independent random-seed runs.
    # ddof=1 uses Bessel's correction (N-1) to estimate the variance
    # across possible training runs.
    std = all_sigmas.std(axis=0, ddof=1)

    print()

    mean_text = "  ".join(
        f"sigma_{j + 1}={sigma:.4f}"
        for j, sigma in enumerate(mean)
    )

    std_text = "  ".join(
        f"sigma_{j + 1}={sigma:.4f}"
        for j, sigma in enumerate(std)
    )

    print(f"Mean: {mean_text}")
    print(f"Std:  {std_text}")

    # Mean row
    mean_row = {
        "kernel_size": kernel_size,
        "run": "Mean"
    }

    for j, sigma in enumerate(mean):
        mean_row[f"sigma_{j + 1}"] = sigma

    rows.append(mean_row)

    # Std row
    std_row = {
        "kernel_size": kernel_size,
        "run": "Std"
    }

    for j, sigma in enumerate(std):
        std_row[f"sigma_{j + 1}"] = sigma

    rows.append(std_row)


# ---------------------------------------------------------
# Write everything into one CSV
# ---------------------------------------------------------

max_sigmas = max(
    len([key for key in row if key.startswith("sigma_")])
    for row in rows
)

fieldnames = (
    ["kernel_size", "run"]
    + [f"sigma_{i}" for i in range(1, max_sigmas + 1)]
)

output_file = "dabpool_sigmas.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)


print(f"\nSaved results to {output_file}")