import torch
import torch.nn.functional as F
import numpy as np

checkpoint_paths = [
    'checkpoint_seed_1.pth',
    'checkpoint_seed_2.pth',
    'checkpoint_seed_3.pth',
    'checkpoint_seed_4.pth',
    'checkpoint_seed_5.pth',
]

all_sigmas = []
lines = []

# Per-seed sigmas
for i, path in enumerate(checkpoint_paths):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    deltas = state_dict['dab_controller.deltas']
    sigmas = torch.cumsum(F.softplus(deltas), dim=0).detach().numpy()
    all_sigmas.append(sigmas)
    lines.append(f"Seed {i+1}: sigma_1={sigmas[0]:.4f}  sigma_2={sigmas[1]:.4f}  sigma_3={sigmas[2]:.4f}  sigma_4={sigmas[3]:.4f}")

# Mean and std over 5 seeds
all_sigmas = np.array(all_sigmas)
mean = all_sigmas.mean(axis=0)
std = all_sigmas.std(axis=0)

lines.append("")
lines.append(f"Mean:  sigma_1={mean[0]:.4f}  sigma_2={mean[1]:.4f}  sigma_3={mean[2]:.4f}  sigma_4={mean[3]:.4f}")
lines.append(f"Std:   sigma_1={std[0]:.4f}  sigma_2={std[1]:.4f}  sigma_3={std[2]:.4f}  sigma_4={std[3]:.4f}")

# Write to file
with open('sigmas_7.txt', 'w') as f:
    f.write('\n'.join(lines))

print('\n'.join(lines))
print("\nSaved to sigmas.txt")