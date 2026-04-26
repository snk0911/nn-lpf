#!/bin/bash
SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    python main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type db3 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type sym2 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type sym3 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type sym4 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed $SEED
done