#!/bin/bash
GPU=${1:-0}  # default 0, override mit ./script.sh 1
SEEDS=(1 2 3 4 5)

export CUDA_VISIBLE_DEVICES=$GPU

for SEED in "${SEEDS[@]}"; do
    python main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed $SEED
    python main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed $SEED
done