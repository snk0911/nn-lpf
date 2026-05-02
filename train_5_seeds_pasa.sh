#!/bin/bash
GPU=${1:-0}  # default 0, override mit ./script.sh 1
SEEDS=(1 2 3 4 5)

export CUDA_VISIBLE_DEVICES=$GPU

for SEED in "${SEEDS[@]}"; do
    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 12 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 4 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 12 -ba 2 --seed $SEED
done