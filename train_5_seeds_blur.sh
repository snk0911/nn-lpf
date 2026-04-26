#!/bin/bash
SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    # python main.py --arch resnet18 --aa_type blur --filter_size 2 --seed $SEED
    python main.py --arch resnet18 --aa_type blur --filter_size 3 --seed $SEED
    # python main.py --arch resnet18 --aa_type blur --filter_size 4 --seed $SEED
    python main.py --arch resnet18 --aa_type blur --filter_size 5 --seed $SEED
    # python main.py --arch resnet18 --aa_type blur --filter_size 6 --seed $SEED
    python main.py --arch resnet18 --aa_type blur --filter_size 7 --seed $SEED
done