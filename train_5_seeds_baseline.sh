#!/bin/bash
SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    python main.py --arch resnet18 --aa_type none --seed $SEED
done

wait