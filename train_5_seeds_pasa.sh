#!/bin/bash
SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 12 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 4 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 -ba 2 --seed $SEED
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 12 -ba 2 --seed $SEED
done