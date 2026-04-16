#!/bin/bash
SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    python main.py --arch resnet18 --aa_type none --seed $SEED &
    python main.py --arch resnet18 --aa_type blur --filter_size 2 --seed $SEED &
    python main.py --arch resnet18 --aa_type blur --filter_size 3 --seed $SEED &
    python main.py --arch resnet18 --aa_type blur --filter_size 4 --seed $SEED &
    python main.py --arch resnet18 --aa_type blur --filter_size 5 --seed $SEED &
    python main.py --arch resnet18 --aa_type blur --filter_size 6 --seed $SEED &
    python main.py --arch resnet18 --aa_type blur --filter_size 7 --seed $SEED &

    python main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed $SEED &
    python main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed $SEED &
    python main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed $SEED &

    python main.py --arch resnet18 --aa_type dab --filter_size 3 --seed $SEED &
    python main.py --arch resnet18 --aa_type dab --filter_size 5 --seed $SEED &
    python main.py --arch resnet18 --aa_type dab --filter_size 7 --seed $SEED &

    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 -ba 2 --seed $SEED &
    python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -ba 2 --seed $SEED &
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 -ba 2 --seed $SEED &
    python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 4 -ba 2 --seed $SEED &
done

wait