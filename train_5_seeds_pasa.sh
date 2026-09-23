#!/bin/bash

set -e

GPU=${1:-0}

SEEDS=(1 2 3 4 5)

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate py312lt

# Select GPU
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Using physical GPU: $GPU"

for SEED in "${SEEDS[@]}"; do

    python main.py \
        --arch resnet18 \
        --aa_type pasa \
        --filter_size 3 \
        --pasa_group 4 \
        -ba 2 \
        --seed "$SEED"

    python main.py \
        --arch resnet18 \
        --aa_type pasa \
        --filter_size 3 \
        --pasa_group 8 \
        -ba 2 \
        --seed "$SEED"

    python main.py \
        --arch resnet18 \
        --aa_type pasa \
        --filter_size 3 \
        --pasa_group 12 \
        -ba 2 \
        --seed "$SEED"

    python main.py \
        --arch resnet18 \
        --aa_type pasa \
        --filter_size 5 \
        --pasa_group 4 \
        -ba 2 \
        --seed "$SEED"

    python main.py \
        --arch resnet18 \
        --aa_type pasa \
        --filter_size 5 \
        --pasa_group 8 \
        -ba 2 \
        --seed "$SEED"

    python main.py \
        --arch resnet18 \
        --aa_type pasa \
        --filter_size 5 \
        --pasa_group 12 \
        -ba 2 \
        --seed "$SEED"

done