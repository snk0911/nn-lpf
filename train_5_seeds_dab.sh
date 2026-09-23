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
        --aa_type dab \
        --filter_size 3 \
        --seed "$SEED"

    python main.py \
        --arch resnet18 \
        --aa_type dab \
        --filter_size 5 \
        --seed "$SEED"

    python main.py \
        --arch resnet18 \
        --aa_type dab \
        --filter_size 7 \
        --seed "$SEED"

done