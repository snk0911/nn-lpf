#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(pwd)}"

# Activate your conda environment before running this launcher.
screen -dmS eval_accuracy bash "$SCRIPT_DIR/eval_accuracy_gpu0.sh"
screen -dmS eval_shift bash "$SCRIPT_DIR/eval_shift_gpu1.sh"
screen -dmS eval_diagonal bash "$SCRIPT_DIR/eval_diagonal_gpu2.sh"
screen -dmS eval_corruption bash "$SCRIPT_DIR/eval_corruption_gpu3.sh"

screen -ls
