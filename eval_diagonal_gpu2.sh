#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/thesis_repo/nn_lpf}"
PYTHON="${PYTHON:-python}"
cd "$REPO"
mkdir -p results/eval_logs
mkdir -p results/diagonal_npy
exec > >(tee "results/eval_logs/diagonal_gpu2.log") 2>&1

echo "=== DIAGONAL | GPU 2 ==="

echo "===== baseline ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type none --seed 1 --weights best_checkpoints/resnet18_none_seed1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/baseline_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type none --seed 2 --weights best_checkpoints/resnet18_none_seed2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/baseline_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type none --seed 3 --weights best_checkpoints/resnet18_none_seed3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/baseline_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type none --seed 4 --weights best_checkpoints/resnet18_none_seed4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/baseline_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type none --seed 5 --weights best_checkpoints/resnet18_none_seed5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/baseline_seed5_${f}"; fi
done

echo "===== blur_f2 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 2 --seed 1 --weights best_checkpoints/resnet18_blur_seed1_filter2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f2_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 2 --seed 2 --weights best_checkpoints/resnet18_blur_seed2_filter2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f2_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 2 --seed 3 --weights best_checkpoints/resnet18_blur_seed3_filter2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f2_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 2 --seed 4 --weights best_checkpoints/resnet18_blur_seed4_filter2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f2_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 2 --seed 5 --weights best_checkpoints/resnet18_blur_seed5_filter2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f2_seed5_${f}"; fi
done

echo "===== blur_f3 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 3 --seed 1 --weights best_checkpoints/resnet18_blur_seed1_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f3_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 3 --seed 2 --weights best_checkpoints/resnet18_blur_seed2_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f3_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 3 --seed 3 --weights best_checkpoints/resnet18_blur_seed3_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f3_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 3 --seed 4 --weights best_checkpoints/resnet18_blur_seed4_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f3_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 3 --seed 5 --weights best_checkpoints/resnet18_blur_seed5_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f3_seed5_${f}"; fi
done

echo "===== blur_f5 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 5 --seed 1 --weights best_checkpoints/resnet18_blur_seed1_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f5_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 5 --seed 2 --weights best_checkpoints/resnet18_blur_seed2_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f5_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 5 --seed 3 --weights best_checkpoints/resnet18_blur_seed3_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f5_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 5 --seed 4 --weights best_checkpoints/resnet18_blur_seed4_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f5_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type blur --filter_size 5 --seed 5 --weights best_checkpoints/resnet18_blur_seed5_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/blur_f5_seed5_${f}"; fi
done

echo "===== dab_f3 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 3 --seed 1 --weights best_checkpoints/resnet18_dab_seed1_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f3_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 3 --seed 2 --weights best_checkpoints/resnet18_dab_seed2_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f3_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 3 --seed 3 --weights best_checkpoints/resnet18_dab_seed3_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f3_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 3 --seed 4 --weights best_checkpoints/resnet18_dab_seed4_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f3_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 3 --seed 5 --weights best_checkpoints/resnet18_dab_seed5_filter3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f3_seed5_${f}"; fi
done

echo "===== dab_f5 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 5 --seed 1 --weights best_checkpoints/resnet18_dab_seed1_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f5_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 5 --seed 2 --weights best_checkpoints/resnet18_dab_seed2_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f5_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 5 --seed 3 --weights best_checkpoints/resnet18_dab_seed3_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f5_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 5 --seed 4 --weights best_checkpoints/resnet18_dab_seed4_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f5_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 5 --seed 5 --weights best_checkpoints/resnet18_dab_seed5_filter5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f5_seed5_${f}"; fi
done

echo "===== dab_f7 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 7 --seed 1 --weights best_checkpoints/resnet18_dab_seed1_filter7_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f7_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 7 --seed 2 --weights best_checkpoints/resnet18_dab_seed2_filter7_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f7_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 7 --seed 3 --weights best_checkpoints/resnet18_dab_seed3_filter7_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f7_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 7 --seed 4 --weights best_checkpoints/resnet18_dab_seed4_filter7_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f7_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dab --filter_size 7 --seed 5 --weights best_checkpoints/resnet18_dab_seed5_filter7_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dab_f7_seed5_${f}"; fi
done

echo "===== dwt_haar ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed 1 --weights best_checkpoints/resnet18_dwt_seed1_haar_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_haar_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed 2 --weights best_checkpoints/resnet18_dwt_seed2_haar_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_haar_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed 3 --weights best_checkpoints/resnet18_dwt_seed3_haar_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_haar_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed 4 --weights best_checkpoints/resnet18_dwt_seed4_haar_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_haar_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed 5 --weights best_checkpoints/resnet18_dwt_seed5_haar_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_haar_seed5_${f}"; fi
done

echo "===== dwt_db2 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed 1 --weights best_checkpoints/resnet18_dwt_seed1_db2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db2_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed 2 --weights best_checkpoints/resnet18_dwt_seed2_db2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db2_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed 3 --weights best_checkpoints/resnet18_dwt_seed3_db2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db2_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed 4 --weights best_checkpoints/resnet18_dwt_seed4_db2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db2_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed 5 --weights best_checkpoints/resnet18_dwt_seed5_db2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db2_seed5_${f}"; fi
done

echo "===== dwt_db4 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed 1 --weights best_checkpoints/resnet18_dwt_seed1_db4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db4_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed 2 --weights best_checkpoints/resnet18_dwt_seed2_db4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db4_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed 3 --weights best_checkpoints/resnet18_dwt_seed3_db4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db4_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed 4 --weights best_checkpoints/resnet18_dwt_seed4_db4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db4_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed 5 --weights best_checkpoints/resnet18_dwt_seed5_db4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_db4_seed5_${f}"; fi
done

echo "===== dwt_bior3_3 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed 1 --weights best_checkpoints/resnet18_dwt_seed1_bior3.3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_bior3_3_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed 2 --weights best_checkpoints/resnet18_dwt_seed2_bior3.3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_bior3_3_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed 3 --weights best_checkpoints/resnet18_dwt_seed3_bior3.3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_bior3_3_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed 4 --weights best_checkpoints/resnet18_dwt_seed4_bior3.3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_bior3_3_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed 5 --weights best_checkpoints/resnet18_dwt_seed5_bior3.3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/dwt_bior3_3_seed5_${f}"; fi
done

echo "===== pasa_f3_g1 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 1 --seed 1 --weights best_checkpoints/resnet18_pasa_seed1_filter3_group1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g1_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 1 --seed 2 --weights best_checkpoints/resnet18_pasa_seed2_filter3_group1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g1_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 1 --seed 3 --weights best_checkpoints/resnet18_pasa_seed3_filter3_group1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g1_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 1 --seed 4 --weights best_checkpoints/resnet18_pasa_seed4_filter3_group1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g1_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 1 --seed 5 --weights best_checkpoints/resnet18_pasa_seed5_filter3_group1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g1_seed5_${f}"; fi
done

echo "===== pasa_f3_g2 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 2 --seed 1 --weights best_checkpoints/resnet18_pasa_seed1_filter3_group2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g2_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 2 --seed 2 --weights best_checkpoints/resnet18_pasa_seed2_filter3_group2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g2_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 2 --seed 3 --weights best_checkpoints/resnet18_pasa_seed3_filter3_group2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g2_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 2 --seed 4 --weights best_checkpoints/resnet18_pasa_seed4_filter3_group2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g2_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 2 --seed 5 --weights best_checkpoints/resnet18_pasa_seed5_filter3_group2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g2_seed5_${f}"; fi
done

echo "===== pasa_f3_g4 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 --seed 1 --weights best_checkpoints/resnet18_pasa_seed1_filter3_group4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g4_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 --seed 2 --weights best_checkpoints/resnet18_pasa_seed2_filter3_group4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g4_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 --seed 3 --weights best_checkpoints/resnet18_pasa_seed3_filter3_group4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g4_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 --seed 4 --weights best_checkpoints/resnet18_pasa_seed4_filter3_group4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g4_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 --seed 5 --weights best_checkpoints/resnet18_pasa_seed5_filter3_group4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g4_seed5_${f}"; fi
done

echo "===== pasa_f3_g8 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 --seed 1 --weights best_checkpoints/resnet18_pasa_seed1_filter3_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g8_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 --seed 2 --weights best_checkpoints/resnet18_pasa_seed2_filter3_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g8_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 --seed 3 --weights best_checkpoints/resnet18_pasa_seed3_filter3_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g8_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 --seed 4 --weights best_checkpoints/resnet18_pasa_seed4_filter3_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g8_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 --seed 5 --weights best_checkpoints/resnet18_pasa_seed5_filter3_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g8_seed5_${f}"; fi
done

echo "===== pasa_f3_g16 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 16 --seed 1 --weights best_checkpoints/resnet18_pasa_seed1_filter3_group16_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g16_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 16 --seed 2 --weights best_checkpoints/resnet18_pasa_seed2_filter3_group16_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g16_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 16 --seed 3 --weights best_checkpoints/resnet18_pasa_seed3_filter3_group16_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g16_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 16 --seed 4 --weights best_checkpoints/resnet18_pasa_seed4_filter3_group16_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g16_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 16 --seed 5 --weights best_checkpoints/resnet18_pasa_seed5_filter3_group16_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f3_g16_seed5_${f}"; fi
done

echo "===== pasa_f5_g8 ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 --seed 1 --weights best_checkpoints/resnet18_pasa_seed1_filter5_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f5_g8_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 --seed 2 --weights best_checkpoints/resnet18_pasa_seed2_filter5_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f5_g8_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 --seed 3 --weights best_checkpoints/resnet18_pasa_seed3_filter5_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f5_g8_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 --seed 4 --weights best_checkpoints/resnet18_pasa_seed4_filter5_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f5_g8_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 --seed 5 --weights best_checkpoints/resnet18_pasa_seed5_filter5_group8_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/pasa_f5_g8_seed5_${f}"; fi
done

echo "===== asap_sp ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_sp --seed 1 --weights best_checkpoints/resnet18_asap_sp_seed1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_sp_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_sp --seed 2 --weights best_checkpoints/resnet18_asap_sp_seed2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_sp_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_sp --seed 3 --weights best_checkpoints/resnet18_asap_sp_seed3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_sp_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_sp --seed 4 --weights best_checkpoints/resnet18_asap_sp_seed4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_sp_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_sp --seed 5 --weights best_checkpoints/resnet18_asap_sp_seed5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_sp_seed5_${f}"; fi
done

echo "===== asap_stbl ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_stbl --seed 1 --weights best_checkpoints/resnet18_asap_stbl_seed1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_stbl_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_stbl --seed 2 --weights best_checkpoints/resnet18_asap_stbl_seed2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_stbl_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_stbl --seed 3 --weights best_checkpoints/resnet18_asap_stbl_seed3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_stbl_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_stbl --seed 4 --weights best_checkpoints/resnet18_asap_stbl_seed4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_stbl_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_stbl --seed 5 --weights best_checkpoints/resnet18_asap_stbl_seed5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_stbl_seed5_${f}"; fi
done

echo "===== asap_lp ====="
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_lp --seed 1 --weights best_checkpoints/resnet18_asap_lp_seed1_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_lp_seed1_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_lp --seed 2 --weights best_checkpoints/resnet18_asap_lp_seed2_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_lp_seed2_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_lp --seed 3 --weights best_checkpoints/resnet18_asap_lp_seed3_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_lp_seed3_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_lp --seed 4 --weights best_checkpoints/resnet18_asap_lp_seed4_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_lp_seed4_${f}"; fi
done
CUDA_VISIBLE_DEVICES=2 "$PYTHON" main.py --arch resnet18 --aa_type asap_lp --seed 5 --weights best_checkpoints/resnet18_asap_lp_seed5_best.pth.tar --evaluate_diagonal
for f in diag_corrs.npy diag_probs.npy diag_probs2.npy diag_preds.npy; do
    if [[ -f "$f" ]]; then mv -f "$f" "results/diagonal_npy/asap_lp_seed5_${f}"; fi
done

echo "=== DIAGONAL FINISHED: 100 runs ==="
