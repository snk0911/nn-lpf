#!/usr/bin/env bash
set -u

SEEDS=(1 2 3 4 5)

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
CONDA_ENV="${CONDA_ENV:-py312lt}"

SCRIPT_PATH="$(readlink -f "$0")"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

RUN_NAME="${RUN_NAME:-tinyimagenet_aa_ablation}"
RUN_DIR="$PROJECT_DIR/training_runs/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
QUEUE_FILE="$RUN_DIR/jobs.txt"
QUEUE_INDEX="$RUN_DIR/next_index"
QUEUE_LOCK="$RUN_DIR/queue.lock"

# ============================================================
# Worker mode
# ============================================================
if [[ "${1:-}" == "worker" ]]; then
    GPU="$2"
    RUN_DIR="$3"
    PROJECT_DIR="$4"
    CONDA_BASE="$5"
    CONDA_ENV="$6"

    LOG_DIR="$RUN_DIR/logs"
    STATUS_DIR="$RUN_DIR/status"
    QUEUE_FILE="$RUN_DIR/jobs.txt"
    QUEUE_INDEX="$RUN_DIR/next_index"
    QUEUE_LOCK="$RUN_DIR/queue.lock"

    cd "$PROJECT_DIR" || exit 1

    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || exit 1

    export CUDA_VISIBLE_DEVICES="$GPU"

    echo "Worker started"
    echo "GPU:    $GPU"
    echo "Conda:  $CONDA_DEFAULT_ENV"
    echo "Python: $(command -v python)"
    echo

    while true; do
        JOB="$(
            (
                flock -x 200

                INDEX="$(cat "$QUEUE_INDEX")"
                TOTAL="$(wc -l < "$QUEUE_FILE")"

                if (( INDEX <= TOTAL )); then
                    sed -n "${INDEX}p" "$QUEUE_FILE"
                    echo $((INDEX + 1)) > "$QUEUE_INDEX"
                fi
            ) 200>"$QUEUE_LOCK"
        )"

        [[ -z "$JOB" ]] && break

        echo "============================================================"
        echo "GPU $GPU claimed job: $JOB"
        echo "============================================================"

        METHOD_FAILED=0

        for SEED in "${SEEDS[@]}"; do
            DONE="$STATUS_DIR/DONE_${JOB}_seed${SEED}"
            FAILED="$STATUS_DIR/FAILED_${JOB}_seed${SEED}"
            LOG="$LOG_DIR/${JOB}_seed${SEED}.log"

            if [[ -f "$DONE" ]]; then
                echo "SKIP: $JOB seed $SEED already completed"
                continue
            fi

            rm -f "$FAILED"

            case "$JOB" in
                baseline)
                    CMD=(python -u main.py --arch resnet18 --aa_type none --seed "$SEED")
                    ;;

                blur_f2)
                    CMD=(python -u main.py --arch resnet18 --aa_type blur --filter_size 2 --seed "$SEED")
                    ;;

                blur_f3)
                    CMD=(python -u main.py --arch resnet18 --aa_type blur --filter_size 3 --seed "$SEED")
                    ;;

                blur_f5)
                    CMD=(python -u main.py --arch resnet18 --aa_type blur --filter_size 5 --seed "$SEED")
                    ;;

                dwt_haar)
                    CMD=(python -u main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed "$SEED")
                    ;;

                dwt_db2)
                    CMD=(python -u main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed "$SEED")
                    ;;

                dwt_db4)
                    CMD=(python -u main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed "$SEED")
                    ;;

                dwt_bior3_3)
                    CMD=(python -u main.py --arch resnet18 --aa_type dwt --wavelet_type bior3.3 --seed "$SEED")
                    ;;

                pasa_f3_g8)
                    CMD=(python -u main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -ba 2 --seed "$SEED")
                    ;;

                pasa_f5_g8)
                    CMD=(python -u main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 -ba 2 --seed "$SEED")
                    ;;

                dab_f3)
                    CMD=(python -u main.py --arch resnet18 --aa_type dab --filter_size 3 --seed "$SEED")
                    ;;

                dab_f5)
                    CMD=(python -u main.py --arch resnet18 --aa_type dab --filter_size 5 --seed "$SEED")
                    ;;

                dab_f7)
                    CMD=(python -u main.py --arch resnet18 --aa_type dab --filter_size 7 --seed "$SEED")
                    ;;

                asap)
                    CMD=(python -u main.py --arch resnet18 --aa_type asap --seed "$SEED")
                    ;;

                *)
                    echo "ERROR: unknown job '$JOB'"
                    exit 1
                    ;;
            esac

            echo
            echo "START: GPU=$GPU | job=$JOB | seed=$SEED"
            printf 'CMD: '
            printf '%q ' "${CMD[@]}"
            printf '\n'

            set -o pipefail
            "${CMD[@]}" 2>&1 | tee "$LOG"
            EXIT_CODE=${PIPESTATUS[0]}
            set +o pipefail

            if (( EXIT_CODE != 0 )); then
                echo "FAILED: $JOB seed $SEED (exit=$EXIT_CODE)"
                touch "$FAILED"
                touch "$STATUS_DIR/FAILED_${JOB}"
                METHOD_FAILED=1
                break
            fi

            touch "$DONE"
            echo "DONE: $JOB seed $SEED"
        done

        if (( METHOD_FAILED == 0 )); then
            rm -f "$STATUS_DIR/FAILED_${JOB}"
            touch "$STATUS_DIR/DONE_${JOB}"
            echo "JOB COMPLETE: $JOB"
        fi

        echo
    done

    echo "GPU $GPU worker finished."
    exit 0
fi

# ============================================================
# Launcher mode
# ============================================================
mkdir -p "$LOG_DIR" "$STATUS_DIR"

for REQUIRED in screen flock conda; do
    if ! command -v "$REQUIRED" >/dev/null 2>&1; then
        echo "ERROR: required command '$REQUIRED' not found"
        exit 1
    fi
done

CONDA_BASE="$(conda info --base)"

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda.sh not found:"
    echo "  $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    echo "ERROR: conda environment '$CONDA_ENV' not found"
    exit 1
fi

cat > "$QUEUE_FILE" <<'JOBS'
baseline
blur_f2
blur_f3
blur_f5
dwt_haar
dwt_db2
dwt_db4
dwt_bior3_3
pasa_f3_g8
pasa_f5_g8
dab_f3
dab_f5
dab_f7
asap
JOBS

echo 1 > "$QUEUE_INDEX"

for SESSION in thesis_gpu0 thesis_gpu1; do
    if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
        echo "ERROR: screen session '$SESSION' already exists."
        screen -ls
        exit 1
    fi
done

echo "Starting Tiny ImageNet AA experiments"
echo
echo "Project: $PROJECT_DIR"
echo "Conda:   $CONDA_ENV"
echo "GPUs:    $GPU0, $GPU1"
echo "Seeds:   ${SEEDS[*]}"
echo "Configs: 14"
echo "Runs:    70"
echo

screen -dmS thesis_gpu0 \
    bash "$SCRIPT_PATH" worker "$GPU0" "$RUN_DIR" "$PROJECT_DIR" "$CONDA_BASE" "$CONDA_ENV"

screen -dmS thesis_gpu1 \
    bash "$SCRIPT_PATH" worker "$GPU1" "$RUN_DIR" "$PROJECT_DIR" "$CONDA_BASE" "$CONDA_ENV"

sleep 1
screen -ls

echo
echo "Attach workers:"
echo "  screen -r thesis_gpu0"
echo "  screen -r thesis_gpu1"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Status:"
echo "  $STATUS_DIR"
