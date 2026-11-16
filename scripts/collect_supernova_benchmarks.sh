#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

IMG_WIDTH=""
IMG_HEIGHT=""
OUT_DIR=""
MODES=(cpu gpu)

usage() {
    echo "Usage: $0 --width PIXELS --height PIXELS [--out-dir DIR] [--cpu-only]" >&2
}

while (($# > 0)); do
    case "$1" in
        --width)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            IMG_WIDTH="$2"
            shift 2
            ;;
        --height)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            IMG_HEIGHT="$2"
            shift 2
            ;;
        --out-dir)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            OUT_DIR="$(realpath -m -- "$2")"
            shift 2
            ;;
        --cpu-only)
            MODES=(cpu)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! "$IMG_WIDTH" =~ ^[1-9][0-9]*$ || ! "$IMG_HEIGHT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Width and height must be positive integers." >&2
    usage
    exit 2
fi

if ((IMG_WIDTH % 2 != 0 || IMG_HEIGHT % 2 != 0)); then
    echo "Width and height must be even." >&2
    exit 2
fi

readonly IMG_WIDTH
readonly IMG_HEIGHT
readonly IMAGE_SIZE="${IMG_WIDTH}x${IMG_HEIGHT}"
# A downloaded archive has no git history; label its results "unknown".
readonly GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
readonly GIT_COMMIT_SHORT="${GIT_COMMIT:0:8}"

readonly REPETITIONS="${REPETITIONS:-5}"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

# Created from the host. Must sit under the repository, which the container
# mounts at /workspace.
readonly HOST_OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmarks/supernova-$RUN_ID}"
case "$HOST_OUT_DIR/" in
    "$REPO_ROOT"/*) ;;
    *) echo "--out-dir must be inside $REPO_ROOT" >&2; exit 2 ;;
esac
readonly HOST_CPU_OUT="$HOST_OUT_DIR/${IMAGE_SIZE}-cpu-${GIT_COMMIT_SHORT}.csv"
readonly HOST_GPU_OUT="$HOST_OUT_DIR/${IMAGE_SIZE}-gpu-${GIT_COMMIT_SHORT}.csv"

# Passed to Rust inside the container. The repository is mounted at /workspace.
readonly CONTAINER_OUT_DIR="/workspace/${HOST_OUT_DIR#"$REPO_ROOT"/}"
readonly CPU_OUT="$CONTAINER_OUT_DIR/${IMAGE_SIZE}-cpu-${GIT_COMMIT_SHORT}.csv"
readonly GPU_OUT="$CONTAINER_OUT_DIR/${IMAGE_SIZE}-gpu-${GIT_COMMIT_SHORT}.csv"

readonly LOG_OUT="$HOST_OUT_DIR/collection.log"
readonly FAILURES_OUT="$HOST_OUT_DIR/failures.log"
# SWEEP_B="1 4 10" trims the sweep, e.g. for 4K where every run takes hours.
read -ra REQUESTED_BATCH_SIZES <<< "${SWEEP_B:-1 2 4 6 8 10 12 15 18}"
readonly REQUESTED_BATCH_SIZES
BATCH_SIZES=()

for batch_size in "${REQUESTED_BATCH_SIZES[@]}"; do
    if ((IMG_HEIGHT % (2 * batch_size) == 0)); then
        BATCH_SIZES+=("$batch_size")
    else
        echo "Skipping B=$batch_size: incompatible with height $IMG_HEIGHT"
    fi
done

readonly BATCH_SIZES

sort_csv_by_batch_size() {
    local path="$1"
    local temporary

    if [[ ! -s "$path" ]]; then
        echo "Cannot sort missing or empty benchmark CSV: $path" >&2
        return 1
    fi

    temporary="$(mktemp "${path}.tmp.XXXXXX")"
    if ! {
        IFS= read -r header
        printf '%s\n' "$header"
        LC_ALL=C sort -s -t, -k5,5n
    } < "$path" > "$temporary"; then
        rm -f "$temporary"
        return 1
    fi

    # -f: the container writes the CSV as root, so plain mv would prompt.
    mv -f "$temporary" "$path"
}

mkdir -p "$HOST_OUT_DIR"
exec > >(tee -a "$LOG_OUT") 2>&1

echo "Campaign: $RUN_ID"
echo "Commit: $GIT_COMMIT"
echo "Started: $(date -u --iso-8601=seconds)"

"$REPO_ROOT/scripts/dev.sh" pipeline \
    --width "$IMG_WIDTH" \
    --height "$IMG_HEIGHT"

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  echo "=== Repetition $repetition/$REPETITIONS ==="

  mapfile -t batch_order < <(
      printf '%s\n' "${BATCH_SIZES[@]}" | shuf
  )

  for batch_size in "${batch_order[@]}"; do
      # Randomize whether CPU or GPU runs first for this pair.
      if ((${#MODES[@]} == 1)); then
          modes=("${MODES[@]}")
      elif ((RANDOM % 2)); then
          modes=(cpu gpu)
      else
          modes=(gpu cpu)
      fi

      for mode in "${modes[@]}"; do
          echo
          echo "Starting: mode=$mode B=$batch_size repetition=$repetition"
          echo "Time: $(date -u --iso-8601=seconds)"
          echo "Host load: $(cat /proc/loadavg)"

          # A failed run (OOM, BENCH_TIMEOUT) adds no CSV row; keep going so
          # the other batch sizes still get measured.
          if [[ "$mode" == cpu ]]; then
              run=("$REPO_ROOT/scripts/dev.sh" benchmark --benchmark-output "$CPU_OUT")
          else
              run=("$REPO_ROOT/scripts/dev.sh" --gpu benchmark --benchmark-output "$GPU_OUT")
          fi
          if ! "${run[@]}" --batch-size "$batch_size"; then
              echo "FAILED: mode=$mode B=$batch_size repetition=$repetition" | tee -a "$FAILURES_OUT"
          fi
      done
  done
done

echo
echo "Sorting CSVs by batch size"
for path in "$HOST_CPU_OUT" "$HOST_GPU_OUT"; do
    if [[ -s "$path" ]]; then
        sort_csv_by_batch_size "$path"
    fi
done

echo
echo "Finished: $(date -u --iso-8601=seconds)"
echo "Results on host: $HOST_OUT_DIR"
