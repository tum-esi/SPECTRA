#!/usr/bin/env bash
# Batch-size sweep (collect_supernova_benchmarks.sh) over every frame size of
# the paper's size table, on one machine. One CSV per size and device lands in
# benchmarks/supernova/<MACHINE>/sweep/; sizes_table.py turns them into rows.
#
#   scripts/sweep_sizes.sh ServerA [--cpu-only] [WxH ...]
#
# Without sizes: the automotive squares 16^2..224^2 first (quick), then 480p,
# 720p, 1080p, 2K (DCI 2048x1080) and 4K. Env REPETITIONS and SWEEP_B pass
# through to collect_supernova_benchmarks.sh.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MACHINE="${1:?usage: $0 MACHINE [--cpu-only] [WxH ...]}"
shift
flags=()
if [[ "${1:-}" == --cpu-only ]]; then
    flags=(--cpu-only)
    shift
fi

sizes=("$@")
if ((${#sizes[@]} == 0)); then
    for n in $(seq 16 16 224); do sizes+=("${n}x${n}"); done
    sizes+=(640x480 1280x720 1920x1080 2048x1080 3840x2160)
fi

for size in "${sizes[@]}"; do
    # A size that fails outright (e.g. pipeline error) must not stop the rest.
    "$SCRIPT_DIR/collect_supernova_benchmarks.sh" \
        --width "${size%x*}" --height "${size#*x}" \
        --out-dir "$SCRIPT_DIR/../benchmarks/supernova/$MACHINE/sweep" \
        "${flags[@]}" || echo "FAILED size $size" >&2
done
