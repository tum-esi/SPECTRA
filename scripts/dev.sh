#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
COMPOSE_FILE="$SCRIPT_DIR/compose.yaml"
SUPERNOVA_DIR=/workspace/src/supernova
USE_DOCKER_ALIAS_CMD=0
RUST_SERVICE=rust

case "${1:-}" in
    --gpu)
        RUST_SERVICE=rust-gpu
        shift
        ;;
esac

# Use the configured Docker CLI normally. Managed servers can provide an
# explicit fallback command when their Docker daemon uses a custom socket.
if ! docker info >/dev/null 2>&1 && [ -n "${DOCKER_ALIAS_CMD:-}" ]; then
    USE_DOCKER_ALIAS_CMD=1
fi

docker_cli() {
    if [ "$USE_DOCKER_ALIAS_CMD" -eq 1 ]; then
        # Intentionally split a simple command such as "sudo docker -H URL".
        set -f
        $DOCKER_ALIAS_CMD "$@"
    else
        docker "$@"
    fi
}

compose() {
    docker_cli compose -f "$COMPOSE_FILE" "$@"
}

usage() {
    echo "Usage: $0 {build|pipeline|check|run|benchmark|profile|mem-profile}"
}

case "${1:-}" in
    build)
        compose build "$RUST_SERVICE"
        ;;
    pipeline)
        shift
        compose run --rm "$RUST_SERVICE" python scripts/python/pipeline.py "$@"
        ;;
    check)
        compose run --rm -w "$SUPERNOVA_DIR" "$RUST_SERVICE" cargo run --release -- --check
        ;;
    run)
        compose run --rm -w "$SUPERNOVA_DIR" "$RUST_SERVICE" cargo run --release
        ;;
    benchmark)
        shift
        compose run --rm -w "$SUPERNOVA_DIR" "$RUST_SERVICE" cargo run --release -- --benchmark "$@"
        ;;
    profile)
        compose run --rm -w "$SUPERNOVA_DIR" "$RUST_SERVICE" \
            cargo run --profile profiling -- --profile
        ;;
    mem-profile)
        shift
        PROFILE_CSV=/workspace/benchmarks/supernova_mem_profile.csv
        PROFILE_PNG=/workspace/benchmarks/supernova_mem_profile_plot.png

        compose run --rm -w "$SUPERNOVA_DIR" "$RUST_SERVICE" \
            cargo run --release --features mem-profile -- \
            --mem-profile --mem-profile-output "$PROFILE_CSV"

        compose run --rm "$RUST_SERVICE" \
            python scripts/plot_supernova_mem_profile.py \
            "$PROFILE_CSV" --output "$PROFILE_PNG"
        ;;

    *)
        usage >&2
        exit 2
        ;;
esac
