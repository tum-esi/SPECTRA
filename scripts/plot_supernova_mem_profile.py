#!/usr/bin/env python3
"""Plot the useful signals in a SuperNova memory-profile CSV.

The allocator totals are cumulative, while live memory and RSS often plateau.
This plot therefore emphasizes per-step time and allocation traffic, and shows
live-memory changes relative to the beginning of proving.

Usage:
    python3 scripts/plot_supernova_mem_profile.py
    python3 scripts/plot_supernova_mem_profile.py profile.csv --output profile.png
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MIB = 1024**2
GIB = 1024**3
REQUIRED_COLUMNS = {
    "elapsed_s",
    "phase",
    "step",
    "circuit_index",
    "allocated_bytes_total",
    "deallocated_bytes_total",
    "live_bytes",
    "peak_live_bytes",
    "current_rss_bytes",
    "peak_rss_bytes",
}
CIRCUITS = {
    0: ("DPC", "#0072B2"),
    1: ("Gain", "#D55E00"),
    2: ("Demosaic", "#009E73"),
}


@dataclass(frozen=True)
class Sample:
    elapsed: float
    phase: str
    step: int | None
    circuit: int | None
    allocated: int
    deallocated: int
    live: int
    peak_live: int
    rss: int | None
    peak_rss: int | None


@dataclass(frozen=True)
class StepDelta:
    sample: Sample
    duration: float
    allocated: int
    deallocated: int


def optional_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def load_profile(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise SystemExit(f"{path}: missing columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)

    if not raw_rows:
        raise SystemExit(f"{path}: no profile rows")

    samples = []
    for line, row in enumerate(raw_rows, start=2):
        try:
            samples.append(
                Sample(
                    elapsed=float(row["elapsed_s"]),
                    phase=row["phase"],
                    step=optional_int(row["step"]),
                    circuit=optional_int(row["circuit_index"]),
                    allocated=int(row["allocated_bytes_total"]),
                    deallocated=int(row["deallocated_bytes_total"]),
                    live=int(row["live_bytes"]),
                    peak_live=int(row["peak_live_bytes"]),
                    rss=optional_int(row["current_rss_bytes"]),
                    peak_rss=optional_int(row["peak_rss_bytes"]),
                )
            )
        except ValueError as error:
            raise SystemExit(f"{path}:{line}: invalid numeric value: {error}") from error

    metadata = {
        key: raw_rows[0].get(key, "").strip()
        for key in ("git_branch", "git_commit", "cpu_model", "gpu_model")
    }
    return samples, metadata


def proving_deltas(samples: list[Sample]) -> list[StepDelta]:
    result = []
    for previous, current in zip(samples, samples[1:]):
        if current.phase != "proving_step":
            continue
        if current.step is None or current.circuit is None:
            raise SystemExit("proving_step row is missing step or circuit_index")
        allocated = current.allocated - previous.allocated
        deallocated = current.deallocated - previous.deallocated
        duration = current.elapsed - previous.elapsed
        if allocated < 0 or deallocated < 0 or duration < 0:
            raise SystemExit("profile counters or elapsed time decreased")
        result.append(StepDelta(current, duration, allocated, deallocated))
    if not result:
        raise SystemExit("profile contains no proving_step rows")
    return result


def circuit_style(index: int):
    return CIRCUITS.get(index, (f"Circuit {index}", "#777777"))


def circuit_runs(steps: list[StepDelta]):
    runs = []
    start = 0
    for index in range(1, len(steps) + 1):
        if index == len(steps) or steps[index].sample.circuit != steps[start].sample.circuit:
            runs.append((start, index - 1, steps[start].sample.circuit))
            start = index
    return runs


def style_axis(ax):
    ax.grid(axis="y", color="#DEDEDE", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def shade_circuits(ax, steps: list[StepDelta], x_values, label=True):
    runs = circuit_runs(steps)
    for start, end, circuit in runs:
        name, color = circuit_style(circuit)
        left = x_values[start]
        right = x_values[end]
        if start > 0:
            left = (x_values[start - 1] + left) / 2
        if end + 1 < len(x_values):
            right = (right + x_values[end + 1]) / 2
        ax.axvspan(left, right, color=color, alpha=0.08, linewidth=0)
        if start > 0:
            ax.axvline(left, color="#777777", linewidth=0.8, linestyle=":")
        if label:
            ax.text(
                (left + right) / 2,
                0.98,
                f"{name} (circuit {circuit})",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                fontweight="bold",
            )


def metadata_subtitle(metadata):
    parts = []
    if metadata.get("git_branch"):
        commit = metadata.get("git_commit", "")[:10]
        parts.append(f"{metadata['git_branch']} @ {commit}" if commit else metadata["git_branch"])
    if metadata.get("cpu_model"):
        parts.append(metadata["cpu_model"])
    if metadata.get("gpu_model") and metadata["gpu_model"].lower() not in {
        "unknown",
        "unavailable",
    }:
        parts.append(metadata["gpu_model"])
    return " · ".join(parts)


def plot_profile(samples, steps, metadata, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    lifecycle, duration_ax, traffic_ax, change_ax = axes.flat
    elapsed = [sample.elapsed for sample in samples]

    lifecycle.plot(
        elapsed,
        [sample.live / MIB for sample in samples],
        label="Rust live requested",
        color="#0072B2",
        linewidth=2,
    )
    lifecycle.plot(
        elapsed,
        [sample.peak_live / MIB for sample in samples],
        label="Rust peak requested",
        color="#0072B2",
        linestyle="--",
        linewidth=1.4,
    )
    if all(sample.rss is not None for sample in samples):
        lifecycle.plot(
            elapsed,
            [sample.rss / MIB for sample in samples],
            label="Current RSS",
            color="#D55E00",
            linewidth=2,
        )
    if all(sample.peak_rss is not None for sample in samples):
        lifecycle.plot(
            elapsed,
            [sample.peak_rss / MIB for sample in samples],
            label="Peak RSS",
            color="#D55E00",
            linestyle="--",
            linewidth=1.4,
        )
    proof_elapsed = [step.sample.elapsed for step in steps]
    shade_circuits(lifecycle, steps, proof_elapsed, label=False)
    lifecycle_samples = [
        sample for sample in samples if sample.phase != "proving_step"
    ]
    for phase_number, sample in enumerate(lifecycle_samples, start=1):
        lifecycle.axvline(sample.elapsed, color="#999999", linewidth=0.7, alpha=0.7)
        lifecycle.text(
            sample.elapsed,
            0.98,
            str(phase_number),
            transform=lifecycle.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
            bbox={"boxstyle": "circle,pad=0.18", "fc": "white", "ec": "#888888"},
        )
    phase_labels = [
        f"{index} {sample.phase.replace('_', ' ')}"
        for index, sample in enumerate(lifecycle_samples, start=1)
    ]
    lifecycle.text(
        0.01,
        0.03,
        "   ".join(phase_labels[:3]) + "\n" + "   ".join(phase_labels[3:]),
        transform=lifecycle.transAxes,
        fontsize=7,
        color="#555555",
        va="bottom",
    )
    lifecycle.set_title("Lifecycle memory levels", loc="left")
    lifecycle.set_xlabel("elapsed time (s)")
    lifecycle.set_ylabel("memory (MiB)")
    lifecycle.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    style_axis(lifecycle)

    step_numbers = [step.sample.step for step in steps]
    shade_circuits(duration_ax, steps, step_numbers)
    duration_ax.plot(
        step_numbers,
        [step.duration for step in steps],
        color="#333333",
        linewidth=1.2,
    )
    duration_ax.scatter(
        step_numbers,
        [step.duration for step in steps],
        color=[circuit_style(step.sample.circuit)[1] for step in steps],
        s=13,
        zorder=3,
    )
    duration_ax.set_title("Approximate time between snapshots", loc="left")
    duration_ax.set_xlabel("proving step (zero-based)")
    duration_ax.set_ylabel("seconds")
    style_axis(duration_ax)

    shade_circuits(traffic_ax, steps, step_numbers)
    traffic_ax.plot(
        step_numbers,
        [step.allocated / MIB for step in steps],
        label="Allocated",
        color="#6F4E7C",
        linewidth=1.7,
    )
    traffic_ax.plot(
        step_numbers,
        [step.deallocated / MIB for step in steps],
        label="Deallocated",
        color="#E17C05",
        linewidth=1.2,
        linestyle="--",
    )
    traffic_ax.set_title("Rust allocation traffic per proving step", loc="left")
    traffic_ax.set_xlabel("proving step (zero-based)")
    traffic_ax.set_ylabel("MiB since previous snapshot")
    traffic_ax.legend(frameon=False, fontsize=8)
    style_axis(traffic_ax)

    baseline_live = steps[0].sample.live
    live_change = [(step.sample.live - baseline_live) / MIB for step in steps]
    shade_circuits(change_ax, steps, step_numbers)
    change_ax.plot(
        step_numbers,
        live_change,
        label="Rust live change",
        color="#0072B2",
        linewidth=2,
    )
    if all(step.sample.rss is not None for step in steps):
        baseline_rss = steps[0].sample.rss
        rss_change = [(step.sample.rss - baseline_rss) / MIB for step in steps]
        change_ax.plot(step_numbers, rss_change, label="RSS change", color="#D55E00", linewidth=2)
    change_ax.axhline(0, color="#777777", linewidth=0.8)
    change_ax.set_title("Retained-memory change during proving", loc="left")
    change_ax.set_xlabel("proving step (zero-based)")
    change_ax.set_ylabel("change from step 0 (MiB)")
    change_ax.legend(frameon=False, fontsize=8)
    style_axis(change_ax)

    title = "SuperNova memory profile"
    subtitle = metadata_subtitle(metadata)
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, x=0.01, ha="left", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def print_summary(samples: list[Sample], steps: list[StepDelta]):
    setup = next((sample for sample in samples if sample.phase == "setup_complete"), samples[0])
    last = steps[-1].sample
    print(f"Profile: {len(steps)} proving steps, {samples[-1].elapsed:.3f} s total")
    print(
        "Setup: "
        f"{(setup.allocated - samples[0].allocated) / GIB:.3f} GiB allocated, "
        f"{(setup.live - samples[0].live) / MIB:.3f} MiB retained"
    )
    print(
        "Proving: "
        f"{(last.allocated - setup.allocated) / GIB:.3f} GiB allocated, "
        f"{(last.deallocated - setup.deallocated) / GIB:.3f} GiB deallocated, "
        f"{(last.live - setup.live) / MIB:+.3f} MiB retained"
    )
    for circuit in dict.fromkeys(step.sample.circuit for step in steps):
        group = [step for step in steps if step.sample.circuit == circuit]
        name, _ = circuit_style(circuit)
        print(
            f"  {name} (circuit {circuit}, "
            f"steps {group[0].sample.step}–{group[-1].sample.step}): "
            f"{sum(step.duration for step in group):.3f} s total, "
            f"{median(step.duration for step in group):.3f} s median/step, "
            f"{sum(step.allocated for step in group) / GIB:.3f} GiB allocated"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("benchmarks/supernova_mem_profile.csv"),
        help="memory-profile CSV (default: benchmarks/supernova_mem_profile.csv)",
    )
    parser.add_argument("--output", type=Path, help="PNG path (default: INPUT with _plot suffix)")
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_plot.png")

    samples, metadata = load_profile(args.input)
    steps = proving_deltas(samples)
    print_summary(samples, steps)
    plot_profile(samples, steps, metadata, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

