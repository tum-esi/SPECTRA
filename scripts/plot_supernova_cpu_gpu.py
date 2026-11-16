#!/usr/bin/env python3
"""Plot matched SuperNova CPU/GPU benchmark runs.

The input CSVs contain repeated runs for several fold batch sizes.  Each plot
shows the median and the full observed range instead of treating repetitions
as independent configurations.

Figures written to the output directory:
  time-and-memory-vs-batch-size.png
  proving-time-vs-memory-pareto.png

Usage:
    python3 scripts/plot_supernova_cpu_gpu.py \
        --cpu src/supernova/supernova_720p_cpu.csv \
        --gpu src/supernova/supernova_720p_gpu.csv \
        --out charts/supernova-720p

Both input paths must be provided as arguments. The output directory defaults
to the directory from which the script was invoked.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {"CPU": "#0072B2", "GPU": "#D55E00"}
MARKERS = {"CPU": "o", "GPU": "s"}
STAGE_LINESTYLES = {"DPC": "-", "Gain": "--", "Demosaic": ":", "WB": "-."}
# Pipeline order. WB is optional: CSVs from before the white-balance stage
# have no wbal column and plot with the first three stages only.
STAGE_COLUMNS = {
    "DPC": "dpc_proving_time_s",
    "Gain": "gain_proving_time_s",
    "Demosaic": "demosaic_proving_time_s",
    "WB": "wbal_proving_time_s",
}
OPTIONAL_STAGES = {"WB"}
REQUIRED_COLUMNS = {
    "width_pixels",
    "height_pixels",
    "batch_size",
    "proving_time_s",
    "peak_rss_mib",
} | {column for stage, column in STAGE_COLUMNS.items() if stage not in OPTIONAL_STAGES}


@dataclass(frozen=True)
class Summary:
    system: str
    batch_size: int
    runs: int
    time_median: float
    time_min: float
    time_max: float
    memory_median: float
    memory_min: float
    memory_max: float
    stage_time_medians: dict  # stage name -> median proving time (s)


def load_runs(path: Path, system: str):
    """Load one CSV and return its resolution and runs grouped by batch size."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
        stages = [
            stage for stage, column in STAGE_COLUMNS.items() if column in reader.fieldnames
        ]

    if not rows:
        raise SystemExit(f"{path}: no benchmark rows")

    resolutions = {(int(row["width_pixels"]), int(row["height_pixels"])) for row in rows}
    if len(resolutions) != 1:
        raise SystemExit(f"{path}: expected one resolution, found {sorted(resolutions)}")

    grouped = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            batch_size = int(row["batch_size"])
            proving_time = float(row["proving_time_s"])
            stage_times = [float(row[STAGE_COLUMNS[stage]]) for stage in stages]
            peak_memory = float(row["peak_rss_mib"])
        except ValueError as error:
            raise SystemExit(f"{path}:{row_number}: invalid numeric value: {error}") from error
        if (
            batch_size < 1
            or proving_time <= 0
            or any(time <= 0 for time in stage_times)
            or peak_memory <= 0
        ):
            raise SystemExit(f"{path}:{row_number}: expected positive benchmark values")
        grouped.setdefault(batch_size, []).append((proving_time, peak_memory, stage_times))

    summaries = []
    for batch_size, values in sorted(grouped.items()):
        times = [value[0] for value in values]
        memories = [value[1] for value in values]
        summaries.append(
            Summary(
                system=system,
                batch_size=batch_size,
                runs=len(values),
                time_median=median(times),
                time_min=min(times),
                time_max=max(times),
                memory_median=median(memories),
                memory_min=min(memories),
                memory_max=max(memories),
                stage_time_medians={
                    stage: median(value[2][index] for value in values)
                    for index, stage in enumerate(stages)
                },
            )
        )
    return resolutions.pop(), summaries


def error_range(center, low, high):
    return [[center - low], [high - center]]


def style_axes(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def add_method_note(fig, summaries):
    run_counts = sorted({summary.runs for summary in summaries})
    count_text = str(run_counts[0]) if len(run_counts) == 1 else "/".join(map(str, run_counts))
    fig.text(
        0.01,
        0.01,
        "Markers show medians; whiskers show the observed range for total proving time "
        f"and memory ({count_text} runs per point). Stage lines show medians.",
        fontsize=8,
        color="#555555",
    )


def draw_batch_metric(ax, summaries, metric, ylabel, title):
    batch_sizes = sorted({summary.batch_size for summary in summaries})
    positions = {batch_size: index for index, batch_size in enumerate(batch_sizes)}

    for system in ("CPU", "GPU"):
        points = [summary for summary in summaries if summary.system == system]
        centers = [getattr(point, f"{metric}_median") for point in points]
        lows = [getattr(point, f"{metric}_min") for point in points]
        highs = [getattr(point, f"{metric}_max") for point in points]
        xs = [positions[point.batch_size] for point in points]
        yerr = [
            [center - low for center, low in zip(centers, lows)],
            [high - center for center, high in zip(centers, highs)],
        ]
        ax.errorbar(
            xs,
            centers,
            yerr=yerr,
            color=COLORS[system],
            marker=MARKERS[system],
            markersize=7,
            linewidth=2,
            capsize=4,
            label=system,
            zorder=3,
        )

    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("fold batch size B")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(batch_sizes)), [str(value) for value in batch_sizes])
    style_axes(ax)


def draw_stage_breakdown(ax, summaries):
    batch_sizes = sorted({summary.batch_size for summary in summaries})
    positions = {batch_size: index for index, batch_size in enumerate(batch_sizes)}
    stages = [
        stage
        for stage in STAGE_COLUMNS
        if any(stage in summary.stage_time_medians for summary in summaries)
    ]

    for system in ("CPU", "GPU"):
        points = [summary for summary in summaries if summary.system == system]
        for stage in stages:
            measured = [point for point in points if stage in point.stage_time_medians]
            if not measured:
                continue
            ax.plot(
                [positions[point.batch_size] for point in measured],
                [point.stage_time_medians[stage] for point in measured],
                color=COLORS[system],
                marker=MARKERS[system],
                markersize=4,
                linewidth=1.6,
                linestyle=STAGE_LINESTYLES[stage],
                alpha=0.9,
                zorder=3,
            )

    ax.set_title("Proving time by stage", loc="left", fontsize=12)
    ax.set_xlabel("fold batch size B")
    ax.set_ylabel("stage proving time (s)")
    ax.set_xticks(range(len(batch_sizes)), [str(value) for value in batch_sizes])
    style_axes(ax)
    stage_handles = [
        Line2D(
            [],
            [],
            color="#444444",
            linewidth=1.8,
            linestyle=STAGE_LINESTYLES[stage],
            label=stage,
        )
        for stage in stages
    ]
    ax.legend(handles=stage_handles, loc="best", frameon=False)


def plot_batch_metrics(summaries, resolution, output):
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0))
    draw_batch_metric(
        axes[0], summaries, "time", "proving time (s)", "Proving time"
    )
    draw_stage_breakdown(axes[1], summaries)
    draw_batch_metric(
        axes[2], summaries, "memory", "peak process memory (MiB)", "Peak memory"
    )

    width, height = resolution
    fig.suptitle(
        f"Chunk-size performance and memory — {width}×{height}",
        x=0.055,
        ha="left",
        fontsize=14,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    add_method_note(fig, summaries)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91), w_pad=2.5)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def pareto_frontier(summaries):
    """Return points not dominated on both median time and median memory."""
    return [
        point
        for point in summaries
        if not any(
            other != point
            and other.time_median <= point.time_median
            and other.memory_median <= point.memory_median
            and (
                other.time_median < point.time_median
                or other.memory_median < point.memory_median
            )
            for other in summaries
        )
    ]


def plot_pareto(summaries, resolution, output):
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    frontier = sorted(pareto_frontier(summaries), key=lambda point: point.memory_median)

    if len(frontier) > 1:
        ax.plot(
            [point.memory_median for point in frontier],
            [point.time_median for point in frontier],
            color="#666666",
            linewidth=1.5,
            linestyle="--",
            label="Pareto frontier",
            zorder=1,
        )

    for point in summaries:
        is_pareto = point in frontier
        ax.errorbar(
            point.memory_median,
            point.time_median,
            xerr=error_range(point.memory_median, point.memory_min, point.memory_max),
            yerr=error_range(point.time_median, point.time_min, point.time_max),
            color=COLORS[point.system],
            marker=MARKERS[point.system],
            markerfacecolor=COLORS[point.system] if is_pareto else "white",
            markeredgewidth=1.8,
            markersize=8,
            capsize=3,
            alpha=1.0 if is_pareto else 0.65,
            zorder=3,
        )
        ax.annotate(
            f"B={point.batch_size}",
            (point.memory_median, point.time_median),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    width, height = resolution
    ax.set_title(f"Proving time–memory trade-off — {width}×{height}", loc="left", fontsize=13)
    ax.set_xlabel("peak process memory (MiB; lower is better)")
    ax.set_ylabel("proving time (s; lower is better)")
    ax.grid(color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([], [], color=COLORS[system], marker=MARKERS[system], linewidth=0, label=system)
        for system in ("CPU", "GPU")
    ]
    handles.extend(
        [
            Line2D([], [], color="#666666", linestyle="--", label="Pareto frontier"),
            Line2D([], [], color="#444444", marker="o", markerfacecolor="#444444", linewidth=0,
                   label="Pareto-optimal"),
            Line2D([], [], color="#777777", marker="o", markerfacecolor="white", linewidth=0,
                   label="Dominated"),
        ]
    )
    ax.legend(handles=handles, frameon=False, ncol=2)
    add_method_note(fig, summaries)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def print_summary(summaries):
    by_system = {
        system: {point.batch_size: point for point in summaries if point.system == system}
        for system in ("CPU", "GPU")
    }
    print("\nMedian results")
    print("B    CPU time    GPU time    GPU speedup    CPU MiB    GPU MiB")
    for batch_size in sorted(by_system["CPU"].keys() & by_system["GPU"].keys()):
        cpu = by_system["CPU"][batch_size]
        gpu = by_system["GPU"][batch_size]
        speedup = cpu.time_median / gpu.time_median
        print(
            f"{batch_size:<4} {cpu.time_median:>8.2f}s   {gpu.time_median:>8.2f}s"
            f"      {speedup:>5.2f}×    {cpu.memory_median:>7.1f}    {gpu.memory_median:>7.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        type=Path,
        help="CPU benchmark CSV",
    )
    parser.add_argument(
        "--gpu",
        type=Path,
        help="GPU benchmark CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.cwd(),
        help="output directory (default: current working directory)",
    )
    args = parser.parse_args()

    missing = [
        flag
        for flag, path in (("--cpu", args.cpu), ("--gpu", args.gpu))
        if path is None
    ]
    if missing:
        parser.error(
            f"benchmark CSV path must be provided with {' and '.join(missing)}\n"
            "example: python3 scripts/plot_supernova_cpu_gpu.py "
            "--cpu benchmarks/cpu.csv --gpu benchmarks/gpu.csv"
        )

    cpu_resolution, cpu = load_runs(args.cpu, "CPU")
    gpu_resolution, gpu = load_runs(args.gpu, "GPU")
    if cpu_resolution != gpu_resolution:
        raise SystemExit(
            f"resolution mismatch: CPU is {cpu_resolution[0]}×{cpu_resolution[1]}, "
            f"GPU is {gpu_resolution[0]}×{gpu_resolution[1]}"
        )
    cpu_batchs = {point.batch_size for point in cpu}
    gpu_batchs = {point.batch_size for point in gpu}
    # A batch size can fail on one backend only (e.g. GPU out of memory); plot
    # what was measured.
    if cpu_batchs != gpu_batchs:
        print(
            f"warning: batch-size mismatch: CPU has {sorted(cpu_batchs)}, "
            f"GPU has {sorted(gpu_batchs)}"
        )

    summaries = cpu + gpu
    args.out.mkdir(parents=True, exist_ok=True)
    plot_batch_metrics(
        summaries,
        cpu_resolution,
        args.out / "time-and-memory-vs-batch-size.png",
    )
    plot_pareto(
        summaries,
        cpu_resolution,
        args.out / "proving-time-vs-memory-pareto.png",
    )
    print_summary(summaries)


if __name__ == "__main__":
    main()
