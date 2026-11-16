#!/usr/bin/env python3
"""Body rows of the paper's size table from sweep_sizes.sh results.

Each cell is the median proving time at the best batch size B (subscript):
the fastest B whose median peak RSS fits --mem-budget-mib (no budget by
default). Machine sweep directories are given in the table's column order; a
machine without GPU CSVs gets a CPU column only. Missing cells print as --.

    python3 scripts/sizes_table.py --mem-budget-mib 8192 \
        benchmarks/supernova/Jetson/sweep benchmarks/supernova/ServerA/sweep

With --fill, one machine's columns are written in place into the table of an
existing .tex file instead; --machine names the column group as it appears in
the table header:

    python3 scripts/sizes_table.py benchmarks/supernova/ServerC/sweep \
        --fill paper/evaluation.tex --machine "Server C"
"""

import argparse
import re
from pathlib import Path

from plot_supernova_cpu_gpu import load_runs

COMMON = {(640, 480): "480p", (1280, 720): "720p", (1920, 1080): "1080p",
          (2048, 1080): "2K", (3840, 2160): "4K"}


def best_runs(root, budget):
    """{((w, h), "cpu"|"gpu"): Summary at the best B, or None if none fits}."""
    cells = {}
    for path in Path(root).glob("*x*-*-*.csv"):
        size, mode, _commit = path.stem.split("-", 2)
        key = (tuple(map(int, size.split("x"))), mode)
        if key in cells:
            raise SystemExit(f"{root}: several {size} {mode} CSVs (different commits?)")
        _, summaries = load_runs(path, mode.upper())
        cells[key] = min(
            (s for s in summaries if s.memory_median <= budget),
            key=lambda s: s.time_median,
            default=None,
        )
    return cells


def cell(summary):
    return "--" if summary is None else f"{summary.time_median:.1f}$_{{{summary.batch_size}}}$"


def size_label(wh):
    return COMMON.get(wh) or (f"${wh[0]}^2$" if wh[0] == wh[1] else f"${wh[0]} \\times {wh[1]}$")


def fill(tex, machine, cells, modes):
    """Overwrite machine's cells in the tab:sizes rows of tex, matched by size label."""
    lines = tex.read_text().split("\n")
    start = next(i for i, line in enumerate(lines) if r"\label{tab:sizes}" in line)
    end = next(i for i in range(start, len(lines)) if r"\end{tabular}" in lines[i])
    header = next((i for i in range(start, end) if machine in lines[i]), None)
    if header is None:
        raise SystemExit(f"{tex}: no column group named {machine!r}")
    # Column of the machine's first cell: label column plus the spans before it.
    first = 1
    for group in lines[header].rstrip().removesuffix(r"\\").split("&")[1:]:
        span = re.search(r"\\multicolumn\{(\d+)\}", group)
        width = int(span.group(1)) if span else 1
        if re.search(rf"\{{{re.escape(machine)}\}}|^\s*{re.escape(machine)}\s*$", group):
            break
        first += width
    else:
        raise SystemExit(f"{tex}: no column group named {machine!r}")
    if width != len(modes):
        raise SystemExit(f"{machine} spans {width} column(s) but the sweep has {'/'.join(modes)}")

    labels = {size_label(wh): wh for wh, _ in cells}
    filled = set()
    for i in range(header + 1, end):
        row = lines[i].rstrip().removesuffix(r"\\").split("&")
        wh = labels.get(row[0].strip())
        if wh is None:
            continue
        for k, mode in enumerate(modes):
            row[first + k] = f" {cell(cells.get((wh, mode)))} "
        lines[i] = "&".join(row) + r"\\"
        filled.add(wh)
    tex.write_text("\n".join(lines))
    print(f"filled {machine} in {len(filled)} rows of {tex}")
    for wh in sorted(set(labels.values()) - filled, key=lambda wh: wh[0] * wh[1]):
        print(f"no table row for {size_label(wh)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps", nargs="+", type=Path, help="sweep directory per machine")
    parser.add_argument("--mem-budget-mib", type=float, default=float("inf"))
    parser.add_argument("--fill", type=Path, help=".tex file whose tab:sizes to update in place")
    parser.add_argument("--machine", help="column group name in the table header (with --fill)")
    args = parser.parse_args()
    if args.fill and (len(args.sweeps) != 1 or not args.machine):
        parser.error("--fill takes exactly one sweep directory and --machine")

    machines = []
    for root in args.sweeps:
        cells = best_runs(root, args.mem_budget_mib)
        modes = ["cpu", "gpu"] if any(mode == "gpu" for _, mode in cells) else ["cpu"]
        machines.append((cells, modes))
    if args.fill:
        return fill(args.fill, args.machine, *machines[0])

    sizes = sorted({wh for cells, _ in machines for wh, _ in cells},
                   key=lambda wh: (wh not in COMMON, wh[0] * wh[1]))
    ncols = 1 + sum(len(modes) for _, modes in machines)
    group = None
    for wh in sizes:
        name = "Common frame sizes" if wh in COMMON else "Automotive ISP sizes"
        if name != group:
            if group:
                print(r"    \midrule")
            print(rf"    \multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{name}}}}} \\")
            group = name
        label = size_label(wh)
        row = [cell(cells.get((wh, mode))) for cells, modes in machines for mode in modes]
        print("    " + " & ".join([label] + row) + r" \\")


if __name__ == "__main__":
    main()
