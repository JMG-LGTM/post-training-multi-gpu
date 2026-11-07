#!/usr/bin/env python3
"""Export raw metrics, EWMA curves and PNG/SVG without changing training logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def flatten(value, prefix=""):
    result = {}
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, dict):
            result.update(flatten(item, key + "."))
        elif type(item) in (int, float):
            result[key] = item if math.isfinite(item) else None
    return result


def export(run_dir, output, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index + 1 == len(lines):
                continue  # A live writer may not have finished its final row.
            raise
        row = flatten(record)
        step = row.get("optimizer_update")
        if type(step) is int and step >= 0:
            records.append(row)
    if not records:
        raise ValueError("No completed optimizer-update metrics to plot")
    steps = [row["optimizer_update"] for row in records]
    if steps != sorted(set(steps)):
        raise ValueError("Metrics must have unique, increasing optimizer updates")
    columns = sorted(set().union(*(row.keys() for row in records)) - {"optimizer_update"})
    columns = ["optimizer_update", *columns]
    output.mkdir(parents=True, exist_ok=True)

    def save_csv(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    save_csv(output / "metrics_raw.csv", records)
    smoothed, running = [], {}
    loss_columns = [key for key in columns if "loss" in key]
    for row in records:
        result = dict(row)
        for key in loss_columns:
            value = row.get(key)
            if value is not None:
                running[key] = value if key not in running else alpha * value + (1 - alpha) * running[key]
                result[key] = running[key]
        smoothed.append(result)
    save_csv(output / "metrics_smoothed.csv", smoothed)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    groups = [
        ("Training losses", [key for key in loss_columns if "val" not in key]),
        ("Held-out losses", [key for key in loss_columns if "val" in key]),
        ("Preference ranking accuracy", [key for key in columns if "accuracy" in key]),
    ]
    for axis, (title, keys) in zip(axes, groups):
        for key in keys:
            raw = [(r["optimizer_update"], r[key]) for r in records if r.get(key) is not None]
            if not raw:
                continue
            x, y = zip(*raw)
            if key in loss_columns:
                line, = axis.plot(x, y, alpha=.2, linewidth=.8)
                smooth = [(r["optimizer_update"], r[key]) for r in smoothed if r.get(key) is not None]
                axis.plot(*zip(*smooth), label=key, color=line.get_color(), linewidth=1.5)
            else:
                axis.plot(x, y, label=key, linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Completed Q/V optimizer updates")
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "No recorded values", ha="center", va="center", transform=axis.transAxes)
    for suffix in ("png", "svg"):
        figure.savefig(output / ("loss." + suffix), dpi=160)
    plt.close(figure)
    receipt = dict(rows=len(records), last_update=steps[-1], ewma_alpha=alpha,
                   source=str(run_dir / "metrics.jsonl"), raw_metrics_unchanged=True)
    (output / "plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")
    print(json.dumps(export(args.run_dir, args.output or args.run_dir / "plots", args.alpha)))


if __name__ == "__main__":
    main()
