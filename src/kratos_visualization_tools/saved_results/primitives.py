"""Plot styles, rendering primitives, and pending figure output."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class DataSeries:
    data_points: list[tuple[float, float]]
    label: str = ""
    line_style: str = "-"
    marker: str | None = None
    color: str | None = None


@dataclass
class AxisOptions:
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xscale: str | None = None
    yscale: str | None = None
    invert_yaxis: bool = False


def _plot_on_axis(axes, series_collection: Iterable[DataSeries]):
    lines = []
    for series in series_collection:
        if not series.data_points:
            continue
        lines.extend(
            axes.plot(
                *zip(*series.data_points),
                label=series.label,
                linestyle=series.line_style,
                marker=series.marker,
                color=series.color,
            )
        )
    axes.grid()
    axes.grid(which="minor", color="0.9")
    return lines


def _apply_axis_options(axes, options: AxisOptions):
    if options.xscale:
        axes.set_xscale(options.xscale)
    if options.yscale:
        axes.set_yscale(options.yscale, nonpositive="mask")
    if options.title:
        axes.set_title(options.title)
    if options.xlabel:
        axes.set_xlabel(options.xlabel)
    if options.ylabel:
        axes.set_ylabel(options.ylabel)
    axes.yaxis.set_inverted(options.invert_yaxis)


def _unique_label_handles(lines):
    handles, seen = [], set()
    for line in lines:
        label = line.get_label()
        if label not in seen:
            handles.append(line)
            seen.add(label)
    return handles


@dataclass
class PendingFigure:
    figure: Any
    path: Path
    preview: Any


# When not None, figures are collected for the viewer instead of being written.
_pending_figures: list[PendingFigure] | None = None


def _write(figure, plot_file_path: Path):
    plot_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to {plot_file_path.resolve()}")
    figure.savefig(plot_file_path, bbox_inches="tight")


def _render_preview(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=80, bbox_inches="tight")
    buffer.seek(0)
    return plt.imread(buffer)


def _save(figure, plot_file_path: Path):
    if _pending_figures is not None:
        _pending_figures.append(
            PendingFigure(figure, plot_file_path, _render_preview(figure))
        )
    else:
        _write(figure, plot_file_path)
    plt.close(figure)


def make_single_plot(
    series_collection: list[DataSeries],
    plot_file_path: Path,
    options: AxisOptions,
    annotation: str | None = None,
):
    figure, axes = plt.subplots(layout="constrained")
    _plot_on_axis(axes, series_collection)
    _apply_axis_options(axes, options)
    if annotation:
        axes.text(
            0.98,
            0.02,
            annotation,
            transform=axes.transAxes,
            va="bottom",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    figure.legend(loc="outside center right")
    _save(figure, plot_file_path)


def make_sub_plots(
    series_collections: list[list[DataSeries]],
    plot_file_path: Path,
    options_per_subplot: list[AxisOptions],
    max_plots_per_row: int = 5,
    figsize: tuple[float, float] = (20.0, 6.0),
    share_ylabel_on_first_column: bool = True,
):
    num_rows = max(1, math.ceil(len(series_collections) / max_plots_per_row))
    num_cols = max(1, math.ceil(len(series_collections) / num_rows))
    figure, axes = plt.subplots(
        num_rows, num_cols, figsize=figsize, layout="constrained", squeeze=False
    )
    flat_axes = [ax for row in axes for ax in row]

    lines = []
    for index, (ax, collection, axis_options) in enumerate(
        zip(flat_axes, series_collections, options_per_subplot)
    ):
        lines.extend(_plot_on_axis(ax, collection))
        if share_ylabel_on_first_column and index % num_cols != 0:
            axis_options = AxisOptions(**{**vars(axis_options), "ylabel": None})
        _apply_axis_options(ax, axis_options)

    for ax in flat_axes[len(series_collections) :]:
        ax.set_axis_off()

    figure.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        handles=_unique_label_handles(lines),
    )
    _save(figure, plot_file_path)
