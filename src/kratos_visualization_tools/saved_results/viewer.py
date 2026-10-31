"""Interactive preview and save controls for pending figures."""

from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .data import read_table
from .primitives import DataSeries, PendingFigure, _plot_on_axis, _render_preview, _write


class FigureViewer:
    """Window that shows the figures together or one by one, with a Save button."""

    def __init__(self, pending_figures: list[PendingFigure]):
        self.pending_figures = pending_figures
        self.mode = "all"
        self.index = 0
        self.figure = plt.figure("plot_saved_results", figsize=(14, 9))
        self._content_axes = []
        self._buttons = []
        for position, (label, callback) in enumerate(
            (
                ("All together", self.show_all),
                ("One by one", self.show_one_by_one),
                ("< Previous", self.previous),
                ("Next >", self.next),
                ("Save", self.save),
                ("Add profile...", self.add_profile),
            )
        ):
            axes = self.figure.add_axes([0.02 + position * 0.16, 0.02, 0.15, 0.05])
            button = Button(axes, label)
            button.on_clicked(callback)
            self._buttons.append(button)
        self._render()
        self.figure.canvas.mpl_connect("button_press_event", self._select)

    def _select(self, event):
        if event.inaxes in self._content_axes:
            position = self._content_axes.index(event.inaxes)
            if self.mode == "all":
                self.index = position
                self._render()

    def add_profile(self, _event=None):
        from tkinter import filedialog, messagebox

        filenames = filedialog.askopenfilenames(
            title="Add profiles to the selected graph",
            filetypes=[("Profile CSV", "*.csv")],
            parent=self.figure.canvas.manager.window,
        )
        if not filenames:
            return
        try:
            self.import_profiles([Path(filename) for filename in filenames])
        except (OSError, KeyError, ValueError) as error:
            messagebox.showerror(
                "Add profile", str(error), parent=self.figure.canvas.manager.window)

    def import_profiles(self, paths: list[Path]):
        """Read all inputs before modifying the selected graph."""
        series = []
        for path in paths:
            table = read_table(path)
            coordinate = ("y_coordinate" if table.header and
                          "y_coordinate" in table.header else "coordinate")
            values = table.column("value")
            coordinates = table.column(coordinate)
            groups = (table.column("series") if table.header and
                      "series" in table.header else [""] * len(values))
            grouped = {}
            for value, y, group in zip(values, coordinates, groups):
                point = float(value), float(y)
                if not all(math.isfinite(number) for number in point):
                    raise ValueError(f"Non-finite profile data in {path.name}.")
                grouped.setdefault(group, []).append(point)
            if not grouped:
                raise ValueError(f"No profile data in {path.name}.")
            for group, points in grouped.items():
                label = f"{path.stem}: {group}" if group else path.stem
                series.append(DataSeries(sorted(points, key=lambda point: point[1]), label))
        self._overlay(self.pending_figures[self.index], series)
        self._render()

    @staticmethod
    def _overlay(pending, series):
        if len(pending.figure.axes) != 1:
            raise ValueError("Select a single-axis graph before adding a profile.")
        axes = pending.figure.axes[0]
        labels = {line.get_label() for line in axes.lines}
        for item in series:
            base = item.label
            suffix = 2
            while item.label in labels:
                item.label = f"{base} ({suffix})"
                suffix += 1
            labels.add(item.label)
        _plot_on_axis(axes, series)
        axes.relim()
        axes.autoscale_view()
        for legend in list(pending.figure.legends):
            legend.remove()
        pending.figure.legend(loc="outside center right")
        pending.preview = _render_preview(pending.figure)

    def add_configuration(self, path: Path, output_dir: Path | None = None):
        from .config import run

        config = json.loads(path.read_text(encoding="utf-8"))
        incoming = run(path, output_dir, show=True)
        if not incoming:
            raise ValueError("The configuration contains no plottable data.")
        is_profile = config.get("metadata", {}).get("mode") in ("single", "comparison")
        for pending in incoming:
            target = None
            if is_profile and len(pending.figure.axes) == 1:
                source = pending.figure.axes[0]
                candidates = [self.pending_figures[self.index]] + self.pending_figures
                target = next((candidate for candidate in candidates
                               if len(candidate.figure.axes) == 1
                               and candidate.figure.axes[0].get_xlabel() == source.get_xlabel()
                               and candidate.figure.axes[0].get_ylabel() == source.get_ylabel()),
                              None)
            if target is None:
                self.pending_figures.append(pending)
                self.index = len(self.pending_figures) - 1
            else:
                series = [DataSeries(
                    list(zip(line.get_xdata(), line.get_ydata())),
                    line.get_label(), line.get_linestyle(), line.get_marker())
                    for line in source.lines]
                self._overlay(target, series)
                self.index = self.pending_figures.index(target)
        self._render()
        manager = self.figure.canvas.manager
        if hasattr(manager, "window"):
            manager.window.deiconify()
            manager.window.lift()

    def show_all(self, _event=None):
        self.mode = "all"
        self._render()

    def show_one_by_one(self, _event=None):
        self.mode = "single"
        self._render()

    def previous(self, _event=None):
        self.mode = "single"
        self.index = (self.index - 1) % len(self.pending_figures)
        self._render()

    def next(self, _event=None):
        self.mode = "single"
        self.index = (self.index + 1) % len(self.pending_figures)
        self._render()

    def save(self, _event=None):
        for pending in self.pending_figures:
            _write(pending.figure, pending.path)

    def _render(self):
        for axes in self._content_axes:
            axes.remove()
        self._content_axes.clear()

        if self.mode == "all":
            shown = self.pending_figures
        else:
            shown = [self.pending_figures[self.index]]
        num_cols = math.ceil(math.sqrt(len(shown)))
        num_rows = math.ceil(len(shown) / num_cols)
        grid = self.figure.add_gridspec(
            num_rows, num_cols, left=0.02, right=0.98, bottom=0.1, top=0.95
        )
        for position, pending in enumerate(shown):
            axes = self.figure.add_subplot(grid[position // num_cols, position % num_cols])
            axes.imshow(pending.preview)
            axes.set_axis_off()
            title = pending.path.name
            if self.mode == "single":
                title += f"  ({self.index + 1}/{len(self.pending_figures)})"
            elif position == self.index:
                title += "  [selected]"
            axes.set_title(title, fontsize=9)
            self._content_axes.append(axes)
        self.figure.canvas.draw_idle()
