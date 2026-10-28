"""Interactive preview and save controls for pending figures."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .primitives import PendingFigure, _write


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
            )
        ):
            axes = self.figure.add_axes([0.02 + position * 0.195, 0.02, 0.18, 0.05])
            button = Button(axes, label)
            button.on_clicked(callback)
            self._buttons.append(button)
        self._render()

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
            axes.set_title(title, fontsize=9)
            self._content_axes.append(axes)
        self.figure.canvas.draw_idle()
