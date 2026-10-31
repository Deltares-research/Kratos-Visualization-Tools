"""Optional desktop controls using the shared two-solution comparison engine."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .comparison import ComparisonViewer
from .config import parse_profile_path
from .models import TENSOR_COMPONENTS, VECTOR_COMPONENTS
from .qt_viewer import GidMainWindow, _render_gui


class QtComparisonViewer(ComparisonViewer):
    """Keep comparison calculations and drawing in a Qt-owned figure."""

    def __init__(self, window, **options):
        self.window = window
        self._profile_layout = None
        minimum, maximum = options.get("vmin"), options.get("vmax")
        if any(value is not None and not math.isfinite(value) for value in (minimum, maximum)):
            raise ValueError("Color range limits must be finite.")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError("Color range requires minimum < maximum.")
        super().__init__(**options)

    def _ensure_comparison_figure(self):
        if self.fig is None:
            self.fig = Figure(figsize=(12, 8), layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
            self.canvas.mpl_connect("button_press_event", self.window.inspect_click)
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        if self._profile_layout != self.show_profile:
            self.fig.clear()
            if self.show_profile:
                self.ax0, self.ax1, self.ax2 = self.fig.subplots(1, 3)
            else:
                self.ax0 = self.fig.subplots()
                self.ax1 = self.ax2 = None
            self._profile_layout = self.show_profile
        else:
            for axes in (self.ax0, self.ax1, self.ax2):
                if axes is not None:
                    axes.clear()
                    axes.set_axis_on()

    def _draw_menu(self):
        pass

    def _show_comparison_error(self, exc):
        raise exc

    def _show_comparison_popup(self, x, y):
        columns, rows = self._build_comparison_table(x, y)
        self.window.report.setPlainText(
            f"Values near ({x:g}, {y:g}) - {self.inspection_method}\n\n"
            + "\n\n".join(columns)
            + "\n\n"
            + "\n\n".join(
                "\n".join(f"{column.splitlines()[0]}: {value}"
                          for column, value in zip(columns, row))
                for row in rows
            )
        )
        self.window.report.show()


class GidComparisonWindow(GidMainWindow):
    """Desktop comparison controls sharing the single-view action/save helpers."""

    _render_state_names = ("fig", "ax0", "ax1", "ax2", "colorbar", "_profile_layout")

    def __init__(self, **options):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("GiD Solution Comparison")
        self.resize(1400, 850)
        self.timer = QtCore.QTimer(self)
        self.report = QtWidgets.QPlainTextEdit(self)
        self.report.setReadOnly(True)
        self.report.setWindowFlag(QtCore.Qt.WindowType.Window)
        self.report.setWindowTitle("Comparison coordinate inspection")
        self.report.resize(800, 600)
        self.report.hide()
        self.viewer = QtComparisonViewer(self, **options)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(240)
        sidebar = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(sidebar)
        self.form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.form.setContentsMargins(8, 8, 8, 8)
        self.form.setSpacing(4)
        scroll.setWidget(sidebar)
        layout.addWidget(scroll)
        plot = QtWidgets.QVBoxLayout()
        self.toolbar = NavigationToolbar2QT(self.viewer.canvas, self)
        plot.addWidget(self.toolbar)
        plot.addWidget(self.viewer.canvas)
        layout.addLayout(plot, 1)

        for solution in self.viewer.solutions:
            label = QtWidgets.QLabel(solution.msh_path.name)
            label.setToolTip(str(solution.msh_path))
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
            label.setWordWrap(True)
            self.form.addRow(solution.label, label)
        self.result = self.combo("Result", [], self.select_result)
        self.location = self.combo("Location", [], self.select_location)
        self.step = self.combo("Step", [], self.select_step)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.form.addRow("Step index", self.slider)
        self.slider.valueChanged.connect(
            lambda index: self.run_action(lambda: self.select_step(self.step.itemText(index))))
        self.button("Previous step", lambda: self.move_step(-1))
        self.button("Next step", lambda: self.move_step(1))
        self.component = self.combo(
            "Vector component", VECTOR_COMPONENTS,
            lambda value: self.change("current_component", value))
        self.tensor = self.combo(
            "Tensor reduction", TENSOR_COMPONENTS,
            lambda value: self.change("current_tensor_component", value))
        self.style = self.combo("Style", self.viewer.styles, self.select_style)
        self.colormap = self.combo(
            "Colormap", self.viewer.colormaps,
            lambda value: self.change("current_colormap", value))
        self.inspection = self.combo(
            "Inspection", ["nearest", "idw"],
            lambda value: self.change("inspection_method", value))
        self.checks = {}
        for label, attribute in (
            ("Mesh", "show_mesh"), ("Isolines", "show_isolines"),
            ("Isoline labels", "show_isoline_labels"),
            ("Domain edges", "show_domain_edge"), ("Profiles", "show_profile"),
        ):
            widget = QtWidgets.QCheckBox(label)
            self.form.addRow(widget)
            widget.toggled.connect(
                lambda value, name=attribute: self.run_action(lambda: self.change(name, value)))
            self.checks[attribute] = widget
        self.levels = QtWidgets.QSpinBox()
        self.levels.setRange(2, 1000)
        self.form.addRow("Contour levels", self.levels)
        self.levels.valueChanged.connect(
            lambda value: self.run_action(lambda: self.change("contour_levels", value)))
        self.minimum = QtWidgets.QLineEdit()
        self.maximum = QtWidgets.QLineEdit()
        self.form.addRow("Color minimum", self.minimum)
        self.form.addRow("Color maximum", self.maximum)
        self.button("Apply range", self.apply_scale)
        self.button("Auto range", self.automatic_scale)
        self.profile_x = QtWidgets.QLineEdit()
        self.profile_path = QtWidgets.QLineEdit()
        self.profile_path.setPlaceholderText("x,y;x,y;... (blank = vertical)")
        self.form.addRow("Profile x", self.profile_x)
        self.form.addRow("Profile path", self.profile_path)
        self.button("Apply profile", self.apply_profile)
        self.coordinate = QtWidgets.QLineEdit()
        self.coordinate.setPlaceholderText("x,y")
        self.form.addRow("Coordinate", self.coordinate)
        self.button("Inspect", self.inspect_typed)
        self.button("Save image", lambda: self.save_file("image"))
        self.button("Save settings", lambda: self.save_file("settings"))
        self.plotter_button = self.button("Open in plotter", self.open_profile_plotter)
        self.difference_plotter_button = self.button(
            "Plot + diff",
            lambda: self.open_profile_plotter(include_difference=True))
        self.sync_controls()

    def sync_controls(self):
        v = self.viewer
        self.set_combo(self.result, v.result_names, v.current_result)
        locations = [
            {h.location for h in solution.headers if h.name == v.current_result}
            for solution in v.solutions
        ]
        self.set_combo(self.location, ["auto", *sorted(set.intersection(*locations))],
                       v.current_location or "auto")
        steps = sorted(set.intersection(*[
            {h.step for h in solution.headers
             if h.name == v.current_result
             and h.location == field["location"]}
            for solution, field in zip(v.solutions, v.fields)
        ]))
        self.set_combo(self.step, ["latest", *steps], v.current_step_str)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.step.count() - 1)
        self.slider.setValue(self.step.currentIndex())
        self.slider.blockSignals(False)
        result_types = {field["result_type"].lower() for field in v.fields}
        self.set_combo(self.component, VECTOR_COMPONENTS, v.current_component)
        self.set_combo(self.tensor, TENSOR_COMPONENTS, v.current_tensor_component)
        self.component.setEnabled("vector" in result_types)
        self.tensor.setEnabled("matrix" in result_types)
        self.set_combo(self.style, v.styles, v.current_style)
        self.set_combo(self.colormap, v.colormaps, v.current_colormap)
        self.set_combo(self.inspection, ["nearest", "idw"], v.inspection_method)
        for attribute, widget in self.checks.items():
            widget.blockSignals(True)
            widget.setChecked(getattr(v, attribute))
            widget.blockSignals(False)
        self.checks["show_isoline_labels"].setEnabled(v.show_isolines)
        self.levels.blockSignals(True)
        self.levels.setValue(v.contour_levels)
        self.levels.blockSignals(False)
        for widget, value in (
            (self.minimum, v.vmin), (self.maximum, v.vmax),
            (self.profile_x, v.profile_x),
            (self.profile_path, ";".join(f"{x},{y}" for x, y in (v.profile_path or ()))),
        ):
            if not widget.hasFocus():
                widget.setText(str(value))

    def redraw(self):
        limits = (self.viewer.ax0.get_xlim(), self.viewer.ax0.get_ylim())
        self.viewer.update_plot()
        self.viewer.ax0.set_xlim(limits[0])
        self.viewer.ax0.set_ylim(limits[1])
        self.toolbar.update()

    def select_result(self, value):
        self.viewer.current_result = value
        self.viewer.result_idx = self.viewer.result_names.index(value)
        self.viewer.current_location = None
        self.viewer.current_step_str = "latest"
        self.redraw()

    def select_location(self, value):
        self.viewer.current_location = None if value == "auto" else value
        self.viewer.current_step_str = "latest"
        self.redraw()

    def select_style(self, value):
        self.viewer.current_style = value
        self.viewer.style_idx = self.viewer.styles.index(value)
        self.redraw()

    def apply_scale(self):
        minimum, maximum = float(self.minimum.text()), float(self.maximum.text())
        if not all(map(math.isfinite, (minimum, maximum))) or minimum >= maximum:
            raise ValueError("Color range requires finite numbers with minimum < maximum.")
        self.viewer.requested_vmin = self.viewer.vmin = minimum
        self.viewer.requested_vmax = self.viewer.vmax = maximum
        self.redraw()

    def automatic_scale(self):
        self.viewer._reset_scale()
        self.redraw()

    def apply_profile(self):
        x = float(self.profile_x.text())
        path = parse_profile_path(self.profile_path.text())
        if not math.isfinite(x) or (path and any(not math.isfinite(c) for p in path for c in p)):
            raise ValueError("Profile coordinates must be finite.")
        self.viewer.profile_x = x
        self.viewer.profile_path = tuple(path) if path else None
        self.viewer.show_profile = True
        self.redraw()

    def inspect_typed(self):
        coordinates = self.coordinate.text().split(",")
        if len(coordinates) != 2:
            raise ValueError("Enter a coordinate as x,y.")
        x, y = map(float, coordinates)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Inspection coordinates must be finite.")
        self.viewer._show_comparison_popup(x, y)

    def inspect_click(self, event):
        if (hasattr(self, "toolbar") and not self.toolbar.mode
                and event.button == 1 and event.inaxes == self.viewer.ax0
                and event.xdata is not None and event.ydata is not None):
            self.run_action(lambda: self.viewer._show_comparison_popup(event.xdata, event.ydata))

    def closeEvent(self, event):
        self.timer.stop()
        self.report.close()
        self.viewer.canvas.close()
        self.viewer.fig.clear()
        QtWidgets.QMainWindow.closeEvent(self, event)


def render_comparison_gui(*, out_path=None, **options):
    """Launch a comparison window without importing Qt on non-GUI routes."""
    if out_path is not None:
        raise ValueError("The comparison GUI cannot render a static output path.")
    return _render_gui(GidComparisonWindow, options)
