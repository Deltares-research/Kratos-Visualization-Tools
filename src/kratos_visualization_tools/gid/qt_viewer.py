"""Optional Qt single-solution shell around the shared GiD rendering engine.

Import this module only when Qt is explicitly requested. Manual color ranges
remain comparison-only; the single-solution engine retains automatic scaling.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .config import parse_profile_path
from .models import TENSOR_COMPONENTS, VECTOR_COMPONENTS
from .parsing import detect_profile_nodes, scan_result_headers
from .viewer import InteractiveViewer
from .profile_bundle import save_profile_bundle


class QtInteractiveViewer(InteractiveViewer):
    """Use shared calculations and exports with a Qt-owned, non-pyplot figure."""

    def __init__(self, window, **options):
        self.window = window
        self._profile_layout = None
        super().__init__(**options)

    def _ensure_plot_figure(self):
        if self.fig is None:
            self.fig = Figure(figsize=(12, 8), layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
            self.canvas.mpl_connect("button_press_event", self.window.inspect_click)
            self.canvas.mpl_connect("button_release_event", lambda event: self._store_current_view())
        else:
            self._clear_click_markers()
            for bar in self.colorbars:
                bar.remove()
            self.colorbars = []
        if self._profile_layout != self.show_profile:
            self.fig.clear()
            if self.show_profile:
                self.ax0, self.ax1 = self.fig.subplots(1, 2)
            else:
                self.ax0 = self.fig.subplots()
                self.ax1 = None
            self._profile_layout = self.show_profile
        else:
            self.ax0.clear()
            if self.ax1 is not None:
                self.ax1.clear()
        self.profile_y_data = self.profile_values = self.profile_gradients = None
        self.current_field_node_ids = self.current_field_xyz = None
        self.current_field_source_values = self.current_field_display_values = None
        self.current_field_grad_x = self.current_field_grad_y = self.current_field_grad_magnitude = None

    def _load_plot_state(self):
        # Unlike the legacy shell, propagate errors to the window's action handler.
        step, result_type, values = self._load_plot_data()
        if not values:
            raise ValueError("No values are available for this result, step and location.")
        return step, result_type, values, self._compute_plot_arrays(result_type, values)

    def draw_menu(self):
        pass

    def get_available_steps(self):
        return sorted({h.step for h in self.headers
                       if h.name == self.current_result and h.location == self.current_location})

    def get_current_step(self):
        if self.current_result not in self.result_names:
            raise ValueError(f"No result '{self.current_result}' is available.")
        return super().get_current_step()

    def _build_animation_frames(self):
        frames = super()._build_animation_frames()
        headers_by_solution = {self.solution_index: self.headers}
        for index, (_, path) in enumerate(self.solution_paths):
            if index not in headers_by_solution:
                headers_by_solution[index] = scan_result_headers(path)
        return [
            frame for frame in frames
            if any(h.name == frame[1] and str(h.step) == frame[2]
                   and h.location == self.current_location
                   for h in headers_by_solution[frame[0]])
        ]

    def _advance_animation_frame(self, frame_index):
        location = self.current_location
        solution_index, result_name, step = self.animation_frames[frame_index]
        if self.solution_paths and solution_index != self.solution_index:
            self._switch_solution(solution_index - self.solution_index)
        self.current_result = result_name
        self.result_idx = self.result_names.index(result_name)
        self.current_step_str = step
        self.current_location = location
        self.update_plot()

    def _show_info_popup(self, text):
        self.window.report.setPlainText(text)
        self.window.report.show()


class GidMainWindow(QtWidgets.QMainWindow):
    """Widget-based controls for one solution or a folder of solution pairs."""

    _render_state_names = (
        "fig", "ax0", "ax1", "colorbars", "_profile_layout", "_click_markers",
    )

    def __init__(self, **options):
        super().__init__()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1250, 850)
        self.report = QtWidgets.QPlainTextEdit(self)
        self.report.setReadOnly(True)
        self.report.setWindowFlag(QtCore.Qt.WindowType.Window)
        self.report.setWindowTitle("Coordinate inspection")
        self.report.resize(750, 550)
        self.report.hide()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(lambda: self.run_action(self.advance_frame))
        self.frame_index = 0
        self.viewer = QtInteractiveViewer(self, **options)

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

        self.solution = self.combo("Solution", [], self.select_solution)
        self.solution.setVisible(bool(self.viewer.solution_paths))
        self.result = self.combo("Result", [], self.select_result)
        self.location = self.combo("Location", [], self.select_location)
        self.step = self.combo("Step", [], self.select_step)
        self.component = self.combo("Vector component", VECTOR_COMPONENTS,
                                    lambda value: self.change("current_component", value))
        self.tensor = self.combo("Tensor reduction", TENSOR_COMPONENTS,
                                 lambda value: self.change("current_tensor_component", value))
        self.checks = {}
        for label, attribute in (
            ("Mesh", "show_mesh"), ("Gradient", "show_gradient"),
            ("Isolines", "show_isolines"), ("Isoline labels", "show_isoline_labels"),
            ("Profile", "show_profile"), ("Deformation", "show_deformation"),
        ):
            checkbox = QtWidgets.QCheckBox(label)
            self.form.addRow(checkbox)
            checkbox.toggled.connect(
                lambda value, name=attribute: self.run_action(lambda: self.change(name, value)))
            self.checks[attribute] = checkbox
        self.levels = QtWidgets.QSpinBox()
        self.levels.setRange(2, 1000)
        self.form.addRow("Contour levels", self.levels)
        self.levels.valueChanged.connect(
            lambda value: self.run_action(lambda: self.change("contour_levels", value)))
        self.colormap = self.combo("Colormap", self.viewer.colormaps,
                                   lambda value: self.change("current_colormap", value))
        self.profile_x = QtWidgets.QLineEdit()
        self.profile_path = QtWidgets.QLineEdit()
        self.profile_path.setPlaceholderText("x,y;x,y;… (blank = vertical)")
        self.form.addRow("Profile x", self.profile_x)
        self.form.addRow("Profile path", self.profile_path)
        self.button("Apply profile", self.apply_profile)
        self.scale = QtWidgets.QDoubleSpinBox()
        self.scale.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.scale.setRange(0.000001, 1e12)
        self.scale.setDecimals(6)
        self.form.addRow("Deformation scale", self.scale)
        self.scale.valueChanged.connect(
            lambda value: self.run_action(lambda: self.change("deformation_scale", value)))
        self.button("Previous", lambda: self.move_step(-1))
        self.play = self.button("Play / pause", self.toggle_play)
        self.button("Next", lambda: self.move_step(1))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.form.addRow(self.slider)
        self.slider.valueChanged.connect(
            lambda index: self.run_action(lambda: self.select_step(self.step.itemText(index))))
        self.button("Save image…", lambda: self.save_file("image"))
        self.button("Save settings…", lambda: self.save_file("settings"))
        self.profile_export = self.button("Profile CSV…", lambda: self.save_file("profile"))
        self.gradient_export = self.button("Gradient CSV…", lambda: self.save_file("gradient"))
        self.plotter_button = self.button("Open in plotter", self.open_profile_plotter)
        self.coordinate = QtWidgets.QLineEdit()
        self.coordinate.setPlaceholderText("x,y")
        self.form.addRow("Coordinate", self.coordinate)
        self.button("Inspect", self.inspect_typed)
        self.coordinate.returnPressed.connect(lambda: self.run_action(self.inspect_typed))
        self.sync_controls()

    def combo(self, label, values, callback):
        widget = QtWidgets.QComboBox()
        widget.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(12)
        widget.addItems(list(values))
        widget.currentTextChanged.connect(lambda value: self.run_action(lambda: callback(value)))
        self.form.addRow(label, widget)
        return widget

    def button(self, label, callback):
        widget = QtWidgets.QPushButton(label)
        widget.clicked.connect(lambda: self.run_action(callback))
        self.form.addRow(widget)
        return widget

    def run_action(self, action):
        previous_state = self.viewer.__dict__.copy()
        try:
            action()
            self.sync_controls()
        except (OSError, ValueError, RuntimeError) as exc:
            self.timer.stop()
            # Axes/colorbars may already have been replaced before rendering failed.
            for name in self._render_state_names:
                previous_state[name] = getattr(self.viewer, name)
            self.viewer.__dict__.update(previous_state)
            message = str(exc)
            try:
                self.redraw()
            except (OSError, ValueError, RuntimeError) as recovery_error:
                message += f"\nCould not redraw the previous selection: {recovery_error}"
            self.sync_controls()
            self.statusBar().showMessage(message)
            QtWidgets.QMessageBox.warning(self, "GiD viewer", message)

    @staticmethod
    def set_combo(widget, values, selected):
        widget.blockSignals(True)
        widget.clear()
        widget.addItems([str(value) for value in values])
        widget.setCurrentText(str(selected))
        widget.blockSignals(False)

    def sync_controls(self):
        v = self.viewer
        current_step = v.get_current_step()
        self.setWindowTitle(f"GiD Viewer — {v.open_file_name}")
        self.set_combo(self.solution, [str(pair[1]) for pair in v.solution_paths], str(v.res_path))
        self.set_combo(self.result, v.result_names, v.current_result)
        headers = [h for h in v.headers if h.name == v.current_result]
        self.set_combo(self.location, sorted({h.location for h in headers}), v.current_location)
        steps = sorted({h.step for h in headers if h.location == v.current_location})
        self.set_combo(self.step, steps, current_step)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(steps) - 1))
        self.slider.setValue(steps.index(current_step) if current_step in steps else 0)
        self.slider.blockSignals(False)
        result_type = next(h.result_type.lower() for h in headers if h.location == v.current_location)
        self.set_combo(self.component, VECTOR_COMPONENTS, v.current_component)
        self.set_combo(self.tensor, TENSOR_COMPONENTS, v.current_tensor_component)
        self.component.setEnabled(result_type == "vector")
        self.tensor.setEnabled(result_type == "matrix")
        deformation = result_type == "vector" and v._is_displacement_result()
        self.checks["show_deformation"].setEnabled(deformation)
        self.scale.setEnabled(deformation)
        self.checks["show_gradient"].setEnabled(v.tri is not None)
        self.checks["show_isoline_labels"].setEnabled(v.show_isolines)
        for attribute, widget in self.checks.items():
            widget.blockSignals(True)
            widget.setChecked(getattr(v, attribute))
            widget.blockSignals(False)
        for widget, value in ((self.scale, v.deformation_scale), (self.levels, v.contour_levels)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.set_combo(self.colormap, v.colormaps, v.current_colormap)
        if not self.profile_x.hasFocus():
            self.profile_x.setText(str(v.profile_x))
        if not self.profile_path.hasFocus():
            self.profile_path.setText(";".join(f"{x},{y}" for x, y in (v.profile_path or ())))
        self.profile_export.setEnabled(v.show_profile and v.profile_values is not None)
        self.plotter_button.setEnabled(v.show_profile and v.profile_values is not None)
        self.gradient_export.setEnabled(v.show_gradient and v.current_field_grad_x is not None)
        self.play.setText("Pause" if self.timer.isActive() else "Play")

    def redraw(self):
        if self.viewer.ax0 is not None:
            self.viewer._store_current_view()
        self.viewer.update_plot()
        self.toolbar.update()

    def change(self, attribute, value):
        setattr(self.viewer, attribute, value)
        self.redraw()

    def select_result(self, value):
        v = self.viewer
        self.timer.stop()
        v._jump_to_result(v.result_names.index(value) - v.result_idx)

    def select_solution(self, value):
        self.timer.stop()
        index = [str(pair[1]) for pair in self.viewer.solution_paths].index(value)
        self.viewer._switch_solution(index - self.viewer.solution_index)

    def select_location(self, value):
        self.timer.stop()
        self.viewer.current_location = value
        self.viewer.current_step_str = "latest"
        self.redraw()

    def select_step(self, value):
        self.timer.stop()
        self.viewer.current_step_str = value
        self.redraw()

    def move_step(self, offset):
        self.timer.stop()
        index = (self.step.currentIndex() + offset) % self.step.count()
        self.select_step(self.step.itemText(index))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.viewer.animation_frames = self.viewer._build_animation_frames()
            if len(self.viewer.animation_frames) > 1:
                current = (self.viewer.solution_index, self.viewer.current_result,
                           str(self.viewer.get_current_step()))
                self.frame_index = (self.viewer.animation_frames.index(current)
                                    if current in self.viewer.animation_frames else -1)
                self.timer.start()

    def advance_frame(self):
        self.frame_index = (self.frame_index + 1) % len(self.viewer.animation_frames)
        self.viewer._advance_animation_frame(self.frame_index)

    def apply_profile(self):
        x = float(self.profile_x.text())
        path = parse_profile_path(self.profile_path.text())
        if not math.isfinite(x) or (path and any(not math.isfinite(c) for p in path for c in p)):
            raise ValueError("Profile coordinates must be finite.")
        self.viewer.profile_x = x
        self.viewer.profile_path = tuple(path) if path else None
        self.viewer.profile_nodes = detect_profile_nodes(
            self.viewer.all_nodes, self.viewer.line_meshes, x, self.viewer.x_tol)
        self.viewer.show_profile = True
        self.redraw()

    def inspect_typed(self):
        coordinates = self.coordinate.text().split(",")
        if len(coordinates) != 2:
            raise ValueError("Enter a coordinate as x,y.")
        x, y = map(float, coordinates)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Inspection coordinates must be finite.")
        self.viewer._inspect_coordinate(x, y, source="Qt")

    def inspect_click(self, event):
        if (hasattr(self, "toolbar") and not self.toolbar.mode
                and event.button == 1 and event.inaxes == self.viewer.ax0
                and event.xdata is not None and event.ydata is not None):
            self.run_action(lambda: self.viewer._inspect_coordinate(event.xdata, event.ydata))

    def save_file(self, kind):
        suffix, filters = {
            "image": (".png", "Image (*.png *.pdf *.svg)"),
            "settings": (".json", "Settings (*.json)"),
            "profile": (".csv", "CSV (*.csv)"),
            "gradient": (".csv", "CSV (*.csv)"),
        }[kind]
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Save {kind}", f"{self.viewer.current_result}_{kind}{suffix}", filters)
        if not filename:
            return
        path = Path(filename)
        if kind == "image":
            saved = self.viewer.save_current_plot(path)
        elif kind == "settings":
            saved = self.viewer.save_current_settings(path)
        elif kind == "profile":
            saved = self.viewer.export_profile_data(path, strict=True)
        else:
            saved = self.viewer.export_current_field_data(path, strict=True)
        self.statusBar().showMessage(f"Saved {kind}: {saved}", 10000)

    def open_profile_plotter(self, *, include_difference=False):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose where to save the profile bundle")
        if not folder:
            return
        config = save_profile_bundle(
            self.viewer, folder, include_difference=include_difference)
        self.statusBar().showMessage(f"Saved profile bundle: {config.parent}")
        self.launch_profile_plotter(config)

    def launch_profile_plotter(self, config):
        # Tk/Agg backend changes in the plotter must not affect the Qt GiD viewer.
        process = QtCore.QProcess()
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        package_root = str(Path(__file__).resolve().parents[2])
        existing = environment.value("PYTHONPATH")
        environment.insert("PYTHONPATH", package_root + (";" + existing if existing else ""))
        process.setProcessEnvironment(environment)
        process.setProgram(sys.executable)
        process.setArguments([
            "-m", "kratos_visualization_tools.saved_results", str(config), "--show"])
        processes = getattr(self, "_plotter_processes", None)
        if processes is None:
            processes = self._plotter_processes = []
        processes.append(process)

        def report_failure(message):
            self.statusBar().showMessage(message)
            QtWidgets.QMessageBox.warning(self, "Saved-results plotter", message)

        def finished(code, status):
            if code != 0 or status == QtCore.QProcess.ExitStatus.CrashExit:
                details = bytes(process.readAllStandardError()).decode(errors="replace")
                details += bytes(process.readAllStandardOutput()).decode(errors="replace")
                report_failure(f"Plotter failed (exit {code}).\n{details}")
            processes.remove(process)
            process.deleteLater()

        process.finished.connect(finished)
        def failed(error):
            if error == QtCore.QProcess.ProcessError.FailedToStart:
                report_failure(f"Could not run plotter: {process.errorString()}")
                processes.remove(process)
                process.deleteLater()

        process.errorOccurred.connect(failed)
        process.start()

    def closeEvent(self, event):
        self.timer.stop()
        self.viewer._stop_animation()
        self.report.close()
        self.viewer.canvas.close()
        self.viewer.fig.clear()
        super().closeEvent(event)


def render_plot_gui(**options):
    """Launch the optional Qt window; reuse an existing application's event loop."""
    return _render_gui(GidMainWindow, options)


def _render_gui(window_type, options):
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])
    window = window_type(**options)
    window.show()
    # Keep the window alive when an embedding caller owns the event loop.
    windows = getattr(app, "_gid_windows", None)
    if windows is None:
        windows = app._gid_windows = []
    windows.append(window)
    window.destroyed.connect(lambda: windows.remove(window))
    if owns_app:
        app.exec()
    return window
