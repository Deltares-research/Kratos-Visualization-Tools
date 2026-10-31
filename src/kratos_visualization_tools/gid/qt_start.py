"""Browse-first Qt entry point for the GiD results viewer."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .comparison import ComparisonViewer
from .parsing import detect_profile_nodes
from .qt_comparison import render_comparison_gui
from .qt_viewer import render_plot_gui
from .results import discover_solution_paths, load_solution_data


class GidStartWindow(QtWidgets.QMainWindow):
    """Choose one solution, a solution comparison, or a results folder."""

    def __init__(self, defaults=None):
        super().__init__()
        self.defaults = defaults or {}
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("GiD Results Viewer")
        self.setMinimumWidth(440)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        title = QtWidgets.QLabel("What would you like to view?")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 12px;")
        layout.addWidget(title)
        self.single_button = self._add_action(
            layout, "View one solution", "Choose a GiD mesh or results file.",
            self.open_single)
        self.compare_button = self._add_action(
            layout, "Compare two solutions", "Choose the first and second solution files.",
            self.open_comparison)
        self.folder_button = self._add_action(
            layout, "Browse a results folder", "Find matching mesh/result pairs in a folder.",
            self.open_folder)
        layout.addStretch(1)
        note = QtWidgets.QLabel("A matching .post.msh/.post.res or .msh/.res pair is required.")
        note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note)

    @staticmethod
    def _add_action(layout, title, description, callback):
        button = QtWidgets.QPushButton(title)
        button.setMinimumHeight(52)
        button.clicked.connect(callback)
        layout.addWidget(button)
        note = QtWidgets.QLabel(description)
        note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note)
        return button

    def _choose_solution(self, title):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            title,
            str(Path.cwd()),
            "GiD mesh/results (*.post.msh *.post.res *.msh *.res);;"
            "GiD mesh (*.post.msh *.msh);;GiD results (*.post.res *.res);;"
            "All files (*)",
        )
        return Path(filename) if filename else None

    def _show_error(self, error):
        QtWidgets.QMessageBox.warning(self, "GiD Results Viewer", str(error))

    def _solution(self, path, label):
        solution = load_solution_data(str(path), label)
        profile_x = self.defaults.get("profile_x", 18.0)
        x_tol = self.defaults.get("x_tol", 0.35)
        solution.profile_nodes = detect_profile_nodes(
            solution.all_nodes, solution.line_meshes, profile_x, x_tol)
        return solution

    def _initial_result(self, headers):
        names = sorted({header.name for header in headers})
        if not names:
            raise RuntimeError("No parsable GiD result blocks were found.")
        requested = self.defaults.get("result")
        return requested if requested in names else (
            "WATER_PRESSURE" if "WATER_PRESSURE" in names else names[0])

    def _open_single_viewer(self, solution, solution_paths=None):
        valid_colormaps = {
            "viridis", "plasma", "inferno", "magma", "coolwarm", "RdYlBu",
            "seismic", "hsv", "jet", "turbo",
        }
        colormap = self.defaults.get("colormap")
        if colormap not in valid_colormaps:
            colormap = None
        renderer_options = {
            "headers": solution.headers,
            "res_path": solution.res_path,
            "all_nodes": solution.all_nodes,
            "triangles": solution.triangles,
            "surface_elements": solution.surface_elements,
            "line_meshes": solution.line_meshes,
            "profile_nodes": solution.profile_nodes,
            "initial_result": self._initial_result(solution.headers),
            "initial_step": self.defaults.get("step", "latest"),
            "initial_location": self.defaults.get("location"),
            "profile_x": self.defaults.get("profile_x", 18.0),
            "x_tol": self.defaults.get("x_tol", 0.35),
            "initial_tensor_component": self.defaults.get("tensor_component", "trace"),
            "active_filter": None,
            "show_isolines": self.defaults.get("isolines", False),
            "show_profile": self.defaults.get("profile", False),
            "contour_levels": self.defaults.get("contour_levels", 12),
            "show_isoline_labels": self.defaults.get("isoline_labels", False),
            "initial_component": self.defaults.get("component", "magnitude"),
            "initial_show_gradient": self.defaults.get("show_gradient", False),
            "initial_show_mesh": self.defaults.get("show_mesh", False),
            "initial_colormap": colormap,
            "profile_path": self.defaults.get("profile_path"),
            "solution_paths": solution_paths,
            "initial_show_deformation": self.defaults.get("show_deformation", False),
            "initial_deformation_scale": self.defaults.get("deformation_scale", 1.0),
        }
        return render_plot_gui(**renderer_options)

    def open_single(self):
        path = self._choose_solution("Choose a GiD solution")
        if path is None:
            return
        try:
            if self.defaults.get("show_gradient") and self.defaults.get("show_deformation"):
                raise ValueError(
                    "Choose either --gradient or --deformation when opening one solution.")
            solution = self._solution(path, path.stem)
            self._open_single_viewer(solution)
        except (OSError, ValueError, RuntimeError) as error:
            self._show_error(error)

    def open_comparison(self):
        first_path = self._choose_solution("Choose the first GiD solution")
        if first_path is None:
            return
        second_path = self._choose_solution("Choose the second GiD solution")
        if second_path is None:
            return
        try:
            first = self._solution(first_path, first_path.stem)
            second = self._solution(second_path, second_path.stem)
            common = sorted({header.name for header in first.headers}
                            & {header.name for header in second.headers})
            if not common:
                raise RuntimeError("The two solutions have no result variables in common.")
            requested = self.defaults.get("result")
            result = requested if requested in common else (
                "WATER_PRESSURE" if "WATER_PRESSURE" in common else common[0])
            colormap = self.defaults.get("colormap")
            if colormap not in ComparisonViewer.colormaps:
                colormap = None
            options = {
                "first": first,
                "second": second,
                "initial_result": result,
                "initial_step": self.defaults.get("step", "latest"),
                "initial_location": self.defaults.get("location"),
                "component": self.defaults.get("component", "magnitude"),
                "tensor_component_mode": self.defaults.get("tensor_component", "trace"),
                "profile_x": self.defaults.get("profile_x", 18.0),
                "style": self.defaults.get("comparison_style", "background+isolines"),
                "vmin": None,
                "vmax": None,
                "show_profile": self.defaults.get("profile", False),
                "show_domain_edge": True,
                "colormap": colormap,
                "contour_levels": self.defaults.get("contour_levels", 12),
                "show_isoline_labels": self.defaults.get("isoline_labels", False),
                "initial_show_mesh": self.defaults.get("show_mesh", False),
                "initial_show_isolines": self.defaults.get("isolines", True),
                "profile_path": self.defaults.get("profile_path"),
            }
            render_comparison_gui(**options)
        except (OSError, ValueError, RuntimeError) as error:
            self._show_error(error)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a folder containing GiD solutions", str(Path.cwd()))
        if not folder:
            return
        try:
            if self.defaults.get("show_gradient") and self.defaults.get("show_deformation"):
                raise ValueError(
                    "Choose either --gradient or --deformation when opening one solution.")
            pairs = discover_solution_paths(Path(folder))
            if not pairs:
                raise RuntimeError(
                    f"No matching .post.msh/.post.res or .msh/.res pairs found in {folder}")
            solution = self._solution(pairs[0][0], pairs[0][0].stem)
            self._open_single_viewer(solution, solution_paths=pairs)
        except (OSError, ValueError, RuntimeError) as error:
            self._show_error(error)


def render_gid_start_gui(defaults=None):
    """Show the workflow chooser and keep it available alongside viewer windows."""
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])
    window = GidStartWindow(defaults)
    window.show()
    windows = getattr(app, "_gid_windows", None)
    if windows is None:
        windows = app._gid_windows = []
    windows.append(window)
    window.destroyed.connect(lambda: windows.remove(window) if window in windows else None)
    if owns_app:
        app.exec()
    return window
