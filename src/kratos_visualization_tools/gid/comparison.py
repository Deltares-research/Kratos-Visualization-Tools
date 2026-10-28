"""Interactive comparison of two GiD solutions."""

from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.widgets import TextBox
from .calculations import build_node_arrays, calculate_shared_scale, calculate_symmetric_scale, interpolate_scalar_field, sample_profile_path_from_triangulation
from .models import SolutionData
from .parsing import detect_profile_nodes
from .drawing import draw_scalar_isolines, draw_triangulation_boundary
from .results import compare_result_names, load_result_values, load_solution_field


class ComparisonViewer:
    """Interactive overlay viewer for two solutions with different meshes."""

    styles = ["background+isolines", "isolines+background", "both-filled", "both-isolines", "difference"]
    colormaps = [
        "viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu",
        "YlOrRd", "coolwarm", "RdBu_r", "seismic", "Spectral", "turbo",
    ]

    def __init__(
        self,
        first: SolutionData,
        second: SolutionData,
        initial_result: str,
        initial_step: str,
        initial_location: Optional[str],
        component: str,
        tensor_component_mode: str,
        profile_x: float,
        style: str,
        vmin: Optional[float],
        vmax: Optional[float],
        inspection_method: str = "nearest",
        show_profile: bool = False,
        show_domain_edge: bool = True,
        colormap: Optional[str] = None,
        contour_levels: int = 12,
        show_isoline_labels: bool = False,
        initial_show_mesh: bool = False,
        initial_show_isolines: bool = True,
        initial_show_isoline_labels: bool = False,
        profile_path: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        self.solutions = (first, second)
        self.common_names, self.first_only, self.second_only = compare_result_names(first.headers, second.headers)
        if not self.common_names:
            raise ValueError("The two solutions have no result variables in common.")
        self.result_names = self.common_names
        self.result_idx = self.result_names.index(initial_result) if initial_result in self.result_names else 0
        self.current_result = self.result_names[self.result_idx]
        self.current_step_str = initial_step
        self.current_location = initial_location
        self.current_location_display = initial_location or "auto"
        self.current_component = component
        self.current_tensor_component = tensor_component_mode
        self.profile_x = profile_x
        self.profile_path = tuple(profile_path) if profile_path else None
        self.style_idx = self.styles.index(style) if style in self.styles else 0
        self.current_style = self.styles[self.style_idx]
        default_colormap = "RdBu_r" if self.current_style == "difference" else "viridis"
        self.current_colormap = colormap or default_colormap
        if self.current_colormap not in self.colormaps:
            raise ValueError(f"Unsupported comparison colormap '{self.current_colormap}'.")
        self.show_mesh = initial_show_mesh
        self.show_isolines = initial_show_isolines
        if inspection_method not in ("nearest", "idw"):
            raise ValueError("Inspection method must be 'nearest' or 'idw'.")
        self.inspection_method = inspection_method
        self.show_profile = show_profile
        self.show_domain_edge = show_domain_edge
        if contour_levels < 2:
            raise ValueError("Contour levels must be at least 2.")
        self.contour_levels = int(contour_levels)
        self.show_isoline_labels = initial_show_isoline_labels or show_isoline_labels
        self.requested_vmin = vmin
        self.requested_vmax = vmax
        self.vmin = vmin
        self.vmax = vmax
        self.fig = None
        self.ax0 = None
        self.ax1 = None
        self.colorbar = None
        self.menu_artists = []
        self.menu_drag_artist = None
        self.menu_drag_offset = None
        self.menu_positions = {}
        self.menu_drag_index = None
        self.fields = None
        self.info_fig = None
        self.coord_input_fig = None
        self.coord_textbox = None
        self.coord_status_artist = None
        self.scale_input_fig = None
        self.scale_textbox = None
        self.scale_status_artist = None
        self.difference_field = None
        self.update_plot()

    def _load_fields(self):
        fields = []
        for solution in self.solutions:
            fields.append(
                load_solution_field(
                    solution,
                    self.current_result,
                    self.current_step_str,
                    self.current_location,
                    self.current_component,
                    self.current_tensor_component,
                )
            )
        self.fields = fields
        self.current_location_display = ", ".join(field["location"] for field in fields)
        interpolated_first = interpolate_scalar_field(fields[0], fields[1]["points"])
        self.difference_field = dict(fields[1])
        self.difference_field["scalar"] = fields[1]["scalar"] - interpolated_first
        if self.requested_vmin is None or self.requested_vmax is None:
            if self.current_style == "difference":
                auto_min, auto_max = calculate_symmetric_scale(self.difference_field["scalar"])
                self.vmin = auto_min if self.requested_vmin is None else self.requested_vmin
                self.vmax = auto_max if self.requested_vmax is None else self.requested_vmax
            else:
                self.vmin, self.vmax = calculate_shared_scale(
                    [field["scalar"] for field in fields], self.requested_vmin, self.requested_vmax
                )

    def _draw_filled(self, field, norm, cmap, alpha):
        scalar = field["scalar"]
        valid = np.isfinite(scalar)
        if not np.any(valid):
            return
        if field["tri"] is not None and np.count_nonzero(valid) >= 3:
            scalar_for_plot = scalar.copy()
            scalar_for_plot[~valid] = float(np.nanmean(scalar[valid]))
            self.ax0.tripcolor(field["tri"], scalar_for_plot, shading="gouraud", cmap=cmap, norm=norm, alpha=alpha)
        else:
            self.ax0.scatter(
                field["points"][valid, 0], field["points"][valid, 1],
                c=scalar[valid], cmap=cmap, norm=norm, s=10, alpha=alpha,
            )

    def _draw_isolines(self, field, norm, color, label):
        scalar = field["scalar"]
        valid = np.isfinite(scalar)
        if field["tri"] is not None and np.count_nonzero(valid) >= 3:
            draw_scalar_isolines(
                self.ax0,
                field["tri"],
                scalar,
                norm.vmin,
                norm.vmax,
                color=color,
                count=self.contour_levels,
                show_labels=self.show_isoline_labels,
            )
        elif np.any(valid):
            self.ax0.scatter(
                field["points"][valid, 0], field["points"][valid, 1],
                facecolors="none", edgecolors=color, s=24, label=label,
            )

    def _draw_profile(self, solution: SolutionData, field, color):
        profile_data = self._get_profile_data(solution, field)
        if profile_data is None:
            return None
        profile_y, profile_values = profile_data
        self.ax1.plot(profile_values, profile_y, "-o", ms=3, color=color, label=solution.label)
        return profile_data

    def _get_profile_data(self, solution: SolutionData, field):
        if self.profile_path and field["tri"] is not None:
            distance, values, _ = sample_profile_path_from_triangulation(
                field["tri"], field["scalar"], self.profile_path
            )
            return distance, values
        node_positions = {node_id: index for index, node_id in enumerate(field["node_ids"])}
        profile_nodes = solution.profile_nodes
        if not profile_nodes:
            profile_nodes = detect_profile_nodes(
                solution.all_nodes, solution.line_meshes, self.profile_x, 0.35
            )
        ids = sorted(node_id for node_id in profile_nodes if node_id in node_positions)
        if not ids:
            return None
        points = np.array([solution.all_nodes[node_id] for node_id in ids], dtype=float)
        values = np.array([field["scalar"][node_positions[node_id]] for node_id in ids], dtype=float)
        valid = np.isfinite(values)
        if not np.any(valid):
            return None
        order = np.argsort(points[valid, 1])
        return points[valid, 1][order], values[valid][order]

    @staticmethod
    def _difference_profile(first_profile, second_profile):
        if first_profile is None or second_profile is None:
            return None
        first_y, first_values = first_profile
        second_y, second_values = second_profile
        lower = max(float(np.min(first_y)), float(np.min(second_y)))
        upper = min(float(np.max(first_y)), float(np.max(second_y)))
        valid = (second_y >= lower) & (second_y <= upper)
        if not np.any(valid):
            return None
        difference_y = second_y[valid]
        interpolated_first = np.interp(difference_y, first_y, first_values)
        return difference_y, second_values[valid] - interpolated_first

    def _reset_scale(self):
        self.requested_vmin = None
        self.requested_vmax = None
        if self.current_style == "difference":
            self.vmin, self.vmax = calculate_symmetric_scale(self.difference_field["scalar"])
        else:
            self.vmin, self.vmax = calculate_shared_scale([field["scalar"] for field in self.fields])

    def _adjust_scale(self, factor):
        center = 0.5 * (self.vmin + self.vmax)
        half_width = 0.5 * (self.vmax - self.vmin) * factor
        self.requested_vmin = center - half_width
        self.requested_vmax = center + half_width
        self.vmin, self.vmax = self.requested_vmin, self.requested_vmax

    def _ensure_comparison_figure(self):
        if self.fig is None:
            if self.show_profile:
                self.fig, (self.ax0, self.ax1, self.ax2) = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
            else:
                self.fig, self.ax0 = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
                self.ax1 = None
                self.ax2 = None
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("pick_event", self._on_menu_pick)
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_menu_motion)
            self.fig.canvas.mpl_connect("button_release_event", self._on_menu_release)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            manager = getattr(self.fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "set_window_title"):
                manager.set_window_title("GiD Solution Comparison")
        else:
            self.ax0.clear()
            if self.ax1 is not None:
                self.ax1.clear()
            if self.ax2 is not None:
                self.ax2.clear()
            if self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None

    def _show_comparison_error(self, exc: Exception):
        self.ax0.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center")
        self.ax0.set_axis_off()
        if self.ax1 is not None:
            self.ax1.set_axis_off()
        if self.ax2 is not None:
            self.ax2.set_axis_off()
        self._draw_menu()
        self.fig.canvas.draw_idle()

    def _draw_comparison_fields(self, norm: Normalize, first_field, second_field):
        plot_cmap = "viridis"
        plot_label = self.current_result
        if self.current_style == "difference":
            self._draw_filled(self.difference_field, norm, self.current_colormap, 0.9)
            plot_cmap = self.current_colormap
            plot_label = f"{self.current_result} ({self.solutions[1].label} - {self.solutions[0].label})"
        elif self.current_style == "background+isolines":
            self._draw_filled(first_field, norm, self.current_colormap, 0.85)
            if self.show_isolines:
                self._draw_isolines(second_field, norm, "black", self.solutions[1].label)
        elif self.current_style == "isolines+background":
            if self.show_isolines:
                self._draw_isolines(first_field, norm, "black", self.solutions[0].label)
            self._draw_filled(second_field, norm, self.current_colormap, 0.65)
        elif self.current_style == "both-filled":
            self._draw_filled(first_field, norm, self.current_colormap, 0.55)
            self._draw_filled(second_field, norm, self.current_colormap, 0.45)
        else:
            if self.show_isolines:
                self._draw_isolines(first_field, norm, "tab:blue", self.solutions[0].label)
                self._draw_isolines(second_field, norm, "tab:red", self.solutions[1].label)
        return plot_cmap, plot_label

    def _draw_comparison_overlays(self, first_field, second_field):
        if self.profile_path:
            path = np.asarray(self.profile_path, dtype=float)
            self.ax0.plot(
                path[:, 0], path[:, 1],
                color="red", linewidth=1.0, alpha=0.8, label="Profile path",
            )
            self.ax0.legend(loc="best")
        if self.show_domain_edge:
            draw_triangulation_boundary(self.ax0, first_field["tri"], color="dimgray", linestyle="-")
            draw_triangulation_boundary(self.ax0, second_field["tri"], color="dimgray", linestyle="--")

        if self.show_mesh:
            for field, color, line_style in (
                (first_field, "tab:blue", "-"),
                (second_field, "tab:red", "--"),
            ):
                if field["tri"] is not None:
                    self.ax0.triplot(
                        field["tri"], color=color, linestyle=line_style,
                        linewidth=0.7, alpha=0.8,
                    )

    def _draw_comparison_profiles(self, first_field, second_field):
        if not (self.show_profile and self.ax1 is not None):
            return
        first_profile = self._get_profile_data(self.solutions[0], first_field)
        second_profile = self._get_profile_data(self.solutions[1], second_field)
        if first_profile is not None:
            self.ax1.plot(first_profile[1], first_profile[0], "-o", ms=3, color="tab:blue", label=self.solutions[0].label)
        if second_profile is not None:
            self.ax1.plot(second_profile[1], second_profile[0], "-o", ms=3, color="tab:red", label=self.solutions[1].label)
        if self.ax1.lines:
            self.ax1.legend(loc="best")
        self.ax1.set_xlabel(self.current_result)
        self.ax1.set_ylabel("distance" if self.profile_path else "y")
        self.ax1.set_title("Profiles")
        self.ax1.grid(True, alpha=0.3)
        if self.ax2 is not None:
            difference_profile = self._difference_profile(first_profile, second_profile)
            if difference_profile is None:
                self.ax2.text(0.5, 0.5, "No overlapping profile data", ha="center", va="center")
                self.ax2.set_axis_off()
            else:
                self.ax2.plot(difference_profile[1], difference_profile[0], "-o", ms=3, color="black")
                self.ax2.axvline(0.0, color="gray", linewidth=0.8)
                self.ax2.set_xlabel(f"{self.solutions[1].label} - {self.solutions[0].label}")
                self.ax2.set_ylabel("y")
                self.ax2.set_title("Profile difference")
                self.ax2.grid(True, alpha=0.3)

    def update_plot(self):
        self._ensure_comparison_figure()

        try:
            self._load_fields()
        except Exception as exc:
            self._show_comparison_error(exc)
            return

        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        first_field, second_field = self.fields
        plot_cmap, plot_label = self._draw_comparison_fields(norm, first_field, second_field)
        self._draw_comparison_overlays(first_field, second_field)
        self._draw_comparison_profiles(first_field, second_field)

        self.ax0.set_aspect("equal", adjustable="box")
        self.ax0.set_xlabel("x")
        self.ax0.set_ylabel("y")
        if self.current_style == "difference":
            self.ax0.set_title(
                f"{plot_label} | target mesh: {self.solutions[1].label} | "
                f"steps {first_field['step']} -> {second_field['step']}"
            )
        else:
            self.ax0.set_title(
                f"{self.current_result} | {self.solutions[0].label}: step {first_field['step']} | "
                f"{self.solutions[1].label}: step {second_field['step']}"
            )
        self.colorbar = self.fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plot_cmap), ax=self.ax0, label=plot_label
        )
        self._draw_menu()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _format_popup_value(result_type: str, arr: Optional[np.ndarray]) -> str:
        if arr is None:
            return "-"
        if arr.size == 0:
            return "<empty>"
        if result_type.lower() == "scalar":
            return f"{float(arr[0]):.6g}"
        return np.array2string(np.asarray(arr, dtype=float), precision=6, separator=", ", suppress_small=False)

    @staticmethod
    def _nearest_node(solution: SolutionData, click_x: float, click_y: float) -> Tuple[int, Tuple[float, float, float], float]:
        node_ids, _, points = build_node_arrays(solution.all_nodes)
        distances = (points[:, 0] - click_x) ** 2 + (points[:, 1] - click_y) ** 2
        position = int(np.argmin(distances))
        return node_ids[position], tuple(points[position]), float(math.sqrt(distances[position]))

    @staticmethod
    def _value_at_coordinate(
        solution: SolutionData,
        values: Dict[int, np.ndarray],
        click_x: float,
        click_y: float,
        method: str,
    ):
        nearest_id, nearest_point, nearest_distance = ComparisonViewer._nearest_node(
            solution, click_x, click_y
        )
        if method == "nearest":
            return nearest_id, nearest_point, nearest_distance, values.get(nearest_id)

        candidates = []
        for node_id, arr in values.items():
            point = solution.all_nodes.get(node_id)
            if point is None or arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            distance_squared = (point[0] - click_x) ** 2 + (point[1] - click_y) ** 2
            candidates.append((distance_squared, node_id, arr))
        if not candidates:
            return nearest_id, nearest_point, nearest_distance, None

        candidates.sort(key=lambda item: item[0])
        distance_squared, _, exact_value = candidates[0]
        if distance_squared < 1e-16:
            return nearest_id, nearest_point, nearest_distance, exact_value.copy()

        selected = candidates[: min(8, len(candidates))]
        shape = selected[0][2].shape
        selected = [item for item in selected if item[2].shape == shape]
        if not selected:
            return nearest_id, nearest_point, nearest_distance, None
        weights = np.array([1.0 / (item[0] + 1e-12) for item in selected], dtype=float)
        weighted_values = np.vstack([item[2] for item in selected])
        return nearest_id, nearest_point, nearest_distance, np.sum(weighted_values * weights[:, None], axis=0) / np.sum(weights)

    def _load_result_for_popup(self, solution: SolutionData, result_name: str):
        headers = [header for header in solution.headers if header.name == result_name]
        if not headers:
            return None

        try:
            requested_step = int(float(self.current_step_str))
        except ValueError:
            requested_step = None

        selected = None
        for location in ("OnNodes", "OnGaussPoints"):
            candidates = [header for header in headers if header.location == location]
            if not candidates:
                continue
            matching = [header for header in candidates if header.step == requested_step]
            selected = matching[-1] if matching else max(candidates, key=lambda header: header.step)
            break

        if selected is None:
            return None
        step, location, result_type, values = load_result_values(
            solution.headers,
            solution.res_path,
            result_name,
            str(selected.step),
            selected.location,
            solution.surface_elements,
            solution.line_meshes,
        )
        return step, location, result_type, values

    def _build_comparison_table(self, click_x: float, click_y: float):
        nearest_nodes = [self._nearest_node(solution, click_x, click_y) for solution in self.solutions]
        result_names = sorted(
            {header.name for solution in self.solutions for header in solution.headers}
        )
        columns = ["Variable"]
        for solution, (node_id, point, distance) in zip(self.solutions, nearest_nodes):
            if self.inspection_method == "idw":
                columns.append(
                    f"{solution.label}\nIDW at ({click_x:.4g}, {click_y:.4g})\n"
                    f"nearest node {node_id}, distance={distance:.4g}"
                )
                continue
            columns.append(
                f"{solution.label}\nnode {node_id} @ ({point[0]:.4g}, {point[1]:.4g})\n"
                f"distance={distance:.4g}"
            )

        rows = []
        for result_name in result_names:
            row = [result_name]
            for solution, (node_id, _, _) in zip(self.solutions, nearest_nodes):
                loaded = self._load_result_for_popup(solution, result_name)
                if loaded is None:
                    row.append("-")
                    continue
                step, location, result_type, values = loaded
                _, _, _, coordinate_value = self._value_at_coordinate(
                    solution, values, click_x, click_y, self.inspection_method
                )
                value = self._format_popup_value(result_type, coordinate_value)
                row.append(f"{value}\n(step {step}, {location})")
            rows.append(row)
        return columns, rows

    def _show_comparison_popup(self, click_x: float, click_y: float):
        columns, rows = self._build_comparison_table(click_x, click_y)
        if self.info_fig is None or not plt.fignum_exists(self.info_fig.number):
            self.info_fig = plt.figure("GiD Solution Comparison - Node Values", figsize=(14, 8))
        else:
            self.info_fig.clf()

        ax = self.info_fig.add_subplot(111)
        ax.set_axis_off()
        table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        self.info_fig.suptitle(
            f"Values near ({click_x:.6g}, {click_y:.6g}) - {self.current_result} comparison"
        )
        self.info_fig.tight_layout()
        self.info_fig.canvas.draw_idle()
        self.info_fig.show()

    def on_click(self, event):
        if self.ax0 is None or event.inaxes != self.ax0:
            return
        if event.xdata is None or event.ydata is None:
            return
        btn = event.button
        is_left = (btn == 1) or (getattr(btn, "name", "") == "LEFT")
        if is_left:
            self._show_comparison_popup(float(event.xdata), float(event.ydata))

    def _open_coordinate_input_popup(self):
        popup_title = "Inspect Coordinate - GiD Solution Comparison"
        if self.coord_input_fig is None or not plt.fignum_exists(self.coord_input_fig.number):
            self.coord_input_fig = plt.figure(popup_title, figsize=(6, 2.3))
        else:
            self.coord_input_fig.clf()

        self.coord_input_fig.text(
            0.02,
            0.86,
            "Type coordinate as: x,y (example: 18.0, -4.2) and press Enter",
            fontsize=9,
        )
        text_ax = self.coord_input_fig.add_axes([0.05, 0.45, 0.9, 0.25])
        self.coord_textbox = TextBox(text_ax, "x,y: ")
        self.coord_status_artist = self.coord_input_fig.text(0.02, 0.1, "", fontsize=9, color="crimson")

        def submit(text: str):
            parts = [part for part in re.split(r"[\s,;]+", text.strip()) if part]
            if len(parts) != 2:
                self.coord_status_artist.set_text("Please enter exactly two numbers: x,y")
                self.coord_input_fig.canvas.draw_idle()
                return
            try:
                x_value = float(parts[0])
                y_value = float(parts[1])
            except ValueError:
                self.coord_status_artist.set_text("Invalid numeric format. Use for example: 18.0, -4.2")
                self.coord_input_fig.canvas.draw_idle()
                return

            self.coord_status_artist.set_text("")
            self.coord_input_fig.canvas.draw_idle()
            self._show_comparison_popup(x_value, y_value)

        self.coord_textbox.on_submit(submit)
        self.coord_textbox.set_active(True)
        self.coord_input_fig.canvas.draw_idle()
        self.coord_input_fig.show()

    def _open_scale_input_popup(self):
        popup_title = "Set Shared Scale - GiD Solution Comparison"
        if self.scale_input_fig is None or not plt.fignum_exists(self.scale_input_fig.number):
            self.scale_input_fig = plt.figure(popup_title, figsize=(6, 2.3))
        else:
            self.scale_input_fig.clf()

        self.scale_input_fig.text(0.02, 0.86, "Type shared limits as: vmin,vmax", fontsize=9)
        text_ax = self.scale_input_fig.add_axes([0.05, 0.45, 0.9, 0.25])
        self.scale_textbox = TextBox(text_ax, "")
        self.scale_status_artist = self.scale_input_fig.text(0.02, 0.1, "", fontsize=9, color="crimson")

        def submit(text: str):
            parts = [part for part in re.split(r"[\s,;]+", text.strip()) if part]
            if len(parts) != 2:
                self.scale_status_artist.set_text("Please enter exactly two numbers: vmin,vmax")
                self.scale_input_fig.canvas.draw_idle()
                return
            try:
                minimum = float(parts[0])
                maximum = float(parts[1])
            except ValueError:
                self.scale_status_artist.set_text("Invalid numeric format. Use for example: -10, 50")
                self.scale_input_fig.canvas.draw_idle()
                return
            if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
                self.scale_status_artist.set_text("vmin and vmax must be finite and vmin < vmax")
                self.scale_input_fig.canvas.draw_idle()
                return

            self.requested_vmin = minimum
            self.requested_vmax = maximum
            self.vmin = minimum
            self.vmax = maximum
            self.scale_status_artist.set_text("")
            self.scale_input_fig.canvas.draw_idle()
            self.update_plot()

        self.scale_textbox.on_submit(submit)
        self.scale_textbox.set_active(True)
        self.scale_input_fig.canvas.draw_idle()
        self.scale_input_fig.show()

    def _draw_menu(self):
        common = textwrap.fill(
            ", ".join(self.common_names) or "-",
            width=32,
            subsequent_indent="  ",
        )
        first_only = textwrap.fill(", ".join(self.first_only) or "-", width=32, subsequent_indent="  ")
        second_only = textwrap.fill(", ".join(self.second_only) or "-", width=32, subsequent_indent="  ")
        for artist in self.menu_artists:
            artist.remove()
        key_menu = (
            "COMPARISON MODE\n"
            "================\n"
            "[n/p] Next/previous common result\n"
            "[j/k] Previous/next requested step\n"
            "[o] Cycle display style\n"
            "[c] Cycle color scale\n"
            "[s] Save current picture\n"
            "[l] Isolines on/off\n"
            "[L] Isoline labels on/off\n"
            "[S] Save visualization settings\n"
            "[m] Toggle meshes (solid first, dashed second)\n"
            "[b] Toggle grey domain boundaries\n"
            "[a] Automatic shared scale\n"
            "[[ and ]] Narrow/widen shared scale\n"
            "[v] Set vmin,vmax\n"
            "[w] Toggle nearest/weighted inspection\n"
            "[i] Inspect typed coordinate\n"
            "[q] Quit"
        )
        state_menu = (
            "CURRENT STATE\n"
            "=============\n"
            f"Result: {self.current_result}\n"
            f"Style: {self.current_style}\n"
            f"Colormap: {self.current_colormap}\n"
            f"Isolines: {'on' if self.show_isolines else 'off'}\n"
            f"Isoline labels: {'on' if self.show_isoline_labels else 'off'}\n"
            "Difference mode: solution 2 - solution 1 on solution 2 mesh\n"
            f"Inspection: {self.inspection_method}\n"
            f"Domain edges: {'on' if self.show_domain_edge else 'off'}\n"
            f"Locations: {self.current_location_display}\n"
            f"Shared scale: [{self.vmin:.6g}, {self.vmax:.6g}]\n"
            f"Common variables: {common}\n"
            f"Only in {self.solutions[0].label}: {first_only}\n"
            f"Only in {self.solutions[1].label}: {second_only}"
        )
        for artist in self.menu_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        positions = [self.menu_positions.get(0, (0.02, 0.98)), self.menu_positions.get(1, (0.52, 0.98))]
        self.menu_artists = [
            self.fig.text(
                *positions[0], key_menu, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), family="monospace", picker=True,
            ),
            self.fig.text(
                *positions[1], state_menu, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), family="monospace", picker=True,
            ),
        ]

    def _on_menu_pick(self, event):
        if event.artist not in self.menu_artists:
            return
        mouse_position = self.fig.transFigure.inverted().transform((event.mouseevent.x, event.mouseevent.y))
        artist_position = event.artist.get_position()
        self.menu_drag_artist = event.artist
        self.menu_drag_index = self.menu_artists.index(event.artist)
        self.menu_drag_offset = (
            artist_position[0] - mouse_position[0],
            artist_position[1] - mouse_position[1],
        )

    def _on_menu_motion(self, event):
        if self.menu_drag_artist is None or event.x is None or event.y is None:
            return
        mouse_position = self.fig.transFigure.inverted().transform((event.x, event.y))
        self.menu_drag_artist.set_position((
            mouse_position[0] + self.menu_drag_offset[0],
            mouse_position[1] + self.menu_drag_offset[1],
        ))
        self.fig.canvas.draw_idle()

    def _on_menu_release(self, event):
        if self.menu_drag_artist is not None and self.menu_drag_index is not None:
            self.menu_positions[self.menu_drag_index] = self.menu_drag_artist.get_position()
        self.menu_drag_artist = None
        self.menu_drag_offset = None
        self.menu_drag_index = None

    def save_current_plot(self, out_path: Optional[Path] = None) -> Path:
        if out_path is None:
            step = str(self.current_step_str).replace(".", "_")
            result = self.current_result.replace(" ", "_")
            out_path = Path(f"comparison_{result}_{step}_{self.current_style}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_path, dpi=180)
        print(f"Current comparison plot saved to: {out_path.resolve()}")
        return out_path

    def save_current_settings(self, out_path: Optional[Path] = None) -> Path:
        if out_path is None:
            out_path = Path(f"comparison_{self.current_result}_settings.json")
        settings = {
            "mode": "comparison",
            "solutions": [
                {"label": solution.label, "mesh": str(solution.msh_path), "results": str(solution.res_path)}
                for solution in self.solutions
            ],
            "result": self.current_result,
            "step": self.current_step_str,
            "location": self.current_location,
            "component": self.current_component,
            "tensor_component": self.current_tensor_component,
            "profile_x": self.profile_x,
            "profile_path": [list(point) for point in self.profile_path] if self.profile_path else None,
            "style": self.current_style,
            "inspection_method": self.inspection_method,
            "colormap": self.current_colormap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_mesh": self.show_mesh,
            "show_isolines": self.show_isolines,
            "show_domain_edge": self.show_domain_edge,
            "show_isoline_labels": self.show_isoline_labels,
            "show_profile": self.show_profile,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"Comparison settings saved to: {out_path.resolve()}")
        return out_path

    def on_key(self, event):
        if event.key == "q":
            plt.close(self.fig)
        elif event.key == "n":
            self.result_idx = (self.result_idx + 1) % len(self.result_names)
            self.current_result = self.result_names[self.result_idx]
            self.current_step_str = "latest"
            self.current_location = None
            self.update_plot()
        elif event.key == "p":
            self.result_idx = (self.result_idx - 1) % len(self.result_names)
            self.current_result = self.result_names[self.result_idx]
            self.current_step_str = "latest"
            self.current_location = None
            self.update_plot()
        elif event.key == "o":
            self.style_idx = (self.style_idx + 1) % len(self.styles)
            self.current_style = self.styles[self.style_idx]
            self.update_plot()
        elif event.key == "c":
            colormap_idx = self.colormaps.index(self.current_colormap)
            self.current_colormap = self.colormaps[(colormap_idx + 1) % len(self.colormaps)]
            self.update_plot()
        elif event.key == "s":
            self.save_current_plot()
        elif event.key == "l":
            self.show_isolines = not self.show_isolines
            self.update_plot()
        elif event.key in ("L", "shift+l"):
            self.show_isoline_labels = not self.show_isoline_labels
            self.update_plot()
        elif event.key in ("S", "shift+s"):
            self.save_current_settings()
        elif event.key == "m":
            self.show_mesh = not self.show_mesh
            self.update_plot()
        elif event.key == "b":
            self.show_domain_edge = not self.show_domain_edge
            self.update_plot()
        elif event.key == "a" and self.fields:
            self._reset_scale()
            self.update_plot()
        elif event.key == "i":
            self._open_coordinate_input_popup()
        elif event.key == "v":
            self._open_scale_input_popup()
        elif event.key == "w":
            self.inspection_method = "idw" if self.inspection_method == "nearest" else "nearest"
            self.update_plot()
        elif event.key == "[" and self.fields:
            self._adjust_scale(0.8)
            self.update_plot()
        elif event.key == "]" and self.fields:
            self._adjust_scale(1.25)
            self.update_plot()
        elif event.key in ("j", "k"):
            try:
                current_step = int(float(self.current_step_str))
            except ValueError:
                current_step = 0
            self.current_step_str = str(current_step - 1 if event.key == "j" else current_step + 1)
            self.update_plot()
