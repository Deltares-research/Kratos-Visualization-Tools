"""Interactive single-solution and multi-stage GiD viewer."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from matplotlib import animation
from matplotlib.widgets import TextBox
from .calculations import build_node_arrays, compute_nodal_gradient, field_scalar_value, finite_difference_gradient, reduce_result_value, sample_profile_path_from_triangulation, sample_vertical_profile_from_triangulation, tensor_component, triangulation_from_connectivity, vector_component
from .exports import build_output_path, build_result_export_path
from .models import ActiveSubmodelPart, RESULT_LOCATION_ON_GAUSS_POINTS, ResultHeader, TENSOR_COMPONENTS, VECTOR_COMPONENTS, ViewState
from .parsing import detect_profile_nodes, load_values, parse_post_msh, scan_result_headers
from .drawing import draw_scalar_isolines
from .results import load_result_values, pick_step


class InteractiveViewer:
    """Interactive matplotlib viewer for result browsing."""

    def __init__(
        self,
        headers: Sequence[ResultHeader],
        res_path: Path,
        all_nodes: Dict[int, Tuple[float, float, float]],
        triangles: Sequence[Tuple[int, int, int]],
        surface_elements: Sequence[Tuple[int, List[int]]],
        line_meshes: Sequence[Tuple[str, set, list]],
        profile_nodes: set,
        initial_result: str,
        initial_step: str,
        initial_location: Optional[str],
        profile_x: float,
        x_tol: float,
        initial_tensor_component: str,
        active_filter: Optional[ActiveSubmodelPart],
        show_isolines: bool = False,
        show_profile: bool = False,
        contour_levels: int = 12,
        show_isoline_labels: bool = False,
        initial_component: str = "magnitude",
        initial_show_gradient: bool = False,
        initial_show_mesh: bool = False,
        initial_colormap: Optional[str] = None,
        profile_path: Optional[Sequence[Tuple[float, float]]] = None,
        solution_paths: Optional[Sequence[Tuple[Path, Path]]] = None,
        initial_solution_index: int = 0,
        initial_show_deformation: bool = False,
        initial_deformation_scale: float = 1.0,
    ):
        self.headers = headers
        self.res_path = res_path
        self.solution_paths = tuple(solution_paths or ())
        self.solution_index = int(initial_solution_index) if self.solution_paths else 0
        self.open_file_name = self.res_path.name
        self.all_nodes = all_nodes
        self.triangles = triangles
        self.surface_elements = surface_elements
        self.line_meshes = line_meshes
        self.profile_nodes = profile_nodes
        self.active_filter = active_filter

        self.node_ids, self.id_to_pos, self.pts = build_node_arrays(all_nodes)
        self.xy = self.pts[:, :2]
        self.tri = triangulation_from_connectivity(triangles, self.id_to_pos, self.xy)

        # Get unique result names
        self.result_names = sorted(set(h.name for h in headers))
        self.result_idx = self.result_names.index(initial_result) if initial_result in self.result_names else 0

        # State
        self.current_result = self.result_names[self.result_idx]
        self.current_location = initial_location
        self.current_component = initial_component
        self.current_tensor_component = initial_tensor_component
        self.show_gradient = initial_show_gradient
        self.current_step_str = initial_step
        self.profile_x = float(profile_x)
        self.profile_path = tuple(profile_path) if profile_path else None
        self.x_tol = float(x_tol)
        self.show_mesh = initial_show_mesh
        self.show_deformation = initial_show_deformation and self._is_displacement_result()
        self.deformation_scale = float(initial_deformation_scale)
        if not math.isfinite(self.deformation_scale) or self.deformation_scale <= 0.0:
            raise ValueError("Deformation magnification must be a positive finite number.")
        self.show_isolines = show_isolines
        self.show_profile = show_profile
        self.show_isoline_labels = show_isoline_labels
        self.state = ViewState(
            result=self.current_result,
            component=self.current_component,
            tensor_component=self.current_tensor_component,
            show_gradient=self.show_gradient,
            show_mesh=self.show_mesh,
            show_isolines=self.show_isolines,
            show_isoline_labels=self.show_isoline_labels,
            show_profile=self.show_profile,
        )
        if contour_levels < 2:
            raise ValueError("Contour levels must be at least 2.")
        self.contour_levels = int(contour_levels)

        self._reset_view_bounds()
        self._pan_active = False
        self._pan_start_data: Optional[Tuple[float, float]] = None
        self._pan_start_xlim: Optional[Tuple[float, float]] = None
        self._pan_start_ylim: Optional[Tuple[float, float]] = None

        # Colormap cycling
        self.colormaps = ["viridis", "plasma", "inferno", "magma", "coolwarm", "RdYlBu", "seismic", "hsv", "jet", "turbo"]
        self.current_colormap = initial_colormap or self.colormaps[0]
        if self.current_colormap not in self.colormaps:
            raise ValueError(f"Unsupported single-solution colormap '{self.current_colormap}'.")
        self.colormap_idx = self.colormaps.index(self.current_colormap)

        # Compute Gauss point locations indexed by element ID
        self.gauss_points_by_id = {}  # Dict: element_id -> (x, y)
        
        # Add surface element centroids keyed by real element IDs.
        for elem_id, nodes in surface_elements:
            if len(nodes) in (3, 6, 10):
                corner_nodes = nodes[:3]
            elif len(nodes) in (4, 8, 9):
                corner_nodes = nodes[:4]
            else:
                corner_nodes = nodes

            pts = [all_nodes[nid] for nid in corner_nodes if nid in all_nodes]
            if not pts:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            self.gauss_points_by_id[elem_id] = (cx, cy)
        
        # Add line element centroids (from conditions with their actual element IDs)
        for _, _, line_elems in line_meshes:
            for elem_id, nodes, _ in line_elems:
                if len(nodes) >= 2:
                    pts = [all_nodes[nid] for nid in nodes[:3] if nid in all_nodes]
                    if pts:
                        cx = sum(p[0] for p in pts) / len(pts)
                        cy = sum(p[1] for p in pts) / len(pts)
                        self.gauss_points_by_id[elem_id] = (cx, cy)
        
        # For backward compatibility, also store as array (but this won't work for sparse IDs)
        if self.gauss_points_by_id:
            sorted_ids = sorted(self.gauss_points_by_id.keys())
            self.gauss_xy = np.array([self.gauss_points_by_id[eid] for eid in sorted_ids], dtype=float)
        else:
            self.gauss_xy = np.empty((0, 2))

        self.fig = None
        self.ax0 = None
        self.ax1 = None
        self.menu_artists = []
        self.menu_drag_artist = None
        self.menu_drag_offset = None
        self.menu_positions = {}
        self.menu_drag_index = None
        self.colorbars = []  # Store references to remove old colorbars

        # Cache loaded results to make click inspection responsive.
        self.result_cache: Dict[Tuple[str, int, str], Tuple[str, Dict[int, np.ndarray]]] = {}
        self.line_elements: List[Tuple[int, List[int], int]] = []
        for _, _, elems in line_meshes:
            self.line_elements.extend(elems)
        self._click_markers = []
        self.info_fig = None
        self.coord_input_fig = None
        self.coord_textbox = None
        self.coord_status_artist = None
        self.deformation_input_fig = None
        self.deformation_textbox = None
        self.deformation_status_artist = None
        self.animation = None
        self.animation_frames = []
        self.animation_running = False
        
        # Profile export data
        self.profile_y_data = None  # numpy array of y coordinates
        self.profile_values = None  # numpy array of values
        self.profile_gradients = None  # numpy array of d(value)/dy or d(value)/ds
        self.profile_x_current = None  # x coordinate of current profile

        # Current nodal field data for CSV export
        self.current_field_node_ids = None
        self.current_field_xyz = None
        self.current_field_source_values = None
        self.current_field_display_values = None
        self.current_field_grad_x = None
        self.current_field_grad_y = None
        self.current_field_grad_magnitude = None

        self.update_plot()

    def _set_profile_data(self, y_data: np.ndarray, values: np.ndarray, profile_x_current: Optional[float]):
        self.profile_y_data = np.asarray(y_data, dtype=float)
        self.profile_values = np.asarray(values, dtype=float)
        self.profile_gradients = finite_difference_gradient(self.profile_y_data, self.profile_values)
        self.profile_x_current = profile_x_current

    def _set_current_field_data(
        self,
        node_ids: Sequence[int],
        xyz: np.ndarray,
        source_values: np.ndarray,
        display_values: np.ndarray,
        grad_x: Optional[np.ndarray] = None,
        grad_y: Optional[np.ndarray] = None,
    ):
        self.current_field_node_ids = np.asarray(node_ids, dtype=int)
        self.current_field_xyz = np.asarray(xyz, dtype=float)
        self.current_field_source_values = np.asarray(source_values, dtype=float)
        self.current_field_display_values = np.asarray(display_values, dtype=float)
        self.current_field_grad_x = None if grad_x is None else np.asarray(grad_x, dtype=float)
        self.current_field_grad_y = None if grad_y is None else np.asarray(grad_y, dtype=float)
        if self.current_field_grad_x is not None and self.current_field_grad_y is not None:
            self.current_field_grad_magnitude = np.sqrt(
                self.current_field_grad_x * self.current_field_grad_x + self.current_field_grad_y * self.current_field_grad_y
            )
        else:
            self.current_field_grad_magnitude = None

    def get_available_steps(self) -> List[int]:
        return sorted(set(h.step for h in self.headers if h.name == self.current_result))

    def get_current_step(self) -> int:
        _, loc = pick_step(self.headers, self.current_result, self.current_step_str, self.current_location)
        self.current_location = loc
        step, _ = pick_step(self.headers, self.current_result, self.current_step_str, self.current_location)
        return step

    def _set_component(self, component: str):
        if component not in VECTOR_COMPONENTS:
            raise ValueError(f"Unsupported vector component '{component}'.")
        self.current_component = component
        self.update_plot()

    def _is_displacement_result(self) -> bool:
        return "DISPLACEMENT" in self.current_result.upper()

    def _toggle_deformation(self):
        if not self._is_displacement_result():
            return
        self.show_deformation = not self.show_deformation
        self.update_plot()

    def _cycle_tensor_component(self):
        index = TENSOR_COMPONENTS.index(self.current_tensor_component) if self.current_tensor_component in TENSOR_COMPONENTS else 0
        self.current_tensor_component = TENSOR_COMPONENTS[(index + 1) % len(TENSOR_COMPONENTS)]
        self.update_plot()

    def _jump_to_result(self, offset: int):
        self.result_idx = (self.result_idx + offset) % len(self.result_names)
        self.current_result = self.result_names[self.result_idx]
        if not self._is_displacement_result():
            self.show_deformation = False
        self.current_step_str = "latest"
        self.current_location = None
        self.update_plot()

    def _reset_view_bounds(self):
        if self.solution_paths and hasattr(self, "folder_view_bounds"):
            self.default_xlim, self.default_ylim = self.folder_view_bounds
            self.view_xlim = self.default_xlim
            self.view_ylim = self.default_ylim
            return

        x_values = [self.xy[:, 0]] if self.xy.size else []
        y_values = [self.xy[:, 1]] if self.xy.size else []
        for msh_path, _ in self.solution_paths:
            if msh_path == self.solution_paths[self.solution_index][0]:
                continue
            all_nodes, _, _, _, _ = parse_post_msh(msh_path)
            if all_nodes:
                points = np.asarray(list(all_nodes.values()), dtype=float)
                x_values.append(points[:, 0])
                y_values.append(points[:, 1])

        x_min = float(np.min(np.concatenate(x_values))) if x_values else 0.0
        x_max = float(np.max(np.concatenate(x_values))) if x_values else 1.0
        y_min = float(np.min(np.concatenate(y_values))) if y_values else 0.0
        y_max = float(np.max(np.concatenate(y_values))) if y_values else 1.0
        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            y_max = y_min + 1.0
        self.default_xlim = (x_min, x_max)
        self.default_ylim = (y_min, y_max)
        if self.solution_paths:
            self.folder_view_bounds = (self.default_xlim, self.default_ylim)
        self.view_xlim = self.default_xlim
        self.view_ylim = self.default_ylim

    def _switch_solution(self, offset: int):
        if len(self.solution_paths) < 2:
            return
        self.solution_index = (self.solution_index + offset) % len(self.solution_paths)
        msh_path, res_path = self.solution_paths[self.solution_index]
        headers = scan_result_headers(res_path)
        all_nodes, triangles, surface_elements, line_meshes, _ = parse_post_msh(msh_path)
        if not headers:
            raise RuntimeError(f"No parsable Result blocks found in {res_path}")

        self.headers = headers
        self.res_path = res_path
        self.open_file_name = res_path.name
        manager = getattr(self.fig.canvas, "manager", None) if self.fig is not None else None
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f"GiD Viewer - {self.open_file_name}")
        self.all_nodes = all_nodes
        self.triangles = triangles
        self.surface_elements = surface_elements
        self.line_meshes = line_meshes
        self.profile_nodes = detect_profile_nodes(
            all_nodes, line_meshes, profile_x=self.profile_x, x_tol=self.x_tol
        )
        self.node_ids, self.id_to_pos, self.pts = build_node_arrays(all_nodes)
        self.xy = self.pts[:, :2]
        self.tri = triangulation_from_connectivity(triangles, self.id_to_pos, self.xy)
        self._reset_view_bounds()
        self.result_names = sorted(set(h.name for h in headers))
        if self.current_result not in self.result_names:
            self.result_idx = 0
            self.current_result = self.result_names[0]
        else:
            self.result_idx = self.result_names.index(self.current_result)
        if not self._is_displacement_result():
            self.show_deformation = False
        available_steps = {header.step for header in headers if header.name == self.current_result}
        if self.current_step_str != "latest":
            try:
                requested_step = int(float(self.current_step_str))
            except (TypeError, ValueError):
                requested_step = None
            if requested_step not in available_steps:
                self.current_step_str = "latest"
                self.current_location = None
        if self.current_location is not None and not any(
            header.name == self.current_result
            and header.location == self.current_location
            and (self.current_step_str == "latest" or header.step == int(float(self.current_step_str)))
            for header in headers
        ):
            self.current_location = None
        self.result_cache.clear()
        self.update_plot()

    def _jump_to_step(self, offset: int):
        steps = self.get_available_steps()
        if not steps:
            return
        try:
            curr_idx = steps.index(int(float(self.current_step_str)))
        except (ValueError, TypeError, IndexError):
            curr_idx = len(steps) - 1
        next_idx = (curr_idx + offset) % len(steps)
        self.current_step_str = str(steps[next_idx])
        self.update_plot()

    def _build_animation_frames(self):
        paths = self.solution_paths or ((None, self.res_path),)
        frames = []
        for solution_index, (_, result_path) in enumerate(paths):
            headers = self.headers if solution_index == self.solution_index else scan_result_headers(result_path)
            result_name = self.current_result if self.current_result in {header.name for header in headers} else None
            if result_name is None:
                result_name = sorted({header.name for header in headers})[0]
            steps = sorted({header.step for header in headers if header.name == result_name})
            frames.extend((solution_index, result_name, str(step)) for step in steps)
        return frames

    def _advance_animation_frame(self, frame_index: int):
        solution_index, result_name, step = self.animation_frames[frame_index]
        if self.solution_paths and solution_index != self.solution_index:
            offset = (solution_index - self.solution_index) % len(self.solution_paths)
            self._switch_solution(offset)
        self.current_result = result_name
        self.result_idx = self.result_names.index(result_name)
        self.current_step_str = step
        self.current_location = None
        self.update_plot()

    def _toggle_animation(self):
        if self.animation_running:
            self.animation_running = False
            if self.animation is not None:
                self.animation.event_source.stop()
            self.draw_menu()
            return

        self.animation_frames = self._build_animation_frames()
        if len(self.animation_frames) < 2:
            return
        self.animation_running = True
        self.animation = animation.FuncAnimation(
            self.fig,
            self._advance_animation_frame,
            frames=len(self.animation_frames),
            interval=250,
            repeat=True,
        )
        self.draw_menu()
        self.fig.canvas.draw_idle()

    def _stop_animation(self):
        self.animation_running = False
        if self.animation is not None:
            self.animation.event_source.stop()

    def _ensure_plot_figure(self):
        if self.fig is None:
            if self.show_profile:
                self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
            else:
                self.fig, self.ax0 = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
                self.ax1 = None
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("pick_event", self._on_menu_pick)
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_menu_motion)
            self.fig.canvas.mpl_connect("button_release_event", self._on_menu_release)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
            self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
            self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
            self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
            manager = getattr(self.fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "set_window_title"):
                manager.set_window_title(f"GiD Viewer - {self.open_file_name}")
        else:
            self._clear_click_markers()
            for cbar in self.colorbars:
                cbar.remove()
            self.colorbars = []
            self.ax0.clear()
            if self.ax1 is not None:
                self.ax1.clear()

    def _show_plot_error(self, message: str):
        self.ax0.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
        self.ax0.set_xlim(0, 1)
        self.ax0.set_ylim(0, 1)
        self.ax0.set_axis_off()
        if self.ax1 is not None:
            self.ax1.set_axis_off()
        self.draw_menu()
        self.fig.canvas.draw()

    def _load_plot_data(self):
        step = self.get_current_step()
        if self.show_gradient or self.current_location == RESULT_LOCATION_ON_GAUSS_POINTS:
            result_type, values = self._load_cached_values(self.current_result, step, self.current_location or "OnNodes")
        else:
            result_type, values = load_values(self.res_path, self.current_result, step, self.current_location)
        return step, result_type, values

    def _compute_plot_arrays(self, result_type: str, values: Dict[int, np.ndarray]):
        is_vector = result_type.lower() == "vector"
        self.current_field_node_ids = None
        self.current_field_xyz = None
        self.current_field_source_values = None
        self.current_field_display_values = None
        self.current_field_grad_x = None
        self.current_field_grad_y = None
        self.current_field_grad_magnitude = None

        active_point_mask = None
        if self.show_gradient:
            if self.tri is None:
                raise ValueError(
                    "Gradient visualization requires nodal data and a valid triangulation."
                )

            plot_xy = self.xy
            scalar = np.full(len(self.node_ids), np.nan, dtype=float)
            ux = np.full(len(self.node_ids), np.nan, dtype=float)
            uy = np.full(len(self.node_ids), np.nan, dtype=float)
            if self.active_filter is not None:
                active_point_mask = np.array([nid in self.active_filter.node_ids for nid in self.node_ids], dtype=bool)
            else:
                active_point_mask = np.ones(len(self.node_ids), dtype=bool)

            for nid, arr in values.items():
                pos = self.id_to_pos.get(nid)
                if pos is None:
                    continue
                scalar[pos] = self._reduce_field_value(result_type, arr)

            if result_type.lower() == "matrix" and not np.any(np.isfinite(scalar[active_point_mask])):
                for nid, arr in values.items():
                    pos = self.id_to_pos.get(nid)
                    if pos is None:
                        continue
                    scalar[pos] = tensor_component(arr, "magnitude")

            scalar[~active_point_mask] = np.nan
            grad_x, grad_y = compute_nodal_gradient(self.tri, scalar)
            grad_magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
            self._set_current_field_data(self.node_ids, self.pts, scalar, grad_magnitude, grad_x, grad_y)
            scalar = grad_magnitude
            ux = grad_x
            uy = grad_y
            scalar[~active_point_mask] = np.nan
            ux[~active_point_mask] = np.nan
            uy[~active_point_mask] = np.nan
        else:
            plot_xy = self.xy
            scalar = np.full(len(self.node_ids), np.nan, dtype=float)
            ux = np.full(len(self.node_ids), np.nan, dtype=float)
            uy = np.full(len(self.node_ids), np.nan, dtype=float)
            if self.active_filter is not None:
                active_point_mask = np.array([nid in self.active_filter.node_ids for nid in self.node_ids], dtype=bool)
            else:
                active_point_mask = np.ones(len(self.node_ids), dtype=bool)

            for nid, arr in values.items():
                pos = self.id_to_pos.get(nid)
                if pos is None:
                    continue
                if is_vector:
                    scalar[pos] = vector_component(arr, self.current_component)
                    ux[pos] = arr[0] if arr.size > 0 else np.nan
                    uy[pos] = arr[1] if arr.size > 1 else np.nan
                elif result_type.lower() == "matrix":
                    scalar[pos] = field_scalar_value(arr, result_type, self.current_component, self.current_tensor_component)
                else:
                    scalar[pos] = arr[0] if arr.size > 0 else np.nan

            self._set_current_field_data(self.node_ids, self.pts, scalar, scalar)

        scalar[~active_point_mask] = np.nan
        ux[~active_point_mask] = np.nan
        uy[~active_point_mask] = np.nan
        return is_vector, plot_xy, scalar, ux, uy, active_point_mask

    def _render_field_plot(self, scalar: np.ndarray, ux: np.ndarray, uy: np.ndarray, plot_xy: np.ndarray, result_type: str, active_point_mask: np.ndarray, is_vector: bool):
        valid = np.isfinite(scalar)
        vmin = vmax = None
        displayed_values = scalar[active_point_mask]
        valid_displayed = np.isfinite(displayed_values)
        if np.any(valid_displayed):
            vmin = float(np.nanmin(displayed_values[valid_displayed]))
            vmax = float(np.nanmax(displayed_values[valid_displayed]))
            if vmin == vmax:
                vmax = vmin + 1e-12

        if self.tri is not None and np.any(valid) and self.active_filter is None and (
            self.current_location in ("OnNodes", RESULT_LOCATION_ON_GAUSS_POINTS) or self.show_gradient
        ):
            scalar_for_plot = scalar.copy()
            fill_value = np.nanmean(scalar[valid])
            scalar_for_plot[~valid] = fill_value
            tpc = self.ax0.tripcolor(
                self.tri,
                scalar_for_plot,
                shading="gouraud",
                cmap=self.current_colormap,
                vmin=vmin,
                vmax=vmax,
            )
            self.colorbars.append(
                self.fig.colorbar(tpc, ax=self.ax0, label=self._current_value_label(result_type))
            )
        elif np.any(valid):
            sc = self.ax0.scatter(
                plot_xy[valid, 0],
                plot_xy[valid, 1],
                c=scalar[valid],
                s=8,
                cmap=self.current_colormap,
                vmin=vmin,
                vmax=vmax,
            )
            self.colorbars.append(
                self.fig.colorbar(sc, ax=self.ax0, label=self._current_value_label(result_type))
            )
        else:
            self.ax0.scatter(plot_xy[:, 0], plot_xy[:, 1], c=scalar, s=8, cmap=self.current_colormap)

        if self.show_isolines and self.tri is not None and self.current_location in (
            "OnNodes", RESULT_LOCATION_ON_GAUSS_POINTS
        ):
            draw_scalar_isolines(
                self.ax0, self.tri, scalar, vmin, vmax,
                count=self.contour_levels, show_labels=self.show_isoline_labels,
            )

        if self.show_mesh and self.tri is not None:
            self.ax0.triplot(self.tri, color="black", linewidth=0.3, alpha=0.35)

        if (
            self.show_deformation
            and is_vector
            and self._is_displacement_result()
            and not self.show_gradient
            and self.current_location == "OnNodes"
            and self.tri is not None
            and len(plot_xy) == len(self.tri.x)
        ):
            displacement_x = np.nan_to_num(ux, nan=0.0)
            displacement_y = np.nan_to_num(uy, nan=0.0)
            deformed_xy = plot_xy.copy()
            deformed_xy[:, 0] += self.deformation_scale * displacement_x
            deformed_xy[:, 1] += self.deformation_scale * displacement_y
            deformed_tri = mtri.Triangulation(
                deformed_xy[:, 0], deformed_xy[:, 1], self.tri.triangles
            )
            self.ax0.triplot(
                deformed_tri,
                color="tab:red",
                linewidth=0.9,
                alpha=0.85,
                label=f"Deformed mesh (x{self.deformation_scale:g})",
            )
            self.ax0.legend(loc="best")

        if is_vector and np.any(np.isfinite(ux)) and np.any(np.isfinite(uy)):
            qmask = np.isfinite(ux) & np.isfinite(uy) & ((ux != 0.0) | (uy != 0.0))
            idx = np.where(qmask)[0]
            if idx.size > 0:
                stride = max(1, idx.size // 800)
                sel = idx[::stride]
                self.ax0.quiver(
                    plot_xy[sel, 0], plot_xy[sel, 1], ux[sel], uy[sel],
                    angles="xy", scale_units="xy", scale=None,
                    width=0.0015, color="k", alpha=0.45,
                )

    def _render_profile_panel(self, scalar: np.ndarray, result_type: str, values: Dict[int, np.ndarray], plot_xy: np.ndarray, active_point_mask: np.ndarray, is_vector: bool):
        profile_drawn = False
        if self.show_profile and self.ax1 is not None and self.tri is not None and self.profile_path and (
            self.current_location in ("OnNodes", RESULT_LOCATION_ON_GAUSS_POINTS) or self.show_gradient
        ):
            try:
                distance, path_values, sampled_points = sample_profile_path_from_triangulation(
                    self.tri, scalar, self.profile_path
                )
                self._set_profile_data(distance, path_values, None)
                self.ax0.plot(
                    sampled_points[:, 0], sampled_points[:, 1],
                    color="red", linewidth=1.0, alpha=0.8, label="Profile path",
                )
                self.ax0.legend(loc="best")
                self.ax1.plot(path_values, distance, "-", linewidth=1.5)
                component_label = self._display_component_name(result_type, is_vector)
                self.ax1.set_xlabel(f"{self.current_result} ({component_label})")
                self.ax1.set_ylabel("distance")
                self.ax1.set_title("Profile path")
                self.ax1.grid(True, alpha=0.3)
                profile_drawn = True
            except (RuntimeError, ValueError):
                pass

        if self.show_profile and self.ax1 is not None and self.tri is not None and not profile_drawn and np.isfinite(self.profile_x) and (
            self.current_location in ("OnNodes", RESULT_LOCATION_ON_GAUSS_POINTS) or self.show_gradient
        ):
            if self.active_filter is not None:
                y_candidates = plot_xy[active_point_mask, 1]
            else:
                y_candidates = self.xy[:, 1]

            finite_y = np.isfinite(y_candidates)
            if np.any(finite_y):
                try:
                    y_min = float(np.min(y_candidates[finite_y]))
                    y_max = float(np.max(y_candidates[finite_y]))
                    y_profile, v_profile = sample_vertical_profile_from_triangulation(
                        self.tri, scalar, self.profile_x, y_min, y_max
                    )
                    self._set_profile_data(y_profile, v_profile, self.profile_x)
                    self.ax0.axvline(self.profile_x, color="red", linewidth=1.0, alpha=0.7, label="Profile x")
                    self.ax0.legend(loc="best")
                    self.ax1.plot(v_profile, y_profile, "-", linewidth=1.5)
                    component_label = self._display_component_name(result_type, is_vector)
                    self.ax1.set_xlabel(f"{self.current_result} ({component_label})")
                    self.ax1.set_ylabel("y")
                    self.ax1.set_title("Profile")
                    self.ax1.grid(True, alpha=0.3)
                    profile_drawn = True
                except RuntimeError:
                    pass

        if self.show_profile and self.ax1 is not None and not profile_drawn:
            sp_ids = sorted([nid for nid in self.profile_nodes if nid in self.id_to_pos])
            if self.active_filter is not None:
                sp_ids = [nid for nid in sp_ids if nid in self.active_filter.node_ids]
            if self.show_gradient:
                sp_pts = np.array([self.all_nodes[nid] for nid in sp_ids], dtype=float) if sp_ids else np.empty((0, 3))
                sp_vals = np.array([scalar[self.id_to_pos[nid]] for nid in sp_ids], dtype=float) if sp_ids else np.array([])
            else:
                sp_ids = [nid for nid in sp_ids if nid in values]
                sp_pts = np.array([self.all_nodes[nid] for nid in sp_ids], dtype=float) if sp_ids else np.empty((0, 3))
                sp_vals = np.array([
                    vector_component(values[nid], self.current_component)
                    if is_vector
                    else tensor_component(values[nid], self.current_tensor_component)
                    if result_type.lower() == "matrix"
                    else float(values[nid][0])
                    for nid in sp_ids
                ]) if sp_ids else np.array([])

            if sp_pts.size > 0:
                self.ax0.scatter(sp_pts[:, 0], sp_pts[:, 1], c="red", s=10, label="Profile nodes")
                self.ax0.legend(loc="best")
                order = np.argsort(sp_pts[:, 1])
                self.ax1.plot(sp_vals[order], sp_pts[:, 1][order], "-o", ms=3)
                self._set_profile_data(sp_pts[:, 1][order], sp_vals[order], None)
                component_label = self._display_component_name(result_type, is_vector)
                self.ax1.set_xlabel(f"{self.current_result} ({component_label})")
                self.ax1.set_ylabel("y")
                self.ax1.set_title("Profile")
                self.ax1.grid(True, alpha=0.3)
            else:
                self.ax1.text(
                    0.5, 0.5, "No profile nodes with selected result values",
                    ha="center", va="center"
                )
                self.ax1.set_axis_off()

    def _load_plot_state(self):
        try:
            step, result_type, values = self._load_plot_data()
        except Exception as exc:
            self._show_plot_error(f"Error: {str(exc)}")
            return None, None, None, None

        try:
            is_vector, plot_xy, scalar, ux, uy, active_point_mask = self._compute_plot_arrays(result_type, values)
        except ValueError as exc:
            self._show_plot_error(str(exc))
            return None, None, None, None

        return step, result_type, values, (is_vector, plot_xy, scalar, ux, uy, active_point_mask)

    def _finalize_plot_layout(self, step: int):
        self.ax0.set_aspect("equal", adjustable="box")
        self.ax0.set_autoscale_on(False)
        self.ax0.set_xlabel("x")
        self.ax0.set_ylabel("y")
        self.ax0.set_title(f"{self.current_result} @ step {step}")
        self.ax0.set_xlim(self.view_xlim)
        self.ax0.set_ylim(self.view_ylim)
        self.draw_menu()
        self.fig.canvas.draw()
        self.fig.set_layout_engine(None)

    def update_plot(self):
        self._ensure_plot_figure()
        step, result_type, values, plot_state = self._load_plot_state()
        if step is None or result_type is None or values is None or plot_state is None:
            return

        is_vector, plot_xy, scalar, ux, uy, active_point_mask = plot_state
        self._render_field_plot(scalar, ux, uy, plot_xy, result_type, active_point_mask, is_vector)
        self._render_profile_panel(scalar, result_type, values, plot_xy, active_point_mask, is_vector)
        self._finalize_plot_layout(step)

    def draw_menu(self):
        for artist in self.menu_artists:
            artist.remove()
        key_menu = (
            "INTERACTIVE MODE\n"
            "================\n"
            "[n/p] Next/Prev result\n"
            "[N/P] Next/Prev solution\n"
            "[a] Animate solutions/steps on/off\n"
            "[j/k] Prev/Next step\n"
            "[x/y/z/g] Component (x,y,z,magnitude)\n"
            "[t] Tensor reduction (xx,xy,yx,yy,trace,magnitude)\n"
            "[d] Gradient on/off\n"
            "[u] Deformed mesh on/off, [U] Set magnification\n"
            "[l] Isolines on/off\n"
            "[L] Isoline labels on/off\n"
            "[c] Next scale/colormap\n"
            "[s] Save current picture\n"
            "[m] Mesh on/off\n"
            "[+/-] Zoom in/out, [r] reset zoom, [mouse wheel] zoom\n"
            "[right-drag or Shift+left-drag] Pan/shift view\n"
            "[mouse-left] Inspect nearest mesh node\n"
            "[i] Type coordinate for inspection\n"
            "[e] Export profile\n"
            "[Shift+e] Export gradient field nodes\n"
            "[?] List results\n"
            "[q] Quit"
        )
        state_menu = (
            "CURRENT STATE\n"
            "=============\n"
            f"Solution: {self.solution_index + 1}/{len(self.solution_paths) or 1}\n"
            f"Animation: {'on' if self.animation_running else 'off'}\n"
            f"Result: {self.current_result}\n"
            f"Location: {self.current_location}\n"
            f"Component: {self.current_component}\n"
            f"Tensor: {self.current_tensor_component}\n"
            f"Gradient: {'on' if self.show_gradient else 'off'}\n"
            f"Deformation: {'on' if self.show_deformation else 'off'} (x{self.deformation_scale:g})\n"
            f"Isolines: {'on' if self.show_isolines else 'off'}\n"
            f"Scale: {self.current_colormap}\n"
            f"Mesh: {'on' if self.show_mesh else 'off'}\n"
            f"Active SubModelPart: {self.active_filter.name if self.active_filter else '-'}"
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
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                family="monospace", picker=True,
            ),
            self.fig.text(
                *positions[1], state_menu, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                family="monospace", picker=True,
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

    def _load_cached_values(self, result_name: str, step: int, location: str) -> Tuple[str, Dict[int, np.ndarray]]:
        key = (result_name, int(step), location)
        if key not in self.result_cache:
            _, _, result_type, values = load_result_values(
                self.headers,
                self.res_path,
                result_name,
                str(step),
                location,
                self.surface_elements,
                self.line_meshes,
            )
            self.result_cache[key] = (result_type, values)
        return self.result_cache[key]

    def _select_step_and_location_for_result(self, result_name: str, preferred_step: int) -> Tuple[int, str]:
        on_nodes_steps = sorted(
            set(h.step for h in self.headers if h.name == result_name and h.location == "OnNodes")
        )
        on_gp_steps = sorted(
            set(h.step for h in self.headers if h.name == result_name and h.location == "OnGaussPoints")
        )

        if preferred_step in on_nodes_steps:
            return preferred_step, "OnNodes"
        if preferred_step in on_gp_steps:
            return preferred_step, "OnGaussPoints"
        if on_nodes_steps:
            return on_nodes_steps[-1], "OnNodes"
        if on_gp_steps:
            return on_gp_steps[-1], "OnGaussPoints"
        raise ValueError(f"No available steps for result '{result_name}'.")

    def _format_result_line(self, result_name: str, step: int, location: str, result_type: str, arr: np.ndarray) -> str:
        if arr.size == 0:
            return f"{result_name}: <empty> [step {step}, {location}]"

        if result_type.lower() == "vector":
            vx = arr[0] if arr.size > 0 else float("nan")
            vy = arr[1] if arr.size > 1 else float("nan")
            vz = arr[2] if arr.size > 2 else float("nan")
            mag = vector_component(arr, "magnitude")
            return (
                f"{result_name}: x={vx:.6g}, y={vy:.6g}, z={vz:.6g}, |v|={mag:.6g} "
                f"[step {step}, {location}]"
            )

        if result_type.lower() == "matrix":
            xx = tensor_component(arr, "xx")
            xy = tensor_component(arr, "xy")
            yx = tensor_component(arr, "yx")
            yy = tensor_component(arr, "yy")
            trace = tensor_component(arr, "trace")
            mag = tensor_component(arr, "magnitude")
            return (
                f"{result_name}: xx={xx:.6g}, xy={xy:.6g}, yx={yx:.6g}, yy={yy:.6g}, "
                f"trace={trace:.6g}, |T|={mag:.6g} [step {step}, {location}]"
            )

        return f"{result_name}: {float(arr[0]):.6g} [step {step}, {location}]"

    def _display_component_name(self, result_type: str, is_vector: bool) -> str:
        rtype = result_type.lower()
        if is_vector:
            return self.current_component
        if rtype == "matrix":
            return self.current_tensor_component
        return "scalar"

    def _current_value_label(self, result_type: str) -> str:
        rtype = result_type.lower()
        if self.show_gradient:
            if rtype == "vector":
                return f"|grad {self.current_result} ({self.current_component})|"
            if rtype == "matrix":
                return f"|grad {self.current_result} ({self.current_tensor_component})|"
            return f"|grad {self.current_result}|"

        if rtype == "vector":
            return f"{self.current_result} ({self.current_component})"
        if rtype == "matrix":
            return f"{self.current_result} ({self.current_tensor_component})"
        return f"{self.current_result} (scalar)"

    def _reduce_field_value(self, result_type: str, arr: np.ndarray) -> float:
        return reduce_result_value(
            arr,
            result_type,
            self.current_component,
            self.current_tensor_component,
        )

    def _build_node_report(self, click_x: float, click_y: float, node_id: int, node_x: float, node_y: float, dist: float) -> str:
        try:
            preferred_step = self.get_current_step()
        except Exception:
            preferred_step = max(h.step for h in self.headers)

        lines = [
            "Nearest Node Inspection",
            "=======================",
            f"Mouse click: x={click_x:.6g}, y={click_y:.6g}",
            f"Nearest node: id={node_id}, x={node_x:.6g}, y={node_y:.6g}",
            f"Distance: {dist:.6g}",
            "",
            "Results at nearest node:",
        ]

        for name in self.result_names:
            try:
                step, loc = self._select_step_and_location_for_result(name, preferred_step)
                result_type, values = self._load_cached_values(name, step, loc)
                arr = values.get(node_id)
                if arr is None:
                    lines.append(f"{name}: <no value> [step {step}, {loc}]")
                else:
                    lines.append(self._format_result_line(name, step, loc, result_type, arr))
            except Exception as exc:
                lines.append(f"{name}: <error: {exc}>")

        return "\n".join(lines)

    def _show_info_popup(self, text: str):
        popup_title = f"Nearest Mesh Node Results - {self.open_file_name}"
        if self.info_fig is None or not plt.fignum_exists(self.info_fig.number):
            self.info_fig = plt.figure(popup_title, figsize=(11, 8))
        else:
            self.info_fig.clf()

        manager = getattr(self.info_fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(popup_title)

        ax = self.info_fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(
            0.01,
            0.99,
            text,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=ax.transAxes,
        )
        self.info_fig.tight_layout()
        self.info_fig.canvas.draw_idle()
        self.info_fig.show()

    def _inspect_coordinate(self, click_x: float, click_y: float, source: str = "mouse"):
        if self.xy.size == 0:
            return

        dx = self.xy[:, 0] - click_x
        dy = self.xy[:, 1] - click_y
        d2 = dx * dx + dy * dy
        pos = int(np.argmin(d2))
        node_id = self.node_ids[pos]
        node_x = float(self.xy[pos, 0])
        node_y = float(self.xy[pos, 1])
        dist = float(math.sqrt(d2[pos]))

        self._clear_click_markers()

        self._click_markers.append(
            self.ax0.plot(click_x, click_y, marker="x", color="black", markersize=8, mew=1.5)[0]
        )
        self._click_markers.append(
            self.ax0.plot(node_x, node_y, marker="o", markerfacecolor="none", markeredgecolor="yellow", markersize=10, mew=1.8)[0]
        )
        self.fig.canvas.draw_idle()

        print(
            f"{source} inspection at (x={click_x:.6g}, y={click_y:.6g}) -> nearest node {node_id} "
            f"at (x={node_x:.6g}, y={node_y:.6g}), distance={dist:.6g}"
        )

        report = self._build_node_report(click_x, click_y, node_id, node_x, node_y, dist)
        self._show_info_popup(report)

    def _clear_click_markers(self):
        for marker in self._click_markers:
            try:
                marker.remove()
            except (ValueError, AttributeError):
                # Marker may already be detached after an axes redraw.
                pass
        self._click_markers = []

    def _open_coordinate_input_popup(self):
        popup_title = f"Inspect Coordinate - {self.open_file_name}"
        if self.coord_input_fig is None or not plt.fignum_exists(self.coord_input_fig.number):
            self.coord_input_fig = plt.figure(popup_title, figsize=(6, 2.3))
        else:
            self.coord_input_fig.clf()

        manager = getattr(self.coord_input_fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(popup_title)

        self.coord_input_fig.text(
            0.02,
            0.86,
            "Type coordinate as: x,y (example: 18.0, -4.2) and press Enter",
            fontsize=9,
        )
        text_ax = self.coord_input_fig.add_axes([0.05, 0.45, 0.9, 0.25])
        self.coord_textbox = TextBox(text_ax, "x,y: ")

        self.coord_status_artist = self.coord_input_fig.text(0.02, 0.1, "", fontsize=9, color="crimson")

        def _submit(text: str):
            parts = [p for p in re.split(r"[\s,;]+", text.strip()) if p]
            if len(parts) != 2:
                self.coord_status_artist.set_text("Please enter exactly two numbers: x,y")
                self.coord_input_fig.canvas.draw_idle()
                return
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
            except ValueError:
                self.coord_status_artist.set_text("Invalid numeric format. Use for example: 18.0, -4.2")
                self.coord_input_fig.canvas.draw_idle()
                return

            self.coord_status_artist.set_text("")
            self.coord_input_fig.canvas.draw_idle()
            self._inspect_coordinate(x_val, y_val, source="typed")

        self.coord_textbox.on_submit(_submit)
        self.coord_textbox.set_active(True)
        self.coord_input_fig.canvas.draw_idle()
        self.coord_input_fig.show()

        # Best-effort focus for common GUI backends so typing works immediately.
        manager = getattr(self.coord_input_fig.canvas, "manager", None)
        window = getattr(manager, "window", None)
        if window is not None:
            if hasattr(window, "focus_force"):
                window.focus_force()
            if hasattr(window, "raise_"):
                window.raise_()
            if hasattr(window, "activateWindow"):
                window.activateWindow()

    def _open_deformation_scale_popup(self):
        popup_title = f"Deformation Magnification - {self.open_file_name}"
        if self.deformation_input_fig is None or not plt.fignum_exists(self.deformation_input_fig.number):
            self.deformation_input_fig = plt.figure(popup_title, figsize=(6, 2.3))
        else:
            self.deformation_input_fig.clf()

        manager = getattr(self.deformation_input_fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(popup_title)

        self.deformation_input_fig.text(
            0.02, 0.86,
            "Type a positive displacement magnification and press Enter",
            fontsize=9,
        )
        text_ax = self.deformation_input_fig.add_axes([0.05, 0.45, 0.9, 0.25])
        self.deformation_textbox = TextBox(text_ax, "magnification: ")
        self.deformation_textbox.set_val(str(self.deformation_scale))
        self.deformation_status_artist = self.deformation_input_fig.text(
            0.02, 0.1, "", fontsize=9, color="crimson"
        )

        def _submit(text: str):
            try:
                scale = float(text)
            except ValueError:
                self.deformation_status_artist.set_text("Enter a positive finite number")
                self.deformation_input_fig.canvas.draw_idle()
                return
            if not math.isfinite(scale) or scale <= 0.0:
                self.deformation_status_artist.set_text("Enter a positive finite number")
                self.deformation_input_fig.canvas.draw_idle()
                return
            self.deformation_scale = scale
            if not self._is_displacement_result():
                self.deformation_status_artist.set_text(
                    "Deformation requires a DISPLACEMENT result"
                )
                self.deformation_input_fig.canvas.draw_idle()
                return
            self.show_deformation = True
            self.deformation_status_artist.set_text("")
            self.deformation_input_fig.canvas.draw_idle()
            self.update_plot()

        self.deformation_textbox.on_submit(_submit)
        self.deformation_textbox.set_active(True)
        self.deformation_input_fig.canvas.draw_idle()
        self.deformation_input_fig.show()

    def on_click(self, event):
        if self.ax0 is None or event.inaxes != self.ax0:
            return

        key_text = str(event.key or "").lower()
        if "shift" in key_text:
            return

        btn = event.button
        is_left = (btn == 1) or (getattr(btn, "name", "") == "LEFT")
        if not is_left:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._inspect_coordinate(float(event.xdata), float(event.ydata), source="mouse")

    def _zoom_main_axes(self, factor: float, center_x: Optional[float] = None, center_y: Optional[float] = None):
        if self.ax0 is None:
            return
        x0, x1 = self.ax0.get_xlim()
        y0, y1 = self.ax0.get_ylim()
        if center_x is None:
            center_x = 0.5 * (x0 + x1)
        if center_y is None:
            center_y = 0.5 * (y0 + y1)

        new_w = (x1 - x0) * factor
        new_h = (y1 - y0) * factor
        if new_w <= 0 or new_h <= 0:
            return

        self.ax0.set_xlim(center_x - 0.5 * new_w, center_x + 0.5 * new_w)
        self.ax0.set_ylim(center_y - 0.5 * new_h, center_y + 0.5 * new_h)
        self._store_current_view()
        self.fig.canvas.draw_idle()

    def _store_current_view(self):
        if self.ax0 is None:
            return
        self.view_xlim = tuple(float(v) for v in self.ax0.get_xlim())
        self.view_ylim = tuple(float(v) for v in self.ax0.get_ylim())

    def on_button_press(self, event):
        if self.ax0 is None or event.inaxes != self.ax0:
            return
        if event.xdata is None or event.ydata is None:
            return

        btn = event.button
        is_right = (btn == 3) or (getattr(btn, "name", "") == "RIGHT")
        key_text = str(event.key or "").lower()
        shift_active = "shift" in key_text
        is_left_with_shift = ((btn == 1) or (getattr(btn, "name", "") == "LEFT")) and shift_active

        if not (is_right or is_left_with_shift):
            return

        self._pan_active = True
        self._pan_start_data = (float(event.xdata), float(event.ydata))
        self._pan_start_xlim = tuple(float(v) for v in self.ax0.get_xlim())
        self._pan_start_ylim = tuple(float(v) for v in self.ax0.get_ylim())

    def on_mouse_move(self, event):
        if not self._pan_active or self.ax0 is None:
            return
        if event.inaxes != self.ax0 or event.xdata is None or event.ydata is None:
            return
        if self._pan_start_data is None or self._pan_start_xlim is None or self._pan_start_ylim is None:
            return

        x0, y0 = self._pan_start_data
        dx = float(event.xdata) - x0
        dy = float(event.ydata) - y0
        sx0, sx1 = self._pan_start_xlim
        sy0, sy1 = self._pan_start_ylim

        self.ax0.set_xlim(sx0 - dx, sx1 - dx)
        self.ax0.set_ylim(sy0 - dy, sy1 - dy)
        self._store_current_view()
        self.fig.canvas.draw_idle()

    def on_button_release(self, event):
        if not self._pan_active:
            return
        self._pan_active = False
        self._pan_start_data = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

    def on_scroll(self, event):
        if self.ax0 is None or event.inaxes != self.ax0:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == "up":
            self._zoom_main_axes(0.85, float(event.xdata), float(event.ydata))
        elif event.button == "down":
            self._zoom_main_axes(1.15, float(event.xdata), float(event.ydata))

    def export_profile_data(self, out_path: Optional[Path] = None, *, strict: bool = False):
        """Export profile CSV; strict callers receive its path or an exception."""
        if self.profile_y_data is None or self.profile_values is None:
            if strict:
                raise ValueError("No profile data to export. Please ensure profile is visible.")
            print("No profile data to export. Please ensure profile is visible.")
            return

        try:
            step = int(float(self.current_step_str)) if self.current_step_str != "latest" else "latest"
            out_path = out_path or build_result_export_path(
                self.current_result,
                str(step),
                self.current_location,
                "profile",
                self.current_component,
            )

            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['y_coordinate', 'value', 'gradient'])
                gradients = self.profile_gradients
                if gradients is None or len(gradients) != len(self.profile_values):
                    gradients = finite_difference_gradient(self.profile_y_data, self.profile_values)
                for y, val, grad in zip(self.profile_y_data, self.profile_values, gradients):
                    writer.writerow([y, val, grad])

            print(f"Profile exported to: {out_path.resolve()}")
            return out_path
        except Exception as e:
            if strict:
                raise
            print(f"Error exporting profile: {e}")

    def export_current_field_data(self, out_path: Optional[Path] = None, *, strict: bool = False):
        """Export gradient CSV; strict callers receive its path or an exception."""
        if not self.show_gradient:
            if strict:
                raise ValueError("Gradient export requires gradient mode.")
            print("Gradient export is only available while gradient mode is on. Press 'd' first.")
            return
        if (
            self.current_field_node_ids is None
            or self.current_field_xyz is None
            or self.current_field_source_values is None
            or self.current_field_display_values is None
            or self.current_field_grad_x is None
            or self.current_field_grad_y is None
            or self.current_field_grad_magnitude is None
        ):
            if strict:
                raise ValueError("No gradient field is currently available to export.")
            print("No gradient field is currently available to export.")
            return

        try:
            step = int(float(self.current_step_str)) if self.current_step_str != "latest" else "latest"
            result_type = next((h.result_type for h in self.headers if h.name == self.current_result), "Scalar")
            tensor_component = self.current_tensor_component if result_type.lower() == "matrix" else None
            out_path = out_path or build_result_export_path(
                self.current_result,
                str(step),
                self.current_location,
                "gradient_nodes",
                self.current_component,
                tensor_component,
            )

            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "node_id",
                    "x",
                    "y",
                    "z",
                    "source_value",
                    "gradient_x",
                    "gradient_y",
                    "gradient_magnitude",
                ])

                valid_mask = np.isfinite(self.current_field_source_values)
                valid_mask &= np.isfinite(self.current_field_grad_x)
                valid_mask &= np.isfinite(self.current_field_grad_y)
                valid_mask &= np.isfinite(self.current_field_grad_magnitude)

                exported_count = 0
                skipped_count = 0
                for idx, node_id in enumerate(self.current_field_node_ids):
                    if not valid_mask[idx]:
                        skipped_count += 1
                        continue
                    x, y, z = self.current_field_xyz[idx]
                    source_value = float(self.current_field_source_values[idx])
                    grad_x = float(self.current_field_grad_x[idx])
                    grad_y = float(self.current_field_grad_y[idx])
                    grad_magnitude = float(self.current_field_grad_magnitude[idx])

                    # Hard guard: never write invalid numeric rows.
                    if not (
                        math.isfinite(source_value)
                        and math.isfinite(grad_x)
                        and math.isfinite(grad_y)
                        and math.isfinite(grad_magnitude)
                    ):
                        skipped_count += 1
                        continue

                    writer.writerow([
                        int(node_id),
                        float(x),
                        float(y),
                        float(z),
                        source_value,
                        grad_x,
                        grad_y,
                        grad_magnitude,
                    ])
                    exported_count += 1

            print(
                f"Gradient field exported to: {out_path.resolve()} "
                f"({exported_count} valid nodes, {skipped_count} skipped)"
            )
            return out_path
        except Exception as e:
            if strict:
                raise
            print(f"Error exporting gradient field: {e}")

    def _toggle_flag(self, attribute_name: str):
        setattr(self, attribute_name, not getattr(self, attribute_name))
        self.update_plot()

    def save_current_plot(self, out_path: Optional[Path] = None) -> Path:
        if out_path is None:
            out_path = build_output_path(self.current_result, self.current_step_str, self.current_location, "_interactive.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_path, dpi=180)
        print(f"Current plot saved to: {out_path.resolve()}")
        return out_path

    def save_current_settings(self, out_path: Optional[Path] = None) -> Path:
        if out_path is None:
            out_path = Path(f"{self.current_result}_settings.json")
        settings = {
            "mode": "single",
            "result": self.current_result,
            "step": self.current_step_str,
            "location": self.current_location,
            "component": self.current_component,
            "tensor_component": self.current_tensor_component,
            "profile_x": self.profile_x,
            "profile_path": [list(point) for point in self.profile_path] if self.profile_path else None,
            "x_tol": self.x_tol,
            "show_gradient": self.show_gradient,
            "show_deformation": self.show_deformation,
            "deformation_scale": self.deformation_scale,
            "show_mesh": self.show_mesh,
            "show_isolines": self.show_isolines,
            "show_isoline_labels": self.show_isoline_labels,
            "show_profile": self.show_profile,
            "colormap": self.current_colormap,
            "active_submodelpart": self.active_filter.name if self.active_filter else None,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"Visualization settings saved to: {out_path.resolve()}")
        return out_path

    def on_key(self, event):
        key_actions = {
            "q": lambda: (self._stop_animation(), plt.close(self.fig)),
            "n": lambda: self._jump_to_result(1),
            "p": lambda: self._jump_to_result(-1),
            "N": lambda: self._switch_solution(1),
            "P": lambda: self._switch_solution(-1),
            "a": self._toggle_animation,
            "k": lambda: self._jump_to_step(1),
            "j": lambda: self._jump_to_step(-1),
            "x": lambda: self._set_component("x"),
            "y": lambda: self._set_component("y"),
            "z": lambda: self._set_component("z"),
            "g": lambda: self._set_component("magnitude"),
            "M": lambda: self._set_component("magnitude"),
            "t": self._cycle_tensor_component,
            "d": lambda: self._toggle_flag("show_gradient"),
            "u": self._toggle_deformation,
            "U": self._open_deformation_scale_popup,
            "m": lambda: self._toggle_flag("show_mesh"),
            "l": lambda: self._toggle_flag("show_isolines"),
            "L": lambda: self._toggle_flag("show_isoline_labels"),
            "shift+l": lambda: self._toggle_flag("show_isoline_labels"),
            "r": lambda: self._reset_view(),
            "+": lambda: self._zoom_main_axes(0.85),
            "=": lambda: self._zoom_main_axes(0.85),
            "-": lambda: self._zoom_main_axes(1.15),
            "c": lambda: self._advance_colormap(),
            "s": self.save_current_plot,
            "e": self.export_profile_data,
            "E": self.export_current_field_data,
            "shift+e": self.export_current_field_data,
            "i": self._open_coordinate_input_popup,
            "?": self._print_available_results,
        }

        action = key_actions.get(event.key)
        if action is None:
            return
        action()

    def _reset_view(self):
        self.ax0.set_xlim(self.default_xlim)
        self.ax0.set_ylim(self.default_ylim)
        self._store_current_view()
        self.fig.canvas.draw_idle()

    def _advance_colormap(self):
        self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
        self.current_colormap = self.colormaps[self.colormap_idx]
        self.update_plot()

    def _print_available_results(self):
        print("\nAvailable results:")
        grouped = defaultdict(lambda: defaultdict(set))
        for h in self.headers:
            grouped[(h.name, h.result_type)][h.location].add(h.step)
        for (name, rtype), locs in sorted(grouped.items()):
            for loc in sorted(locs.keys()):
                steps = locs[loc]
                print(f"  {name:30s} type={rtype:8s} loc={loc:15s} steps={min(steps)}..{max(steps)}")
