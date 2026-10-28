"""Static rendering and interactive viewer entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from .calculations import build_node_arrays, build_scalar_field_from_values, field_scalar_value, sample_profile_path_from_triangulation, triangulation_from_connectivity, vector_component
from .comparison import ComparisonViewer
from .drawing import draw_scalar_isolines, draw_triangulation_boundary
from .models import ActiveSubmodelPart, ResultHeader, SolutionData
from .viewer import InteractiveViewer


# The script uses S for direct picture export and L for isoline controls; avoid
# Matplotlib's conflicting save-dialog and logarithmic-Y shortcuts.
plt.rcParams["keymap.save"] = [key for key in plt.rcParams["keymap.save"] if key not in ("s", "S")]
plt.rcParams["keymap.yscale"] = [key for key in plt.rcParams["keymap.yscale"] if key not in ("l", "L")]


def render_plot_static(
    all_nodes: Dict[int, Tuple[float, float, float]],
    triangles: Sequence[Tuple[int, int, int]],
    values: Dict[int, np.ndarray],
    result_name: str,
    step: int,
    result_type: str,
    component: str,
    tensor_component_mode: str,
    profile_nodes: set,
    out_path: Path,
    active_filter: Optional[ActiveSubmodelPart],
    show_isolines: bool = False,
    show_profile: bool = False,
    contour_levels: int = 12,
    show_isoline_labels: bool = False,
    profile_path: Optional[Sequence[Tuple[float, float]]] = None,
):
    """Static plot saved to file (non-interactive)."""
    node_ids, id_to_pos, pts = build_node_arrays(all_nodes)
    xy = pts[:, :2]

    tri = triangulation_from_connectivity(triangles, id_to_pos, xy)

    scalar = build_scalar_field_from_values(
        values,
        node_ids,
        id_to_pos,
        result_type,
        component,
        tensor_component_mode,
    )
    ux = np.full(len(node_ids), np.nan, dtype=float)
    uy = np.full(len(node_ids), np.nan, dtype=float)

    is_vector = result_type.lower() == "vector"

    for nid, arr in values.items():
        pos = id_to_pos.get(nid)
        if pos is None:
            continue

        if is_vector:
            ux[pos] = arr[0] if arr.size > 0 else np.nan
            uy[pos] = arr[1] if arr.size > 1 else np.nan

    if active_filter is not None:
        active_mask = np.array([nid in active_filter.node_ids for nid in node_ids], dtype=bool)
        scalar[~active_mask] = np.nan
        ux[~active_mask] = np.nan
        uy[~active_mask] = np.nan
    else:
        active_mask = np.ones(len(node_ids), dtype=bool)

    vmin = vmax = None
    finite_mask = np.isfinite(scalar)
    if np.any(finite_mask):
        vmin = float(np.nanmin(scalar[finite_mask]))
        vmax = float(np.nanmax(scalar[finite_mask]))
        if vmin == vmax:
            vmax = vmin + 1e-12

    if show_profile:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
        ax1 = None

    valid = np.isfinite(scalar)
    if tri is not None and np.any(valid) and active_filter is None:
        scalar_for_plot = scalar.copy()
        fill_value = np.nanmean(scalar[valid])
        scalar_for_plot[~valid] = fill_value
        tpc = ax0.tripcolor(
            tri,
            scalar_for_plot,
            shading="gouraud",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(
            tpc,
            ax=ax0,
            label=f"{result_name} ({component if is_vector else tensor_component_mode if result_type.lower() == 'matrix' else 'scalar'})",
        )
    else:
        ax0.scatter(xy[valid, 0], xy[valid, 1], c=scalar[valid], s=8, cmap="viridis", vmin=vmin, vmax=vmax)

    if show_isolines and tri is not None and result_type.lower() != "vector":
        draw_scalar_isolines(
            ax0, tri, scalar, vmin, vmax,
            count=contour_levels, show_labels=show_isoline_labels,
        )

    if is_vector and np.any(np.isfinite(ux)) and np.any(np.isfinite(uy)):
        qmask = np.isfinite(ux) & np.isfinite(uy) & ((ux != 0.0) | (uy != 0.0))
        idx = np.where(qmask)[0]
        if idx.size > 0:
            stride = max(1, idx.size // 800)
            sel = idx[::stride]
            ax0.quiver(
                xy[sel, 0], xy[sel, 1], ux[sel], uy[sel],
                angles="xy", scale_units="xy", scale=None,
                width=0.0015, color="k", alpha=0.45,
            )

    # In static mode we still use detected profile nodes; interactive mode now
    # performs triangulation-based profile sampling to keep void regions empty.
    sp_ids = sorted([nid for nid in profile_nodes if nid in id_to_pos and nid in values])
    if active_filter is not None:
        sp_ids = [nid for nid in sp_ids if nid in active_filter.node_ids]
    sp_pts = np.array([all_nodes[nid] for nid in sp_ids], dtype=float) if sp_ids else np.empty((0, 3))
    sp_vals = np.array([
        vector_component(values[nid], component)
        if is_vector
        else field_scalar_value(values[nid], result_type, component, tensor_component_mode)
        if result_type.lower() == "matrix"
        else float(values[nid][0])
        for nid in sp_ids
    ]) if sp_ids else np.array([])

    if show_profile and ax1 is not None and profile_path and tri is not None:
        distance, path_values, sampled_points = sample_profile_path_from_triangulation(
            tri, scalar, profile_path
        )
        ax0.plot(
            sampled_points[:, 0], sampled_points[:, 1],
            color="red", linewidth=1.0, alpha=0.8, label="Profile path",
        )
        ax0.legend(loc="best")
        ax1.plot(path_values, distance, "-", linewidth=1.5)
        ax1.set_xlabel(f"{result_name} ({component if is_vector else tensor_component_mode if result_type.lower() == 'matrix' else 'scalar'})")
        ax1.set_ylabel("distance")
        ax1.set_title("Profile path")
        ax1.grid(True, alpha=0.3)
    elif show_profile and ax1 is not None and sp_pts.size > 0:
        ax0.scatter(sp_pts[:, 0], sp_pts[:, 1], c="red", s=10, label="Profile nodes")
        ax0.legend(loc="best")
        order = np.argsort(sp_pts[:, 1])
        ax1.plot(sp_vals[order], sp_pts[:, 1][order], "-o", ms=3)
        ax1.set_xlabel(
            f"{result_name} ({component if is_vector else tensor_component_mode if result_type.lower() == 'matrix' else 'scalar'})"
        )
        ax1.set_ylabel("y")
        ax1.set_title("Profile")
        ax1.grid(True, alpha=0.3)
    elif show_profile and ax1 is not None:
        ax1.text(0.5, 0.5, "No profile nodes with selected result values", ha="center", va="center")
        ax1.set_axis_off()

    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_title(f"{result_name} at step {step}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved plot to: {out_path}")
    plt.close(fig)


def render_plot_interactive(
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
    """Launch interactive viewer."""
    viewer = InteractiveViewer(
        headers=headers,
        res_path=res_path,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements,
        line_meshes=line_meshes,
        profile_nodes=profile_nodes,
        initial_result=initial_result,
        initial_step=initial_step,
        initial_location=initial_location,
        profile_x=profile_x,
        x_tol=x_tol,
        initial_tensor_component=initial_tensor_component,
        active_filter=active_filter,
        show_isolines=show_isolines,
        show_profile=show_profile,
        contour_levels=contour_levels,
        show_isoline_labels=show_isoline_labels,
        initial_component=initial_component,
        initial_show_gradient=initial_show_gradient,
        initial_show_mesh=initial_show_mesh,
        initial_colormap=initial_colormap,
        profile_path=profile_path,
        solution_paths=solution_paths,
        initial_solution_index=initial_solution_index,
        initial_show_deformation=initial_show_deformation,
        initial_deformation_scale=initial_deformation_scale,
    )
    plt.show()


def render_comparison(
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
    out_path: Optional[Path] = None,
    inspection_method: str = "nearest",
    show_profile: bool = False,
    show_domain_edge: bool = True,
    colormap: Optional[str] = None,
    contour_levels: int = 12,
    show_isoline_labels: bool = False,
    profile_path: Optional[Sequence[Tuple[float, float]]] = None,
):
    """Render two solutions with one shared scale, interactively or to an image."""
    viewer = ComparisonViewer(
        first=first,
        second=second,
        initial_result=initial_result,
        initial_step=initial_step,
        initial_location=initial_location,
        component=component,
        tensor_component_mode=tensor_component_mode,
        profile_x=profile_x,
        style=style,
        vmin=vmin,
        vmax=vmax,
        inspection_method=inspection_method,
        show_profile=show_profile,
        show_domain_edge=show_domain_edge,
        colormap=colormap,
        contour_levels=contour_levels,
        show_isoline_labels=show_isoline_labels,
        profile_path=profile_path,
    )
    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        viewer.fig.savefig(out_path, dpi=180)
        print(f"Saved comparison plot to: {out_path}")
        plt.close(viewer.fig)
