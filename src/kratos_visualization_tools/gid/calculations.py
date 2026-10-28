"""Scalar reductions, projection, gradients, and profile interpolation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
import matplotlib.tri as mtri


def cycle_sequence(values: Sequence[str], current: Optional[str], step: int = 1) -> str:
    """Return the next value in a sequence, wrapping around at the ends."""
    if not values:
        raise ValueError("Cannot cycle an empty sequence.")
    if current not in values:
        return values[0]
    index = values.index(current)
    return values[(index + step) % len(values)]


def validate_component(component: str, allowed: Sequence[str], name: str = "component") -> str:
    """Validate a user-selected component and raise a clear error on invalid input."""
    if component not in allowed:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"Unsupported {name} '{component}'. Allowed values: {allowed_values}.")
    return component


def calculate_shared_scale(
    fields: Sequence[np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[float, float]:
    """Calculate one color scale from all finite values in both solutions."""
    finite_values = [values[np.isfinite(values)] for values in fields if np.any(np.isfinite(values))]
    if not finite_values:
        return (0.0, 1.0) if vmin is None or vmax is None else (vmin, vmax)

    combined = np.concatenate(finite_values)
    lower = float(np.nanmin(combined)) if vmin is None else float(vmin)
    upper = float(np.nanmax(combined)) if vmax is None else float(vmax)
    if lower == upper:
        upper = lower + 1e-12
    if lower > upper:
        raise ValueError("Shared scale minimum must not exceed its maximum.")
    return lower, upper


def interpolate_scalar_field(source_field: Dict[str, object], target_points: np.ndarray) -> np.ndarray:
    """Interpolate a scalar field onto target points with inverse-distance weighting."""
    source_points = np.asarray(source_field["points"], dtype=float)
    source_values = np.asarray(source_field["scalar"], dtype=float)
    result = np.full(len(target_points), np.nan, dtype=float)
    valid = np.isfinite(source_values)
    if not np.any(valid):
        return result

    source_points = source_points[valid, :2]
    source_values = source_values[valid]
    for index, point in enumerate(np.asarray(target_points, dtype=float)[:, :2]):
        distances_squared = np.sum((source_points - point) ** 2, axis=1)
        nearest = int(np.argmin(distances_squared))
        if distances_squared[nearest] < 1e-16:
            result[index] = source_values[nearest]
            continue

        count = min(8, len(source_values))
        selected = np.argpartition(distances_squared, count - 1)[:count]
        weights = 1.0 / (distances_squared[selected] + 1e-12)
        result[index] = float(np.sum(weights * source_values[selected]) / np.sum(weights))
    return result


def calculate_symmetric_scale(values: np.ndarray) -> Tuple[float, float]:
    """Return a color scale centered on zero for signed difference values."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return -1.0, 1.0
    limit = float(np.max(np.abs(finite_values)))
    if limit == 0.0:
        limit = 1e-12
    return -limit, limit


def project_element_to_nodes(
    gp_values: Dict[int, np.ndarray],
    surface_elements: Sequence[Tuple[int, List[int]]],
    line_elements: Optional[List[Tuple[int, List[int], int]]],
) -> Dict[int, np.ndarray]:
    """Extrapolate Gauss-point values to element nodes and average shared nodes."""
    node_accum = defaultdict(lambda: [np.array([]), 0])  # [sum_array, count]

    def extrapolate(values: np.ndarray, number_of_nodes: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return np.repeat(values[np.newaxis, :], number_of_nodes, axis=0)

        number_of_points = values.shape[0]
        shape_matrix = None
        if number_of_nodes == 2 and number_of_points == 2:
            offset = 1.0 / math.sqrt(3.0)
            shape_matrix = np.array([
                [(1.0 + offset) / 2.0, (1.0 - offset) / 2.0],
                [(1.0 - offset) / 2.0, (1.0 + offset) / 2.0],
            ])
        elif number_of_nodes == 3 and number_of_points == 3:
            shape_matrix = np.array([
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ])
        elif number_of_nodes == 4 and number_of_points == 4:
            offset = 1.0 / math.sqrt(3.0)
            gauss_coordinates = [
                (-offset, -offset),
                (offset, -offset),
                (offset, offset),
                (-offset, offset),
            ]
            shape_matrix = np.array([
                [
                    (1.0 - x) * (1.0 - y) / 4.0,
                    (1.0 + x) * (1.0 - y) / 4.0,
                    (1.0 + x) * (1.0 + y) / 4.0,
                    (1.0 - x) * (1.0 + y) / 4.0,
                ]
                for x, y in gauss_coordinates
            ])

        if shape_matrix is not None:
            try:
                return np.linalg.solve(shape_matrix, values)
            except np.linalg.LinAlgError:
                pass

        mean_value = np.nanmean(values, axis=0)
        return np.repeat(mean_value[np.newaxis, :], number_of_nodes, axis=0)

    for elem_id, elem_nodes in surface_elements:
        if elem_id not in gp_values:
            continue
        nodal_values = extrapolate(gp_values[elem_id], len(elem_nodes))
        for nid, val in zip(elem_nodes, nodal_values):
            prev_sum, count = node_accum[nid]
            if count == 0:
                node_accum[nid] = [val.copy(), 1]
            elif prev_sum.shape == val.shape:
                node_accum[nid] = [prev_sum + val, count + 1]

    if line_elements:
        for elem_id, nodes, _ in line_elements:
            if elem_id not in gp_values or len(nodes) < 2:
                continue
            nodal_values = extrapolate(gp_values[elem_id], min(len(nodes), 2))
            for nid, val in zip(nodes[:2], nodal_values):
                prev_sum, count = node_accum[nid]
                if count == 0:
                    node_accum[nid] = [val.copy(), 1]
                elif prev_sum.shape == val.shape:
                    node_accum[nid] = [prev_sum + val, count + 1]

    result = {}
    for nid, (total, count) in node_accum.items():
        if count > 0:
            result[nid] = total / count

    return result


def vector_component(arr: np.ndarray, component: str) -> float:
    if arr.size == 0:
        return float("nan")

    comp = component.lower()
    if comp == "x":
        return arr[0] if arr.size > 0 else float("nan")
    if comp == "y":
        return arr[1] if arr.size > 1 else float("nan")
    if comp == "z":
        return arr[2] if arr.size > 2 else float("nan")
    if comp == "magnitude":
        return float(np.linalg.norm(arr[: min(3, arr.size)]))

    raise ValueError(f"Unsupported component '{component}'.")


def tensor_component(arr: np.ndarray, component: str) -> float:
    if arr.size == 0:
        return float("nan")

    comp = component.lower()
    flat = np.asarray(arr, dtype=float).ravel()
    if flat.size == 6:
        xx, yy, zz, xy, yz, xz = flat
        if comp == "xx":
            return float(xx)
        if comp in ("xy", "yx"):
            return float(xy)
        if comp == "yy":
            return float(yy)
        if comp == "zz":
            return float(zz)
        if comp in ("yz", "zy"):
            return float(yz)
        if comp in ("xz", "zx"):
            return float(xz)
        if comp == "trace":
            return float(xx + yy + zz)
        if comp == "magnitude":
            return float(np.linalg.norm(flat))

    dim = int(round(math.sqrt(float(flat.size))))
    matrix = flat.reshape((dim, dim)) if dim > 0 and dim * dim == flat.size else flat

    if comp == "xx":
        return float(matrix[0, 0]) if matrix.ndim == 2 else float(flat[0])
    if comp == "xy":
        return float(matrix[0, 1]) if matrix.ndim == 2 and matrix.shape[1] > 1 else float(flat[1]) if flat.size > 1 else float("nan")
    if comp == "yx":
        return float(matrix[1, 0]) if matrix.ndim == 2 and matrix.shape[0] > 1 else float(flat[2]) if flat.size > 2 else float("nan")
    if comp == "yy":
        if matrix.ndim == 2 and matrix.shape[0] > 1 and matrix.shape[1] > 1:
            return float(matrix[1, 1])
        return float(flat[3]) if flat.size > 3 else float("nan")
    if comp == "zz":
        if matrix.ndim == 2 and matrix.shape[0] > 2 and matrix.shape[1] > 2:
            return float(matrix[2, 2])
        return float(flat[2]) if flat.size > 2 else float("nan")
    if comp == "trace":
        if matrix.ndim == 2 and matrix.shape[0] > 1 and matrix.shape[1] > 1:
            return float(np.trace(matrix))
        if flat.size >= 4:
            return float(flat[0] + flat[3])
        if flat.size >= 3:
            return float(np.sum(flat[:3]))
        return float("nan")
    if comp == "magnitude":
        return float(np.linalg.norm(flat))

    raise ValueError(f"Unsupported tensor component '{component}'.")


def field_scalar_value(arr: np.ndarray, result_type: str, component: str, tensor_component_mode: str) -> float:
    rtype = result_type.lower()
    if rtype == "vector":
        return vector_component(arr, component)
    if rtype == "matrix":
        return tensor_component(arr, tensor_component_mode)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def reduce_result_value(arr: np.ndarray, result_type: str, component: str, tensor_component_mode: str) -> float:
    """Convert a raw result array to a scalar value used for plotting or export."""
    rtype = result_type.lower()
    if rtype == "vector":
        return vector_component(arr, component)
    if rtype == "matrix":
        return field_scalar_value(arr, result_type, component, tensor_component_mode)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def build_scalar_field_from_values(
    values: Dict[int, np.ndarray],
    node_ids: Sequence[int],
    id_to_pos: Dict[int, int],
    result_type: str,
    component: str,
    tensor_component_mode: str,
) -> np.ndarray:
    """Create a nodal scalar field from raw result arrays without any plotting concerns."""
    scalar = np.full(len(node_ids), np.nan, dtype=float)
    for node_id, arr in values.items():
        pos = id_to_pos.get(node_id)
        if pos is None:
            continue
        scalar[pos] = reduce_result_value(arr, result_type, component, tensor_component_mode)
    return scalar


def finite_difference_gradient(coords: Sequence[float], values: Sequence[float]) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=float)
    values_arr = np.asarray(values, dtype=float)

    if coords_arr.shape != values_arr.shape:
        raise ValueError("Coordinate and value arrays must have the same shape.")

    gradients = np.full(values_arr.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(coords_arr) & np.isfinite(values_arr)
    if np.count_nonzero(finite_mask) < 2:
        return gradients

    finite_indices = np.flatnonzero(finite_mask)
    split_points = np.where(np.diff(finite_indices) != 1)[0] + 1
    contiguous_runs = np.split(finite_indices, split_points)

    for run in contiguous_runs:
        if run.size < 2:
            continue

        run_coords = coords_arr[run]
        run_values = values_arr[run]

        order = np.argsort(run_coords)
        sorted_coords = run_coords[order]
        sorted_values = run_values[order]

        unique_coords, inverse = np.unique(sorted_coords, return_inverse=True)
        if unique_coords.size < 2:
            continue

        unique_values = np.zeros(unique_coords.shape, dtype=float)
        counts = np.zeros(unique_coords.shape, dtype=int)
        for idx, group_idx in enumerate(inverse):
            unique_values[group_idx] += sorted_values[idx]
            counts[group_idx] += 1
        unique_values /= counts

        if unique_coords.size == 2:
            delta = unique_coords[1] - unique_coords[0]
            if delta == 0.0:
                continue
            unique_gradients = np.array([
                (unique_values[1] - unique_values[0]) / delta,
                (unique_values[1] - unique_values[0]) / delta,
            ], dtype=float)
        else:
            unique_gradients = np.gradient(unique_values, unique_coords, edge_order=1)

        sorted_gradients = unique_gradients[inverse]
        run_gradients = np.empty_like(sorted_gradients)
        run_gradients[order] = sorted_gradients
        gradients[run] = run_gradients

    return gradients


def compute_nodal_gradient(
    tri: Optional[mtri.Triangulation],
    nodal_scalar: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if tri is None or tri.triangles.size == 0:
        return np.full_like(nodal_scalar, np.nan, dtype=float), np.full_like(nodal_scalar, np.nan, dtype=float)

    grad_x = np.full_like(nodal_scalar, np.nan, dtype=float)
    grad_y = np.full_like(nodal_scalar, np.nan, dtype=float)
    weight = np.zeros_like(nodal_scalar, dtype=float)

    points = np.column_stack((np.asarray(tri.x, dtype=float), np.asarray(tri.y, dtype=float)))

    for tri_nodes in np.asarray(tri.triangles, dtype=int):
        vals = nodal_scalar[tri_nodes]
        if not np.all(np.isfinite(vals)):
            continue

        pts = points[tri_nodes]
        mat = np.array(
            [
                [pts[0, 0], pts[0, 1], 1.0],
                [pts[1, 0], pts[1, 1], 1.0],
                [pts[2, 0], pts[2, 1], 1.0],
            ],
            dtype=float,
        )

        try:
            coeffs = np.linalg.solve(mat, vals)
        except np.linalg.LinAlgError:
            continue

        area = 0.5 * abs(
            (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
            - (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1])
        )
        if not np.isfinite(area) or area <= 0.0:
            continue

        for node_idx in tri_nodes:
            if np.isnan(grad_x[node_idx]):
                grad_x[node_idx] = 0.0
                grad_y[node_idx] = 0.0
            grad_x[node_idx] += coeffs[0] * area
            grad_y[node_idx] += coeffs[1] * area
            weight[node_idx] += area

    valid = weight > 0.0
    grad_x[valid] /= weight[valid]
    grad_y[valid] /= weight[valid]
    grad_x[~valid] = np.nan
    grad_y[~valid] = np.nan
    return grad_x, grad_y


def build_node_arrays(all_nodes: Dict[int, Tuple[float, float, float]]):
    sorted_ids = sorted(all_nodes.keys())
    id_to_pos = {nid: i for i, nid in enumerate(sorted_ids)}
    pts = np.array([all_nodes[nid] for nid in sorted_ids], dtype=float)
    return sorted_ids, id_to_pos, pts


def sample_vertical_profile_from_triangulation(
    tri: mtri.Triangulation,
    nodal_scalar: np.ndarray,
    profile_x: float,
    y_min: float,
    y_max: float,
    n_samples: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a vertical profile from triangulated nodal data.

    Values outside the triangulated domain (void area) are set to NaN.
    """
    ys = np.linspace(float(y_min), float(y_max), int(n_samples))
    xs = np.full_like(ys, float(profile_x), dtype=float)

    tri_finder = tri.get_trifinder()
    tri_ids = tri_finder(xs, ys)

    scalar_masked = np.ma.masked_invalid(nodal_scalar)
    interpolator = mtri.LinearTriInterpolator(tri, scalar_masked)
    sampled = interpolator(xs, ys)

    if np.ma.isMaskedArray(sampled):
        values = np.asarray(sampled.filled(np.nan), dtype=float)
    else:
        values = np.asarray(sampled, dtype=float)

    values[tri_ids < 0] = np.nan
    return ys, values


def sample_profile_path_from_triangulation(
    tri: mtri.Triangulation,
    nodal_scalar: np.ndarray,
    path_points: Sequence[Tuple[float, float]],
    n_samples: int = 1200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample nodal data along a line segment or connected polyline."""
    points = np.asarray(path_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("A profile path must contain at least two x,y points.")

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0.0:
        raise ValueError("A profile path must have non-zero length.")

    distances = np.linspace(0.0, total_length, int(n_samples))
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_indices = np.searchsorted(cumulative_lengths, distances, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(segment_lengths) - 1)
    local_lengths = distances - cumulative_lengths[segment_indices]
    fractions = local_lengths / segment_lengths[segment_indices]
    sampled_points = points[segment_indices] + (
        segment_vectors[segment_indices] * fractions[:, None]
    )

    tri_finder = tri.get_trifinder()
    tri_ids = tri_finder(sampled_points[:, 0], sampled_points[:, 1])
    scalar_masked = np.ma.masked_invalid(nodal_scalar)
    interpolator = mtri.LinearTriInterpolator(tri, scalar_masked)
    sampled = interpolator(sampled_points[:, 0], sampled_points[:, 1])
    if np.ma.isMaskedArray(sampled):
        values = np.asarray(sampled.filled(np.nan), dtype=float)
    else:
        values = np.asarray(sampled, dtype=float)
    values[tri_ids < 0] = np.nan
    return distances, values, sampled_points


def triangulation_from_connectivity(
    triangles: Sequence[Tuple[int, int, int]],
    id_to_pos: Dict[int, int],
    points_xy: np.ndarray,
):
    tri_idx = []
    for n1, n2, n3 in triangles:
        if n1 in id_to_pos and n2 in id_to_pos and n3 in id_to_pos:
            tri_idx.append((id_to_pos[n1], id_to_pos[n2], id_to_pos[n3]))

    if not tri_idx:
        return None

    tri_arr = np.array(tri_idx, dtype=int)
    return mtri.Triangulation(points_xy[:, 0], points_xy[:, 1], tri_arr)
