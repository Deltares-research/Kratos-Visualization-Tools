"""GiD mesh/result readers and active-submodelpart parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
from .models import ActiveSubmodelPart, MESH_HEADER_RE, MeshBlock, RESULT_HEADER_RE, RESULT_LOCATION_ON_GAUSS_POINTS, ResultHeader


def parse_post_msh(msh_path: Path):
    blocks: List[MeshBlock] = []
    all_nodes: Dict[int, Tuple[float, float, float]] = {}

    current: Optional[MeshBlock] = None
    mode = None

    with msh_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            header = MESH_HEADER_RE.match(line)
            if header:
                if current is not None:
                    blocks.append(current)
                current = MeshBlock(
                    name=header.group("name"),
                    elem_type=header.group("etype"),
                    nnode=int(header.group("nnode")),
                    coordinates={},
                    elements=[],
                )
                mode = None
                continue

            if current is None:
                continue

            if line == "Coordinates":
                mode = "coords"
                continue
            if line == "End Coordinates":
                mode = None
                continue
            if line == "Elements":
                mode = "elems"
                continue
            if line == "End Elements":
                mode = None
                continue

            if mode == "coords":
                parts = line.split()
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                current.coordinates[nid] = (x, y, z)
                all_nodes[nid] = (x, y, z)
            elif mode == "elems":
                parts = line.split()
                if len(parts) < 2 + current.nnode:
                    continue
                eid = int(parts[0])
                nodes = [int(v) for v in parts[1 : 1 + current.nnode]]
                prop = int(parts[1 + current.nnode]) if len(parts) > 1 + current.nnode else -1
                current.elements.append((eid, nodes, prop))

    if current is not None:
        blocks.append(current)

    triangles = []
    surface_elements: List[Tuple[int, List[int]]] = []
    line_meshes = []
    mesh_block_nodes: Dict[str, Set[int]] = {}
    for blk in blocks:
        block_nodes = set(blk.coordinates.keys())
        if not block_nodes:
            for _, nodes, _ in blk.elements:
                block_nodes.update(nodes)
        mesh_block_nodes[blk.name] = block_nodes

        elem_type = blk.elem_type.lower()
        if elem_type in ("triangle", "quadrilateral"):
            for eid, nodes, _ in blk.elements:
                if len(nodes) >= 3:
                    surface_elements.append((eid, list(nodes)))

                    # Build plotting triangulation from corner nodes.
                    if len(nodes) in (3, 6, 10):
                        n0, n1, n2 = nodes[0], nodes[1], nodes[2]
                        triangles.append((n0, n1, n2))
                    elif len(nodes) in (4, 8, 9):
                        n0, n1, n2, n3 = nodes[0], nodes[1], nodes[2], nodes[3]
                        triangles.append((n0, n1, n2))
                        triangles.append((n0, n2, n3))
        elif blk.elem_type.lower() == "linear":
            node_set = set()
            for _, nodes, _ in blk.elements:
                node_set.update(nodes)
            line_meshes.append((blk.name, node_set, blk.elements))

    return all_nodes, triangles, surface_elements, line_meshes, mesh_block_nodes


def resolve_active_nodes(
    mesh_block_nodes: Dict[str, Set[int]],
    active_submodelpart: Optional[str],
) -> Optional[Set[int]]:
    if not active_submodelpart:
        return None

    requested = [p.strip() for p in active_submodelpart.split(",") if p.strip()]
    if not requested:
        return None

    selected: Set[int] = set()
    missing: List[str] = []
    for name in requested:
        if name in mesh_block_nodes:
            selected.update(mesh_block_nodes[name])
        else:
            missing.append(name)

    if missing:
        known = sorted(mesh_block_nodes.keys())
        preview = ", ".join(known[:20])
        more = "" if len(known) <= 20 else f" ... (+{len(known) - 20} more)"
        raise ValueError(
            f"Unknown --active submodelpart(s): {', '.join(missing)}. "
            f"Available mesh block names: {preview}{more}"
        )

    return selected


def scan_result_headers(res_path: Path) -> List[ResultHeader]:
    headers: List[ResultHeader] = []

    with res_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line.startswith("Result "):
                continue

            m = RESULT_HEADER_RE.match(line)
            if not m:
                continue

            try:
                step = int(float(m.group("step")))
            except ValueError:
                continue

            headers.append(
                ResultHeader(
                    name=m.group("name"),
                    step=step,
                    result_type=m.group("rtype"),
                    location=m.group("loc"),
                    gauss_points=m.group("gp"),
                    header_line=line_no,
                )
            )

    return headers


def scan_gauss_point_counts(res_path: Path) -> Dict[str, int]:
    counts = {}
    current_name = None

    with res_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("GaussPoints "):
                match = re.match(r'^GaussPoints\s+"([^"]+)"', line)
                current_name = match.group(1) if match else None
            elif current_name is not None and line == "End GaussPoints":
                current_name = None
            elif current_name is not None:
                match = re.match(r"^Number Of Gauss Points:\s*(\d+)", line)
                if match:
                    counts[current_name] = int(match.group(1))

    return counts


def load_values(
    res_path: Path,
    target_name: str,
    target_step: int,
    target_location: str,
) -> Tuple[str, Dict[int, np.ndarray]]:
    """Load nodal values or per-element Gauss-point arrays from a .post.res file."""
    gauss_point_counts = scan_gauss_point_counts(res_path)
    in_target = False
    in_values = False
    result_type = ""
    values: Dict[int, np.ndarray] = {}
    best_result_type = ""
    best_values: Dict[int, np.ndarray] = {}
    gauss_point_count = 1
    current_element_id = None
    current_element_values = []

    def store_pending_gauss_values():
        nonlocal current_element_id, current_element_values
        if current_element_id is not None and current_element_values:
            values[current_element_id] = np.asarray(current_element_values, dtype=float)
        current_element_id = None
        current_element_values = []

    with res_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("Result "):
                in_values = False
                in_target = False

                m = RESULT_HEADER_RE.match(line)
                if not m:
                    continue

                name = m.group("name")
                try:
                    step = int(float(m.group("step")))
                except ValueError:
                    continue
                loc = m.group("loc")

                if name == target_name and step == target_step and loc == target_location:
                    in_target = True
                    result_type = m.group("rtype")
                    values = {}
                    current_element_id = None
                    current_element_values = []
                    gauss_point_count = gauss_point_counts.get(m.group("gp"), 1)
                continue

            if in_target and line == "Values":
                in_values = True
                continue

            if in_target and line == "End Values":
                if target_location == RESULT_LOCATION_ON_GAUSS_POINTS:
                    store_pending_gauss_values()
                    best_result_type = result_type
                    best_values.update(values)
                elif len(values) > len(best_values):
                    best_result_type = result_type
                    best_values = values.copy()
                in_target = False
                in_values = False
                continue

            if in_target and in_values:
                parts = line.split()
                if not parts or (
                    target_location != RESULT_LOCATION_ON_GAUSS_POINTS and len(parts) < 2
                ):
                    continue
                if target_location == RESULT_LOCATION_ON_GAUSS_POINTS:
                    try:
                        if current_element_id is None:
                            current_element_id = int(parts[0])
                            data = [float(v) for v in parts[1:]]
                        else:
                            data = [float(v) for v in parts]
                    except ValueError:
                        store_pending_gauss_values()
                        continue
                    current_element_values.append(data)
                    if len(current_element_values) >= gauss_point_count:
                        store_pending_gauss_values()
                else:
                    try:
                        idx = int(parts[0])
                        data = np.array([float(v) for v in parts[1:]], dtype=float)
                    except ValueError:
                        continue
                    values[idx] = data

    if in_target and target_location == RESULT_LOCATION_ON_GAUSS_POINTS:
        store_pending_gauss_values()
        best_result_type = result_type
        best_values.update(values)
    elif len(values) > len(best_values):
        best_result_type = result_type
        best_values = values.copy()

    return best_result_type, best_values


def detect_profile_nodes(
    all_nodes: Dict[int, Tuple[float, float, float]],
    line_meshes: Sequence[Tuple[str, set, list]],
    profile_x: Optional[float],
    x_tol: float,
) -> set:
    # First gather vertical line meshes (small x-span, large y-span).
    vertical_candidates: List[Tuple[float, set]] = []
    for _, node_set, _ in line_meshes:
        pts = [all_nodes[nid] for nid in node_set if nid in all_nodes]
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        if x_span < x_tol and y_span > 2.0:
            mean_x = float(sum(xs) / len(xs))
            vertical_candidates.append((mean_x, node_set))

    # If a profile x-coordinate is explicitly requested, prefer vertical
    # candidates close to that x. This avoids selecting unrelated line meshes.
    if profile_x is not None and vertical_candidates:
        selected = set()
        for mean_x, node_set in vertical_candidates:
            if abs(mean_x - profile_x) <= 2.0 * x_tol:
                selected.update(node_set)
        if selected:
            return selected

    # Without an explicit profile x-coordinate, use all vertical candidates.
    if profile_x is None and vertical_candidates:
        selected = set()
        for _, node_set in vertical_candidates:
            selected.update(node_set)
        return selected

    # Fallback: geometric slice by x value.
    if profile_x is None:
        return set()

    selected = set()
    for nid, (x, _, _) in all_nodes.items():
        if abs(x - profile_x) <= x_tol:
            selected.add(nid)

    return selected


def parse_submodelpart_entities(mdpa_path: Path, target_submodelpart: str) -> ActiveSubmodelPart:
    node_ids: set = set()
    elem_ids: set = set()
    cond_ids: set = set()

    in_target = False
    mode: Optional[str] = None

    with mdpa_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("Begin SubModelPart "):
                current_name = line[len("Begin SubModelPart ") :].strip()
                in_target = current_name == target_submodelpart
                mode = None
                continue

            if line == "End SubModelPart":
                in_target = False
                mode = None
                continue

            if not in_target:
                continue

            if line == "Begin SubModelPartNodes":
                mode = "nodes"
                continue
            if line == "End SubModelPartNodes" and mode == "nodes":
                mode = None
                continue

            if line == "Begin SubModelPartElements":
                mode = "elems"
                continue
            if line == "End SubModelPartElements" and mode == "elems":
                mode = None
                continue

            if line == "Begin SubModelPartConditions":
                mode = "conds"
                continue
            if line == "End SubModelPartConditions" and mode == "conds":
                mode = None
                continue

            try:
                idx = int(line.split()[0])
            except (ValueError, IndexError):
                continue

            if mode == "nodes":
                node_ids.add(idx)
            elif mode == "elems":
                elem_ids.add(idx)
            elif mode == "conds":
                cond_ids.add(idx)

    return ActiveSubmodelPart(
        name=target_submodelpart,
        node_ids=node_ids,
        entity_ids=elem_ids | cond_ids,
    )
