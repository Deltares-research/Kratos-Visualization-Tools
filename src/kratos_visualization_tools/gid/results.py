"""Solution discovery, result selection, and scalar field loading."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
from .calculations import build_node_arrays, build_scalar_field_from_values, project_element_to_nodes, triangulation_from_connectivity
from .models import ResultHeader, SolutionData
from .parsing import detect_profile_nodes, load_values, parse_post_msh, scan_result_headers


def compare_result_names(first_headers: Sequence[ResultHeader], second_headers: Sequence[ResultHeader]) -> Tuple[List[str], List[str], List[str]]:
    """Return common result names and names unique to each solution."""
    first_names = {header.name for header in first_headers}
    second_names = {header.name for header in second_headers}
    return (
        sorted(first_names & second_names),
        sorted(first_names - second_names),
        sorted(second_names - first_names),
    )


def find_best_location(headers: Sequence[ResultHeader], result_name: str) -> Optional[str]:
    """Return best available location for result: prefer OnNodes, fallback to OnGaussPoints."""
    for loc in ["OnNodes", "OnGaussPoints"]:
        if any(h.name == result_name and h.location == loc for h in headers):
            return loc
    return None


def pick_step(headers: Sequence[ResultHeader], result_name: str, requested_step: str, location: Optional[str]) -> Tuple[int, str]:
    """Pick timestep; auto-detect location if not specified. Returns (step, actual_location)."""
    if location is None:
        location = find_best_location(headers, result_name)
        if location is None:
            raise ValueError(f"No result '{result_name}' found in any location.")
    
    candidates = [h.step for h in headers if h.name == result_name and h.location == location]
    if not candidates:
        available_locs = list(set(h.location for h in headers if h.name == result_name))
        raise ValueError(
            f"No '{result_name}' result found at location '{location}'. "
            f"Available locations: {available_locs}. Try --location {available_locs[0]} if present."
        )

    if requested_step == "latest":
        return max(candidates), location

    try:
        step = int(float(requested_step))
    except ValueError as exc:
        raise ValueError(f"Invalid step '{requested_step}'. Use integer or 'latest'.") from exc

    if step not in set(candidates):
        raise ValueError(f"Step {step} not found for result '{result_name}' at location '{location}'.")
    return step, location


def resolve_default_mdpa_path(name_base: str) -> Optional[Path]:
    base = Path(name_base)
    candidates = [
        Path(f"{name_base}.mdpa"),
        base.parent / "eemdijk.mdpa",
        base.parent / f"{base.name}.mdpa",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def resolve_mesh_result_paths(name_or_path: str, base_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Resolve mesh/result file pair from base name or explicit file path.

    Accepted inputs:
    - base name/path without extension
    - explicit mesh file: .msh or .post.msh
    - explicit result file: .res or .post.res
    """
    raw = Path(name_or_path)
    if not raw.is_absolute() and base_dir is not None:
        raw = (base_dir / raw).resolve()

    raw_str = str(raw)
    lower = raw_str.lower()

    candidates: List[Tuple[Path, Path]] = []

    if lower.endswith('.post.msh'):
        candidates.append((raw, Path(raw_str[:-9] + '.post.res')))
    elif lower.endswith('.msh'):
        candidates.append((raw, Path(raw_str[:-4] + '.res')))
    elif lower.endswith('.post.res'):
        candidates.append((Path(raw_str[:-9] + '.post.msh'), raw))
    elif lower.endswith('.res'):
        candidates.append((Path(raw_str[:-4] + '.msh'), raw))
    else:
        # Prefer GiD .post naming first for backward compatibility.
        candidates.append((Path(raw_str + '.post.msh'), Path(raw_str + '.post.res')))
        candidates.append((Path(raw_str + '.msh'), Path(raw_str + '.res')))

    for msh_path, res_path in candidates:
        if msh_path.exists() and res_path.exists():
            return msh_path, res_path

    # Fall back to first candidate so caller reports deterministic missing file.
    return candidates[0]


def discover_solution_paths(folder: Path) -> List[Tuple[Path, Path]]:
    """Find matching GiD mesh/result pairs in a folder."""
    def solution_sort_key(mesh_path: Path):
        name = mesh_path.stem.lower()
        leading_number = re.match(r"^(\d+)", name)
        if leading_number:
            return (0, int(leading_number.group(1)), name)
        stage_number = re.search(r"stage[_ -]?(\d+)", name)
        if stage_number:
            return (1, int(stage_number.group(1)), name)
        return (2, 0, name)

    pairs = []
    mesh_paths = list(folder.glob("*.post.msh")) + [
        path for path in folder.glob("*.msh") if not path.name.endswith(".post.msh")
    ]
    for mesh_path in sorted(mesh_paths, key=solution_sort_key):
        result_path = mesh_path.with_name(mesh_path.name[:-4] + ".res")
        if result_path.exists():
            pairs.append((mesh_path.resolve(), result_path.resolve()))
    return pairs


def load_solution_data(name_or_path: str, label: str, base_dir: Optional[Path] = None) -> SolutionData:
    """Load mesh, result headers and profile metadata for one GiD solution."""
    msh_path, res_path = resolve_mesh_result_paths(name_or_path, base_dir=base_dir)
    if not msh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {msh_path}")
    if not res_path.exists():
        raise FileNotFoundError(f"Result file not found: {res_path}")

    headers = scan_result_headers(res_path)
    if not headers:
        raise RuntimeError(f"No parsable Result blocks found in {res_path}")

    all_nodes, triangles, surface_elements, line_meshes, _ = parse_post_msh(msh_path)
    profile_nodes = detect_profile_nodes(all_nodes, line_meshes, profile_x=None, x_tol=0.35)
    return SolutionData(
        label=label,
        msh_path=msh_path,
        res_path=res_path,
        headers=headers,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements,
        line_meshes=line_meshes,
        profile_nodes=profile_nodes,
    )


def load_result_values(
    headers: Sequence[ResultHeader],
    res_path: Path,
    result_name: str,
    requested_step: str,
    requested_location: Optional[str],
    surface_elements: Sequence[Tuple[int, List[int]]] = (),
    line_meshes: Sequence[Tuple[str, set, list]] = (),
) -> Tuple[int, str, str, Dict[int, np.ndarray]]:
    """Load result values and project Gauss-point data to nodes when required."""
    try:
        step, location = pick_step(headers, result_name, requested_step, requested_location)
    except ValueError:
        if requested_location is None:
            raise
        step, location = pick_step(headers, result_name, requested_step, None)

    result_type, values = load_values(res_path, result_name, step, location)
    if location == "OnGaussPoints":
        line_elements = []
        for _, _, elements in line_meshes:
            line_elements.extend(elements)
        values = project_element_to_nodes(
            values,
            surface_elements,
            line_elements if line_elements else None,
        )
    return step, location, result_type, values


def load_solution_field(
    solution: SolutionData,
    result_name: str,
    requested_step: str,
    requested_location: Optional[str],
    component: str,
    tensor_component_mode: str,
):
    """Load and reduce one solution's selected result to plottable scalar data."""
    step, location, result_type, values = load_result_values(
        solution.headers,
        solution.res_path,
        result_name,
        requested_step,
        requested_location,
        solution.surface_elements,
        solution.line_meshes,
    )

    node_ids, id_to_pos, points = build_node_arrays(solution.all_nodes)
    scalar = build_scalar_field_from_values(
        values,
        node_ids,
        id_to_pos,
        result_type,
        component,
        tensor_component_mode,
    )
    tri = triangulation_from_connectivity(solution.triangles, id_to_pos, points[:, :2])
    return {
        "step": step,
        "location": location,
        "result_type": result_type,
        "values": values,
        "node_ids": node_ids,
        "points": points,
        "scalar": scalar,
        "tri": tri,
    }


def print_result_summary(headers: Sequence[ResultHeader]):
    grouped = defaultdict(lambda: defaultdict(set))
    for h in headers:
        grouped[(h.name, h.result_type, h.location)][h.gauss_points].add(h.step)

    print("Available results:")
    for (name, rtype, loc), gp_map in sorted(grouped.items(), key=lambda x: x[0][0]):
        for gp, steps in gp_map.items():
            gp_text = f", gp={gp}" if gp else ""
            print(
                f"  - {name:28s} type={rtype:8s} loc={loc:13s}{gp_text} "
                f"steps=[{min(steps)}..{max(steps)}], count={len(steps)}"
            )


def print_comparison_summary(first: SolutionData, second: SolutionData):
    common, first_only, second_only = compare_result_names(first.headers, second.headers)
    print(f"Common variables ({first.label} and {second.label}):")
    print(f"  {', '.join(common) if common else '-'}")
    print(f"Only in {first.label} (not shown in comparison):")
    print(f"  {', '.join(first_only) if first_only else '-'}")
    print(f"Only in {second.label} (not shown in comparison):")
    print(f"  {', '.join(second_only) if second_only else '-'}")
